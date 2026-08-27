# Modified by the Minchul Kang (2026).
# Changes from the original petuum/adaptdl (Apache-2.0): added process_group
# support and ADAPTDL_GNS_* environment-variable options.
# Original work: Copyright 2020 Petuum, Inc. Licensed under the Apache License,
# Version 2.0. See utils/adaptdl/LICENSE.

import functools
import logging
import math
import os
import numpy as np
import torch.distributed
import torch.optim

from torch.autograd import Variable

import adaptdl.utils

__all__ = ["GradientNoiseScale"]

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

_GNS_DEBUG = os.getenv("ADAPTDL_GNS_DEBUG", "0") == "1"
_GNS_SMOOTHING_ENV = os.getenv("ADAPTDL_GNS_SMOOTHING", "").strip()
_GNS_STRICT_SYNC = os.getenv("ADAPTDL_GNS_STRICT_SYNC", "0") == "1"
# Optional flag to skip resetting GNS stats when switching from biased->unbiased.
# Default is False to preserve existing (GPT) behavior. Enable explicitly for
# BERT runs that want to avoid continuity breaks.
_GNS_SKIP_RESET = os.getenv("ADAPTDL_GNS_SKIP_RESET", "0") == "1"


def _average_groups(grads1, grads2):
    ret = []
    for group1, group2 in zip(grads1, grads2):
        ret.append([])
        for g1, g2 in zip(group1, group2):
            if g1 is None:
                ret[-1].append(g2)
            elif g2 is None:
                ret[-1].append(g1)
            else:
                ret[-1].append((g1 + g2) / 2)
    return ret


def _normsqr_groups(grads, pinvs):
    ret = []
    for group, pinv_group in zip(grads, pinvs):
        normsqr = [(g / pinv).pow(2).sum(dtype=torch.float64)
                   for g, pinv in zip(group, pinv_group) if g is not None]
        ret.append(sum(normsqr).item() if normsqr else 0.0)
    return np.array(ret)


class GradientNoiseScale(object):
    """This class tracks gradient related stats and takes care of gradient
    accumulation."""
    def __init__(self, adp, optimizer,
                 mp_scaler=None,
                 num_replicas=None,
                 accum_scale=None,
                 process_group=None):
        self._adp = adp
        self._optimizer = optimizer
        self._orig_optimizer_zero_grad = optimizer.zero_grad
        self._should_zero_grad = True
        self._mp_scaler = mp_scaler
        self._local_sqr = None
        self._process_group = process_group
        if num_replicas is not None:
            self._num_replicas = num_replicas
        else:
            # Default to the size of the provided process group (if any),
            # otherwise fall back to the global world size.
            try:
                self._num_replicas = torch.distributed.get_world_size(group=process_group)
            except Exception:
                self._num_replicas = torch.distributed.get_world_size()
        self._accum_scale = accum_scale or self._num_replicas
        self._prev_grads = None

        self.reset_accumulation()

        self._optimizer.state.setdefault("gns", {
            "progress": 0.0,
            "prev_scale": 0.0,
            # Averages of n and v
            "sqr_avg": np.ones(len(optimizer.param_groups)),
            "var_avg": np.zeros(len(optimizer.param_groups)),
            # Whether estimates are biased (using differenced estimator).
            "biased": False,
        })

        for idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                param.register_hook(
                    functools.partial(self._backward_hook, idx, param))
        self._callback_queued = False
        # Exponential moving average smoothing factor for GNS estimates.
        # Keep upstream default (0.999), but allow overriding for benchmarking /
        # diagnosis to verify whether low efficiency is an estimator artifact.
        # This does NOT change the underlying GNS/efficiency formula.
        try:
            if _GNS_SMOOTHING_ENV:
                self._smoothing = float(_GNS_SMOOTHING_ENV)
            else:
                self._smoothing = 0.999
        except Exception:
            self._smoothing = 0.999
        # Clamp to a sane range.
        self._smoothing = float(min(max(self._smoothing, 0.0), 0.999999))
        self._debug_seen_backward = False
        self._debug_num_final = 0

    @property
    def _state(self):
        return self._optimizer.state["gns"]

    def reset_accumulation(self):
        """reset accumulation calculations and gradients."""
        self._orig_optimizer_zero_grad()
        self._local_sqr = None
        self._accum_count = 0

    @property
    def should_zero_grad(self):
        return self._should_zero_grad

    @property
    def accum_scale(self):
        return self._accum_scale

    @property
    def accum_count(self):
        return self._accum_count

    def set_accum_scale(self, accum_scale):
        if not np.isclose(self._accum_scale, accum_scale):
            self.reset_accumulation()
            self._accum_scale = accum_scale

    @property
    def raw_sqr_avg(self):
        view = self._state["sqr_avg"].view()
        view.flags.writeable = False
        return view

    def sqr_avg(self):
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared).

        Returns (float): Estimate of squared l2-norm.
        """
        return float(np.sum(np.maximum(self._state["sqr_avg"], 0.0)))

    @property
    def raw_var_avg(self):
        view = self._state["var_avg"].view()
        view.flags.writeable = False
        return view

    def var_avg(self):
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared).

        Returns (float): Estimate of trace of the covariance.
        """
        return float(np.sum(np.maximum(self._state["var_avg"], 1e-6)))

    def get_progress(self):
        return self._state["progress"]

    def set_progress(self, progress):
        self._state["progress"] = progress

    def gain(self, scale):
        """
        Current estimate of the GradientNoiseScale gain ratio.

        Arguments:
            scale (float): The total scale to estimate the gain ratio for.

        Returns (float): Estimate of gain ratio.
        """
        var = self.var_avg()
        norm = self.sqr_avg()
        return (var + norm) / (var / scale + norm)

    def _update_avg(self, param_name, value, factor):
        biased = self._state.get(param_name + "_biased", 0.0)
        unbias = self._state.get(param_name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self._state[param_name + "_biased"] = biased
        self._state[param_name + "_unbias"] = unbias
        self._state[param_name] = biased / unbias

    def _reset_avg(self, param_name):
        self._state.pop(param_name + "_biased", None)
        self._state.pop(param_name + "_unbias", None)

    @adaptdl.utils.print_exc
    def _backward_hook(self, idx, param, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if self._local_sqr is None:
            self._local_sqr = torch.zeros(len(self._optimizer.param_groups),
                                          device=grad.device,
                                          dtype=torch.float64)

        # Get the preconditioning matrix for the optimizer
        preconditioner = self._calculate_preconditioner(idx, param)

        # Update the local gradient square sum
        # IMPORTANT: Cast to float32 before squaring to avoid fp16 overflow
        # (common with mixed precision and/or loss scaling).
        g = grad.detach().float()
        p = preconditioner.float()
        self._local_sqr[idx] += (g / p).pow(2).sum(dtype=torch.float64)
        if _GNS_DEBUG and not self._debug_seen_backward:
            self._debug_seen_backward = True
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                rank = -1
            LOG.warning(f"[ADAPTDL_GNS_DEBUG][rank{rank}] first backward hook fired (idx={idx})")
        if not self._callback_queued:
            Variable._execution_engine.queue_callback(self._queue_callback)
        self._callback_queued = True

    @adaptdl.utils.print_exc
    def _queue_callback(self):
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each replica. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        self._callback_queued = False
        self._accum_count += 1
        # Compute boundary once. NOTE: accessing `require_backward_grad_sync` may
        # have side-effects (e.g., internal counters) in some shims.
        try:
            req_sync = bool(self._adp.require_backward_grad_sync)
        except Exception:
            req_sync = False

        if _GNS_DEBUG:
            try:
                rank = torch.distributed.get_rank()
            except Exception:
                rank = -1
            LOG.warning(
                f"[ADAPTDL_GNS_DEBUG][rank{rank}] queue_callback: accum_count={self._accum_count} "
                f"require_backward_grad_sync={req_sync} num_replicas={self._num_replicas}"
            )
        if req_sync:
            # Asynchronously sum the local squared-gradient statistics. The
            # actual gradient averaging should also be happening at the same
            # time, until self._final_callback is invoked.
            if self._num_replicas > 1:
                if self._process_group is not None:
                    self._async_op = torch.distributed.all_reduce(
                        self._local_sqr, async_op=True, group=self._process_group)
                else:
                    self._async_op = torch.distributed.all_reduce(
                        self._local_sqr, async_op=True)
            Variable._execution_engine.queue_callback(self._final_callback)
            self._should_zero_grad = True
        else:
            # Keep on accumulating gradients, should not zero grad.
            self._should_zero_grad = False

    @adaptdl.utils.print_exc
    def _final_callback(self):
        # This method should be invoked once the gradients have been
        # synchronized between all replicas and accumulation steps.
        if self._num_replicas > 1:
            self._async_op.wait()
        grads = []
        if self._mp_scaler is not None:
            mixed_precision_scale = self._mp_scaler.get_scale()
        else:
            mixed_precision_scale = 1.0
        for group in self._optimizer.param_groups:
            grads.append([])
            for param in group["params"]:
                if param.grad is None:
                    grads[-1].append(None)
                    continue
                # IMPORTANT: Avoid mutating gradients in-place. In AdaptDL this is
                # safe because it owns the accumulation semantics, but in other
                # training stacks (e.g., DeepSpeed) mutating `param.grad` can
                # affect training correctness. We compute the averaged gradient
                # for statistics without modifying `param.grad`.
                g = param.grad.detach()
                if self._accum_count != 1:
                    g = g / float(self._accum_count)
                g = g.float() / mixed_precision_scale
                # Some training stacks (notably DeepSpeed pipeline/ZeRO variants)
                # may not have synchronized `param.grad` across data-parallel
                # replicas at the time this callback runs. AdaptDL's estimator
                # assumes `total_sqr` is computed from the averaged gradient.
                #
                # When enabled, we explicitly all-reduce a copy of the gradient
                # tensor in the provided process_group to obtain the true
                # averaged gradient for statistics. This keeps the definition
                # intact and avoids estimator artifacts like efficiency≈1/scale
                # due to inconsistent `total_sqr` across ranks.
                if _GNS_STRICT_SYNC and self._num_replicas > 1:
                    try:
                        gg = g.clone()
                        if self._process_group is not None:
                            torch.distributed.all_reduce(gg, group=self._process_group)
                        else:
                            torch.distributed.all_reduce(gg)
                        gg /= float(self._num_replicas)
                        g = gg
                    except Exception:
                        # Fall back to local gradient if sync is unavailable.
                        pass
                grads[-1].append(g)
        preconditioner = self._get_preconditioner()

        # Note: mixed precision can result in nan/inf gradients,
        # which propogate into our norm and variance estimates.
        # Mixed precision autoscaling skips the skip where
        # there are nan/inf, so we also skip the update here
        grads_normsqr = _normsqr_groups(grads, preconditioner)
        if not np.all(np.isfinite(grads_normsqr)):
            LOG.warning(f"GradientNoiseScale detected invalid gradient! "
                        f"at scale {mixed_precision_scale}, Skipping step.")
            return
        count = self._num_replicas * self._accum_count
        scale = self._accum_scale * self._accum_count
        if count > 1:
            # Average local squared-norm samples.
            local_sqr = self._local_sqr.cpu().numpy() / count
            # Gradient is squared in local_sqr, so need to square the
            # mixed precision scale as well
            local_sqr = (local_sqr / mixed_precision_scale ** 2)
            total_sqr = grads_normsqr
            if self._state["biased"]:
                # Default (GPT-safe) behavior: reset stats unless explicitly disabled.
                if not _GNS_SKIP_RESET:
                    self._reset_avg("sqr_avg")
                    self._reset_avg("var_avg")
            self._state["biased"] = False
            self._prev_grads = None
            if _GNS_DEBUG:
                try:
                    rank = torch.distributed.get_rank()
                except Exception:
                    rank = -1
                try:
                    sum_local = float(np.sum(local_sqr))
                    sum_total = float(np.sum(total_sqr))
                    ratio = sum_local / (sum_total + 1e-12)
                except Exception:
                    sum_local, sum_total, ratio = float("nan"), float("nan"), float("nan")
                LOG.warning(
                    f"[ADAPTDL_GNS_DEBUG][rank{rank}] norms: "
                    f"accum_count={self._accum_count} accum_scale={self._accum_scale} "
                    f"count={count} scale={scale} "
                    f"sum_local_sqr={sum_local:.6e} sum_total_sqr={sum_total:.6e} ratio={ratio:.6f}"
                )
        else:
            # Single gradient datapoint, use difference estimation.
            if self._prev_grads is not None:
                local_sqr = (_normsqr_groups(self._prev_grads, preconditioner)
                             + grads_normsqr) / 2
                avg_grads = _average_groups(grads, self._prev_grads)
                total_sqr = _normsqr_groups(avg_grads, preconditioner)
                count = 2
                scale = 2 * self._accum_scale
            self._state["biased"] = True
            self._prev_grads = [[g.clone() if g is not None else None
                                 for g in group] for group in grads]
        if count > 1:
            grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
            grad_var = (local_sqr - total_sqr) * scale / (count - 1)
            theta = self._smoothing ** scale
            self._update_avg('sqr_avg', grad_sqr, theta)
            self._update_avg('var_avg', grad_var, theta)
            if _GNS_DEBUG:
                self._debug_num_final += 1
                try:
                    rank = torch.distributed.get_rank()
                except Exception:
                    rank = -1
                LOG.warning(
                    f"[ADAPTDL_GNS_DEBUG][rank{rank}] final_callback #{self._debug_num_final}: "
                    f"count={count} scale={scale} grad_sqr={np.sum(grad_sqr):.6e} grad_var={np.sum(grad_var):.6e}"
                )

    def _get_preconditioner(self):
        out = []
        for idx, group in enumerate(self._optimizer.param_groups):
            pinvs = []
            for param in group["params"]:
                pinv = self._calculate_preconditioner(idx, param)
                pinvs.append(pinv)
            out.append(pinvs)
        return out

    def _calculate_preconditioner(self, idx, param):
        return torch.ones_like(param, memory_format=torch.preserve_format)


class AdamGradientNoiseScale(GradientNoiseScale):
    def __init__(self, adp, optimizer,
                 mp_scaler=None,
                 num_replicas=None,
                 accum_scale=None,
                 process_group=None):
        self._adam_param_group = {'beta': [], 'eps': []}
        super().__init__(adp, optimizer, mp_scaler, num_replicas, accum_scale,
                         process_group=process_group)
        for idx, param_group in enumerate(self._optimizer.param_groups):
            self._adam_param_group['beta'].append(param_group['betas'][1])
            self._adam_param_group['eps'].append(param_group['eps'])

    def _calculate_preconditioner(self, idx, param):
        # Some training stacks (e.g., DeepSpeed ZeRO / parameter flattening)
        # may not expose Adam state keyed by the exact Parameter object we
        # registered hooks on. In that case, fall back to identity
        # preconditioning for that parameter.
        state = self._optimizer.state.get(param, None)
        if state is None:
            return torch.ones_like(param, memory_format=torch.preserve_format)
        if state.get('step', 0) < 5:
            return torch.ones_like(param, memory_format=torch.preserve_format)

        exp_avg_sq = state["exp_avg_sq"].clone()  # not sure if clone is needed
        beta2 = self._adam_param_group['beta'][idx]
        eps = self._adam_param_group['eps'][idx]
        correction = 1 - beta2 ** state['step']
        pinv = (exp_avg_sq.sqrt() / math.sqrt(correction)).add_(eps)
        return pinv.to(param.device)

    def _reset_adam_state(self, step=0):
        for group in self._optimizer.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                state = self._optimizer.state.get(param, None)
                if not state:
                    continue
                if state.get("step", 0) > 0:
                    state["exp_avg"].mul_(
                        (1 - beta1 ** step) / (1 - beta1 ** state["step"]))
                    state["exp_avg_sq"].mul_(
                        (1 - beta2 ** step) / (1 - beta2 ** state["step"]))
                    state["step"] = step

    def _final_callback(self):
        scale = self._accum_scale * self._accum_count
        if not np.isclose(scale, self._state["prev_scale"]):
            # reset Adam states when scale is changed
            self._reset_adam_state()
            self._state["prev_scale"] = scale
        return super()._final_callback()
