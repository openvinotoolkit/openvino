# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from functools import wraps
from threading import RLock, local


_capture_lock = RLock()
_capture_state = local()
_random_prefixes = (
    "rand", "normal", "bernoulli", "multinomial", "dropout",
    "native_dropout", "_fused_dropout", "poisson", "rrelu",
)


def check_vmap_call(target, args=(), kwargs=None):
    """Reject unsupported original operations before batching or functionalization."""
    kwargs = kwargs or {}
    schema = getattr(target, "_schema", None)
    name = schema.name.split("::")[-1] if schema is not None else getattr(target, "__name__", "")
    if name == "_vmap_increment_nesting":
        randomness = args[1] if len(args) > 1 else kwargs.get("randomness", "error")
        if randomness != "error":
            raise RuntimeError("vmap supports only randomness='error'.")
    if (schema is not None and schema.is_mutable) or kwargs.get("out") is not None or (
        name.endswith("_") and not name.endswith("__")
    ) or name == "__setitem__":
        raise RuntimeError("Mutation inside vmap is not supported.")
    if "Generator" in name or "manual_seed" in name:
        raise RuntimeError("Random generators inside vmap are not supported.")
    if name.startswith(_random_prefixes):
        raise RuntimeError("Random operations inside vmap are not supported.")


@contextmanager
def capture_vmap():
    """Capture vectorized ATen operations during an existing export trace."""
    try:
        from torch._C._functorch import _unwrap_batched
        from torch._functorch import apis, vmap
        from torch.fx.experimental.proxy_tensor import (
            ProxyTorchDispatchMode,
            _temp_remove_pre_dispatch_torch_function_mode,
            get_proxy_mode,
        )
        from torch.overrides import TorchFunctionMode
        original = apis.vmap_impl
        original_remove = vmap._remove_batch_dim
    except (ImportError, AttributeError):
        yield
        return

    class VmapFunctionMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            check_vmap_call(func, args, kwargs)
            return func(*args, **kwargs)

    @wraps(original_remove)
    def remove_batch_dim(value, level, batch_size, out_dim):
        if getattr(_capture_state, "in_vmap", False):
            rank = value.dim() + 1
            out_dim = out_dim + rank if out_dim < 0 else out_dim
            if out_dim < 0 or out_dim >= rank:
                raise IndexError("vmap output dimension is out of range.")
            if _unwrap_batched(value, level)[1] is None:
                # Torch 2.11 mishandles nonzero output axes for invariant tensors.
                shape = list(value.shape)
                shape.insert(out_dim, batch_size)
                return value.unsqueeze(out_dim).expand(shape)
        return original_remove(value, level, batch_size, out_dim)

    @wraps(original)
    def traced_vmap(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs):
        mode = get_proxy_mode() if getattr(_capture_state, "enabled", False) else None
        if mode is not None and (getattr(mode, "pre_dispatch", False) or
                                 getattr(_capture_state, "in_vmap", False)) and randomness != "error":
            raise RuntimeError("vmap supports only randomness='error'.")
        if mode is None or not getattr(mode, "pre_dispatch", False):
            return original(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs)
        try:
            region_mode = ProxyTorchDispatchMode(
                mode.tracer,
                mode.tracing_mode,
                pre_dispatch=False,
                _allow_fake_constant=mode._allow_fake_constant,
                _error_on_data_dependent_ops=mode._error_on_data_dependent_ops,
            )
        except (TypeError, AttributeError):
            return original(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs)
        in_vmap = getattr(_capture_state, "in_vmap", False)
        _capture_state.in_vmap = True
        try:
            with _temp_remove_pre_dispatch_torch_function_mode(), region_mode, VmapFunctionMode():
                return original(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs)
        finally:
            _capture_state.in_vmap = in_vmap

    with _capture_lock:
        original = apis.vmap_impl
        original_remove = vmap._remove_batch_dim
        enabled = getattr(_capture_state, "enabled", False)
        _capture_state.enabled = True
        apis.vmap_impl = traced_vmap
        vmap._remove_batch_dim = remove_batch_dim
        try:
            yield
        finally:
            apis.vmap_impl = original
            vmap._remove_batch_dim = original_remove
            _capture_state.enabled = enabled
