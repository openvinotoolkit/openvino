# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

import functools
import logging
import threading
import torch
from openvino.frontend.pytorch import ModuleExtension

log = logging.getLogger(__name__)


def patch_model(model, module_extensions, orig_forward_name):
    def module_patcher(module, name):
        extension = None
        if module in module_extensions:
            extension = module_extensions[module]
        elif module.__class__ in module_extensions:
            extension = module_extensions[module.__class__]
        elif name in module_extensions:
            extension = module_extensions[name]

        if extension and extension.condition(module):
            log.debug("Patching module %s", module)
            # The Trampoline class is instantiated for every module replacement, so we can use
            # class members individually for each module.

            class Trampoline(torch.autograd.Function):
                # required to be saved in class
                target_extension = extension

                @staticmethod
                @torch.jit.ignore
                def forward(ctx, *args, **kwargs):
                    # Temporarily restore the original forward function of `module` to avoid
                    # recursion issues in `evaluate`, then revert it back.
                    patched_forward = module.forward
                    # set original forward for the module
                    module.forward = getattr(module, orig_forward_name)
                    # call user code
                    results = extension.evaluate(module, *args, **kwargs)
                    module.forward = patched_forward  # return patched forward back
                    return results

            def new_forward(*args, **kwargs):
                return extension.convert(module, Trampoline.apply, *args, **kwargs)

            # make signature of new_forward same as of forward
            new_forward = functools.wraps(module.forward)(new_forward)
            setattr(module, orig_forward_name, module.forward)
            module.forward = new_forward

    for name, module in model.named_modules():
        if hasattr(module, orig_forward_name):
            # already patched, skipping. It may happen when patching applied for same module twice
            log.debug("Unexpectedly found already patched module %s while applying "
                      "ModuleExtension during PyTorch model conversion. "
                      "Result of the conversion maybe broken. Depending on the exact issue "
                      "it may lead to broken original model.", name)
            continue

        module_patcher(module, name)


def unpatch_model(model, orig_forward_name):
    # Restore patched torch functions (bmm, baddbmm, etc.)
    _unpatch_torch_functions()

    for _, module in model.named_modules():
        if hasattr(module, orig_forward_name):
            try:
                module.forward = getattr(module, orig_forward_name)
                delattr(module, orig_forward_name)
            except Exception as error:
                log.warning("Exception raised during model unpatching. "
                            "Depending on the exact issue it may lead to broken original model.\n"
                            "Original exception details:\n%s", error)


def _create_function_wrapper(extension):
    """Create a wrapper for a torch function using the same Trampoline pattern as modules."""

    class Trampoline(torch.autograd.Function):
        target_extension = extension

        @staticmethod
        @torch.jit.ignore
        def forward(ctx, *args, **kwargs):
            return extension.evaluate(None, *args, **kwargs)

    def wrapper(*args, **kwargs):
        return extension.convert(None, Trampoline.apply, *args, **kwargs)

    return wrapper


def _fp32_tensor(*shape, device=None):
    """Create a placeholder FP32 tensor with the given shape and device."""
    return torch.full(shape, 0.5, dtype=torch.float32, device=device)


# Extension for torch.bmm: (b, n, m) @ (b, m, p) -> (b, n, p)
_bmm_extension = ModuleExtension(
    None, "ov_ext::bmm",
    convert=lambda module, target_op, *args, **kwargs: target_op(*args),
    evaluate=lambda module, *args, **kwargs: _fp32_tensor(
        args[0].shape[0], args[0].shape[1], args[1].shape[2], device=args[0].device)
)


# Thread-safe, reference-counted storage for patched torch functions.
# Each entry: key -> (orig_fn, ref_count)
_patched_torch_functions = {}
_patch_lock = threading.Lock()


def _patch_torch_functions():
    """Patch torch functions that don't work well with 16-bit types (e.g., bmm for MoE models).

    These patches skip actual computation and create custom ops in the TorchScript graph,
    similar to how ModuleExtension works for modules. This speeds up tracing and avoids
    loading weights from mmap.

    Thread-safe and ref-counted: the wrapper is installed only on the first call and
    restored only when every matching _unpatch_torch_functions() call has been made,
    so concurrent or nested patching is safe.
    """
    functions_to_patch = [
        (torch, "bmm", _bmm_extension),
    ]

    with _patch_lock:
        for module, fn_name, extension in functions_to_patch:
            key = (module, fn_name)
            if key in _patched_torch_functions:
                orig_fn, ref_count = _patched_torch_functions[key]
                _patched_torch_functions[key] = (orig_fn, ref_count + 1)
                log.debug("Already patched torch function: %s.%s (ref_count=%d)",
                          module.__name__, fn_name, ref_count + 1)
            else:
                orig_fn = getattr(module, fn_name)
                _patched_torch_functions[key] = (orig_fn, 1)
                setattr(module, fn_name, _create_function_wrapper(extension))
                log.debug("Patched torch function: %s.%s", module.__name__, fn_name)


def _unpatch_torch_functions():
    """Restore original torch functions.

    Decrements the ref count; the original function is restored only when the
    count reaches zero, so nested/concurrent patch pairs work correctly.
    """
    with _patch_lock:
        to_remove = []
        for (module, fn_name), (orig_fn, ref_count) in _patched_torch_functions.items():
            new_count = ref_count - 1
            if new_count <= 0:
                setattr(module, fn_name, orig_fn)
                to_remove.append((module, fn_name))
                log.debug("Restored torch function: %s.%s", module.__name__, fn_name)
            else:
                _patched_torch_functions[(module, fn_name)] = (orig_fn, new_count)
                log.debug("Decremented ref count for torch function: %s.%s (ref_count=%d)",
                          module.__name__, fn_name, new_count)
        for key in to_remove:
            del _patched_torch_functions[key]


# Single FRAGMENT library handle for auto-registered ModuleExtension ops.
# Created lazily on first use; reused for all subsequent registrations to
# avoid unbounded Library object accumulation in long-lived processes.
_module_ext_lib = None
_module_ext_registered_ops = {}  # op_name → tuple(schema_args)
_module_ext_lock = threading.Lock()

# Thread-local context used by auto-registered Meta/CPU impls to call the
# extension's ``evaluate`` callback with the original forward arguments.
# ``new_forward`` pushes a callable onto the stack before calling
# ``convert`` and pops it afterwards, so that the Meta dispatch inside
# ``target_op(...)`` can obtain correct output shapes.
_export_tracing_ctx = threading.local()

# Cache of output (shape, dtype) keyed by (op_name, input_tensor_metadata).
# Populated during dynamo tracing (phase 1 of torch.export) when the
# evaluate stack is available.  Consumed during aot_export metadata
# collection (phase 2) when the stack is no longer on the call path.
# Module-level (not thread-local) because phase 2 may run in a context
# where thread-local attributes are reset.  Cleared after each export
# by ``_clear_export_cache()``.
_meta_shape_cache = {}


def _clear_export_cache():
    """Clear per-export shape cache.

    Called after ``torch.export`` completes to prevent stale shape
    metadata from leaking across independent exports in long-lived
    processes.
    """
    _meta_shape_cache.clear()


def _auto_register_module_extension_op(namespace, op_name, schema_args):
    """Auto-register a ``torch.library`` op for a ``ModuleExtension`` target.

    ``schema_args`` is a list of schema-argument strings understood by
    ``torch.library`` (e.g. ``["Tensor x0", "int x1", "Tensor? x2"]``).
    The op returns a single ``Tensor``.

    Uses a single shared ``FRAGMENT`` library handle for the ``ov_ext``
    namespace to avoid creating a new ``Library`` object per op.

    Thread-safe: guarded by ``_module_ext_lock``.
    """
    global _module_ext_lib
    schema_key = tuple(schema_args)

    with _module_ext_lock:
        if _module_ext_lib is None:
            # hasattr() is insufficient: torch.ops.__getattr__ raises
            # RuntimeError (not AttributeError) for missing namespaces,
            # so hasattr() may propagate the exception instead of
            # returning False.
            try:
                getattr(torch.ops, namespace)
                kind = "FRAGMENT"
            except (AttributeError, RuntimeError):
                kind = "DEF"
            _module_ext_lib = torch.library.Library(namespace, kind)

        if op_name in _module_ext_registered_ops:
            existing = _module_ext_registered_ops[op_name]
            if existing != schema_key:
                raise RuntimeError(
                    f"ModuleExtension op '{namespace}::{op_name}' was already "
                    f"registered with schema ({', '.join(existing)}) but is now "
                    f"requested with ({', '.join(schema_key)}). Use distinct "
                    f"target_op names for extensions with different signatures.")
            return getattr(getattr(torch.ops, namespace), op_name)

        args = ", ".join(schema_args)
        _module_ext_lib.define(f"{op_name}({args}) -> Tensor")

        @torch.library.impl(_module_ext_lib, op_name, "Meta")
        def _meta(*xs):
            # Build a cache key from tensor shapes/dtypes for
            # cross-phase lookup (phase 1 populates, phase 2 consumes).
            cache_key = (op_name,) + tuple(
                (tuple(a.shape), a.dtype) if isinstance(a, torch.Tensor)
                else (type(a).__name__, a) for a in xs)

            # Try evaluate callback (available during dynamo tracing).
            stack = getattr(_export_tracing_ctx, "evaluate_stack", None)
            if stack:
                try:
                    result = stack[-1]()
                    if isinstance(result, torch.Tensor):
                        out_shape = tuple(result.shape)
                        out_dtype = result.dtype
                        _meta_shape_cache[cache_key] = (out_shape, out_dtype)
                        return torch.empty(out_shape, dtype=out_dtype, device="meta")
                except Exception:
                    pass

            # Fall back to cached shape from a previous evaluate call
            # (needed during aot_export metadata collection).
            if cache_key in _meta_shape_cache:
                out_shape, out_dtype = _meta_shape_cache[cache_key]
                return torch.empty(out_shape, dtype=out_dtype, device="meta")

            for arg in xs:
                if isinstance(arg, torch.Tensor):
                    return torch.empty_like(arg)
            return torch.empty(1, device="meta")

        @torch.library.impl(_module_ext_lib, op_name, "CPU")
        def _cpu(*xs):
            stack = getattr(_export_tracing_ctx, "evaluate_stack", None)
            if stack:
                try:
                    result = stack[-1]()
                    if isinstance(result, torch.Tensor):
                        return result
                except Exception:
                    pass
            for arg in xs:
                if isinstance(arg, torch.Tensor):
                    return arg
            return torch.empty(1)

        _module_ext_registered_ops[op_name] = schema_key
        return getattr(getattr(torch.ops, namespace), op_name)


def _derive_schema_from_args(call_args, target_op_name):
    """Derive ``torch.library`` schema-argument strings from actual call arguments.

    Returns a list such as ``["Tensor x0", "int x1", "Tensor? x2"]``.
    Raises ``RuntimeError`` for unsupported argument types.

    ``None`` is not accepted because it is ambiguous (could be
    ``Tensor?``, ``int?``, etc.) and the correct schema type cannot be
    inferred.  Use an explicit sentinel or avoid passing ``None`` from
    ``convert()`` for auto-registered ops.
    """
    schema_parts = []
    for i, arg in enumerate(call_args):
        if isinstance(arg, torch.Tensor):
            schema_parts.append(f"Tensor x{i}")
        elif arg is None:
            raise RuntimeError(
                f"ModuleExtension '{target_op_name}': convert() passed "
                f"None at position {i} to target_op. None is ambiguous "
                f"for schema inference (Tensor?, int?, …). For ops with "
                f"optional arguments, pre-register the op with an "
                f"explicit schema in ov_custom_ops.py instead of relying "
                f"on lazy auto-registration.")
        elif isinstance(arg, bool):
            # bool before int: bool is a subclass of int in Python.
            schema_parts.append(f"bool x{i}")
        elif isinstance(arg, int):
            schema_parts.append(f"int x{i}")
        elif isinstance(arg, float):
            schema_parts.append(f"float x{i}")
        elif isinstance(arg, str):
            schema_parts.append(f"str x{i}")
        else:
            raise RuntimeError(
                f"ModuleExtension '{target_op_name}': convert() passed "
                f"unsupported argument type {type(arg).__name__} at position "
                f"{i} to target_op. Supported types: Tensor, bool, "
                f"int, float, str.")
    return schema_parts


def patch_model_for_export(model, module_extensions, orig_forward_name):
    """Patch model modules for ``torch.export`` by replacing forwards with ``torch.library`` op calls.

    Unlike ``patch_model`` (which uses ``torch.autograd.Function`` / ``torch.jit.ignore``
    for TorchScript tracing), this function creates forwards that call registered
    ``torch.library`` custom ops so that ``torch.export`` captures them as
    ``call_function`` nodes in the FX graph.

    Returns:
        dict: Mapping from the registered ``namespace::op_name`` to the
        user-provided ``target_op`` string.  The FX decoder uses this in
        ``get_op_type()`` so that the C++ frontend sees the user's original
        ``target_op`` name.
    """
    import openvino.frontend.pytorch.ov_custom_ops  # noqa: F401 – triggers registration

    op_type_mapping = {}  # registered_name → user target_op

    def _resolve_target_op(extension):
        """Resolve ``target_op`` to a ``torch.ops`` callable.

        If the op is already registered (from ``ov_custom_ops.py`` or a
        previous call), returns it directly.  Otherwise returns a
        lazy-registering wrapper that auto-registers the op on its first
        call during ``torch.export`` tracing, deriving the schema from
        the actual arguments that ``convert()`` passes.
        """
        target_op = extension.target_op
        parts = target_op.split("::")
        if len(parts) == 2:
            namespace, op_name = parts
        elif len(parts) == 1:
            namespace, op_name = "ov_ext", parts[0]
        else:
            raise RuntimeError(
                f"Invalid target_op format: '{target_op}'. "
                "Expected 'op_name' or 'namespace::op_name'.")

        # Warn if target_op collides with an existing PyTorch op.
        # Note: torch.ops namespace __getattr__ raises RuntimeError (not
        # AttributeError) for missing ops, so getattr(ns, name, default)
        # does not work — explicit try/except is required.
        if namespace != "ov_ext":
            try:
                getattr(getattr(torch.ops, namespace), op_name)
                log.warning(
                    "ModuleExtension target_op '%s' matches an existing PyTorch "
                    "op. A passthrough op will be registered under the ov_ext "
                    "namespace instead (the original op will NOT be called, "
                    "consistent with TorchScript ModuleExtension behavior).",
                    target_op)
            except (AttributeError, RuntimeError):
                pass

        # Always register under ov_ext — target_op is just a label.
        # Sanitize dots in op_name (torch.library rejects them).
        safe_op_name = op_name.replace(".", "_")
        # Use the FX-style dotted name as the mapping key so that
        # get_op_type() can do a direct dict lookup without parsing.
        fx_name = f"ov_ext.{safe_op_name}.default"

        # Detect mapping collisions: same fx_name but different target_op.
        if (fx_name in op_type_mapping
                and op_type_mapping[fx_name] != target_op):
            raise RuntimeError(
                f"ModuleExtension target_op collision: '{target_op}' and "
                f"'{op_type_mapping[fx_name]}' both map to "
                f"FX name '{fx_name}' (after dot "
                "sanitization). Use distinct target_op names.")

        # Reuse if already registered (pre-registered or previous extension).
        try:
            op_fn = getattr(torch.ops.ov_ext, safe_op_name)
            op_type_mapping[fx_name] = target_op
            return op_fn
        except (AttributeError, RuntimeError):
            pass

        # Return a lazy wrapper: the op schema is derived from the actual
        # arguments that convert() passes on the first call during tracing.
        log.debug("Will lazily register torch.library op ov_ext::%s for "
                  "ModuleExtension (target_op='%s')", safe_op_name, target_op)

        def _lazy_register_and_call(*call_args):
            schema_args = _derive_schema_from_args(call_args, target_op)
            op_fn = _auto_register_module_extension_op(
                "ov_ext", safe_op_name, schema_args)
            op_type_mapping[fx_name] = target_op
            return op_fn(*call_args)

        return _lazy_register_and_call

    def module_patcher(module, name):
        extension = None
        if module in module_extensions:
            extension = module_extensions[module]
        elif module.__class__ in module_extensions:
            extension = module_extensions[module.__class__]
        elif name in module_extensions:
            extension = module_extensions[name]

        if extension and extension.condition(module):
            log.debug("Patching module %s for torch.export", module)
            target_op = _resolve_target_op(extension)
            orig_fwd = module.forward  # capture before overwrite

            def new_forward(*args, **kwargs):
                # Push an evaluate thunk so the auto-registered Meta/CPU
                # impls can produce the correct output shape instead of
                # blindly returning ``empty_like(first_tensor)``.
                def _evaluate():
                    # Temporarily restore the original forward to avoid
                    # recursion (evaluate's default calls module()).
                    patched = module.forward
                    module.forward = orig_fwd
                    try:
                        return extension.evaluate(module, *args, **kwargs)
                    finally:
                        module.forward = patched

                stack = getattr(_export_tracing_ctx, "evaluate_stack", None)
                if stack is None:
                    stack = []
                    _export_tracing_ctx.evaluate_stack = stack
                stack.append(_evaluate)
                try:
                    return extension.convert(module, target_op, *args, **kwargs)
                finally:
                    stack.pop()

            new_forward = functools.wraps(orig_fwd)(new_forward)
            setattr(module, orig_forward_name, orig_fwd)
            module.forward = new_forward

    for name, module in model.named_modules():
        if hasattr(module, orig_forward_name):
            log.debug("Unexpectedly found already patched module %s while applying "
                      "ModuleExtension for torch.export during PyTorch model conversion. "
                      "Result of the conversion maybe broken.", name)
            continue
        module_patcher(module, name)

    return op_type_mapping


def _get_16bit_extensions(patch_condition=None):
    """Return a dict of ModuleExtension entries for known 16-bit module types."""
    supported = {torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2}

    if patch_condition is None:
        def patch_condition(module):
            dtype_to_patch = {torch.float32, *supported}
            weight = getattr(module, "weight", None)
            return weight is not None and weight.dtype in dtype_to_patch

    extensions = {
        torch.nn.Linear: ModuleExtension(
            torch.nn.Linear, "ov_ext::linear",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: _fp32_tensor(
                *args[0].shape[:-1], module.out_features, device=args[0].device),
            condition=patch_condition),
        torch.nn.Embedding: ModuleExtension(
            torch.nn.Embedding, "ov_ext::embedding",
            convert=lambda module, target_op, *args, **kwargs: target_op(module.weight,
                                                                         args[0],
                                                                         module.padding_idx,
                                                                         module.scale_grad_by_freq,
                                                                         module.sparse),
            evaluate=lambda module, *args, **kwargs: _fp32_tensor(
                *args[1].shape, module.embedding_dim, device=args[1].device),
            condition=patch_condition),
    }
    try:
        from transformers.pytorch_utils import Conv1D
        extensions[Conv1D] = ModuleExtension(
            Conv1D, "ov_ext::conv1d",
            convert=lambda module, target_op, *args, **kwargs: target_op(args[0],
                                                                         module.weight,
                                                                         module.bias),
            evaluate=lambda module, *args, **kwargs: _fp32_tensor(
                *args[0].shape[:-1], module.nf, device=args[0].device),
            condition=patch_condition)
    except ImportError:
        pass
    return extensions, supported


def __make_16bit_traceable(model: torch.nn.Module,
                           orig_forward_name: str = "_openvino_module_extension_patch_orig_forward",
                           patch_condition=None):
    """Prepare a 16-bit PyTorch model for tracing with OpenVINO.

    - Replace known list of modules with ModuleExtension.
    - Patch torch functions (bmm, baddbmm, etc.) for MoE and similar models.
    - Convert other modules with weights to FP32.
    """
    # Patch torch functions for operations like bmm used in MoE models
    _patch_torch_functions()

    extensions, supported = _get_16bit_extensions(patch_condition)
    patch_model(model, extensions, orig_forward_name)
    for _, module in model.named_modules():
        if (module.__class__ not in extensions
            and (any(p.dtype in supported for p in module.parameters(False))
                 or any(b.dtype in supported for b in module.buffers(False)))):
            log.debug("Casting module %s to float32", module)
            module.float()
