# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import logging
import os
from functools import partial
from hashlib import sha256

import torch
from torch._dynamo.backends.common import fake_tensor_unsupported, aot_autograd
from torch._dynamo.backends.registry import register_backend
from torch._inductor.compile_fx import compile_fx
from torch._inductor.freezing import replace_params_with_constants
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import decomposition_table, get_decompositions

from openvino.frontend import FrontEndManager
from openvino import Core, Type, PartialShape
from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
from openvino.frontend.pytorch.torchdynamo import decompositions
from openvino.frontend.pytorch.torchdynamo.decompositions import get_aot_decomposition_list, get_inf_decomposition_list
from openvino.frontend.pytorch.torchdynamo.partition import Partitioner
from openvino.frontend.pytorch.torchdynamo.execute import execute, execute_cached
from openvino.frontend.pytorch.torchdynamo.compile import cached_model_name, openvino_compile_cached_model
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_cache_dir, _get_device, _get_model_caching, _get_decompositions, _get_aot_autograd

from openvino import Core, Type, PartialShape

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

"""
    This is a preview feature in OpenVINO. This feature
    enables users to compile PyTorch models using torch.compile
    with OpenVINO as a target backend in PyTorch applications

    Sample usage:
    This sample code loads resnet50 torchvision model and compiles it using torch dynamo.
    We can then use this model for inference. We only need to add two lines of code to
    the Pytorch applications which are marked in the code below

    1) import openvino.torch
    model = torchvision.models.resnet50()
    2) model = torch.compile(model, backend="openvino")
"""

openvino_options = {}

# Real (non-fake) example inputs of the dynamo graph currently being compiled,
# pushed by openvino() for _freeze_static_inputs. A stack rather than a single
# slot only so that an exception cannot leave a stale entry behind.
_real_example_inputs = []


def _freeze_static_inputs(gm, example_inputs, fw_metadata):
    """Fold weight placeholders into Constants using fw_metadata.static_input_indices.

    ``replace_params_with_constants`` pairs ``params_flat[i]`` with the i-th
    placeholder, so it can only freeze weights that dynamo kept as module
    attributes. Once inlining of builtin nn modules became unconditional
    (torch 2.13 made ``inline_inbuilt_nn_modules`` a no-op that is always
    True), every weight is lifted to a graph input instead and both
    ``params_flat`` and ``params_flat_unwrap_subclasses`` come back empty --
    nothing is frozen. The weights then reach OpenVINO as Parameters rather
    than Constants, which costs a Transpose per MatMul, blocks constant
    folding, and yields wrong results.

    ``fw_metadata.static_input_indices`` marks the weights regardless, so
    freeze by index. This is not a 2.13-only fallback: on 2.11, where
    params_flat is populated, the index set is identical to the one the
    upstream helper derives, so this one path serves both. The indices are
    not contiguous (the real runtime inputs are interleaved), hence the
    explicit complement rather than a prefix slice. Returns the preserved
    (unfrozen) indices, or None if this path does not apply and the caller
    should use the upstream helper.
    """
    static = set(getattr(fw_metadata, "static_input_indices", None) or ())
    if not static:
        return None
    from torch._inductor.constant_folding import replace_node_with_constant
    from torch._functorch._aot_autograd.schemas import MutationType

    placeholders = gm.graph.find_nodes(op="placeholder")
    # example_inputs are aot_autograd's FakeTensors; the real ones were
    # stashed by openvino(). Require an exact length match -- if the two
    # graphs ever stop lining up 1:1, folding by index would bake the wrong
    # tensor into the weights, so fall back to the upstream helper instead.
    real = _real_example_inputs[-1] if _real_example_inputs else None
    if real is None or len(real) != len(placeholders):
        logger.debug("static-input freezing skipped: real inputs %s vs %d placeholders",
                     None if real is None else len(real), len(placeholders))
        return None
    example_inputs = real

    # Same exclusions as replace_params_with_constants: an input that is
    # mutated or aliased by an output cannot become a Constant.
    mutated = {i for i, m in enumerate(fw_metadata.input_info)
               if m.mutation_type in (MutationType.MUTATED_IN_GRAPH,
                                      MutationType.MUTATED_OUT_GRAPH)}
    aliased = {o.base_idx for o in fw_metadata.output_info if o.base_idx is not None}

    from torch._subclasses.fake_tensor import FakeTensor
    preserved_arg_indices = []
    for i, node in enumerate(placeholders):
        freezable = (i in static and i not in mutated and i not in aliased
                     and isinstance(example_inputs[i], torch.Tensor)
                     and not isinstance(example_inputs[i], FakeTensor))
        if freezable:
            replace_node_with_constant(gm, node, example_inputs[i])
        else:
            preserved_arg_indices.append(i)
    # Every static input just became a Constant, so none remain to track.
    fw_metadata.static_input_indices = []
    gm.recompile()
    return preserved_arg_indices

# Disable regional compilation which was enabled by default from Torch 2.5.0
if hasattr(torch._dynamo.config, "inline_inbuilt_nn_modules"):
    torch._dynamo.config.inline_inbuilt_nn_modules=False

@fake_tensor_unsupported
def openvino(subgraph, example_inputs, options=None):
    if _get_aot_autograd(options):
        global openvino_options
        openvino_options = options
        decompositions = _get_decompositions(options) + get_inf_decomposition_list() + get_aot_decomposition_list()
        # aot_autograd hands fw_compiler FakeTensors, which cannot be folded
        # into Constants. @fake_tensor_unsupported means the inputs here are
        # real and in the same order as the AOT placeholders, so stash them
        # for _freeze_static_inputs. Dynamo compiles one graph at a time, so a
        # single slot is enough; it is cleared once the compile returns.
        _real_example_inputs.append(list(example_inputs))
        try:
            return aot_autograd(fw_compiler=fx_openvino, bw_compiler=fx_openvino, decompositions=get_decompositions(decompositions))(subgraph, example_inputs)
        finally:
            _real_example_inputs.pop()
    return fx_openvino(subgraph, example_inputs, options)

if "openvino" not in torch.compiler.list_backends():
    register_backend(compiler_fn=openvino, name="openvino")

def fx_openvino(subgraph, example_inputs, options=None):
    try:
        if len(openvino_options) != 0:
            options = openvino_options
        executor_parameters = None
        inputs_reversed = False
        openvino_model_caching = _get_model_caching(options)
        if openvino_model_caching is not None and openvino_model_caching:
            # Create a hash to be used for caching
            model_hash_str = sha256(subgraph.code.encode("utf-8")).hexdigest()
            executor_parameters = {"model_hash_str": model_hash_str}
            # Check if the model was fully supported and already cached
            example_inputs.reverse()
            inputs_reversed = True
            maybe_fs_cached_name = cached_model_name(model_hash_str + "_fs", _get_device(options), example_inputs, _get_cache_dir(options))
            if os.path.isfile(maybe_fs_cached_name + ".xml") and os.path.isfile(maybe_fs_cached_name + ".bin"):
                # Model is fully supported and already cached. Run the cached OV model directly.
                compiled_model = openvino_compile_cached_model(maybe_fs_cached_name, options, *example_inputs)

                def _call(*args):
                    res = execute_cached(compiled_model, *args)
                    return res

                return _call
        if inputs_reversed:
            example_inputs.reverse()

        preserved_arg_indices = []
        if _get_aot_autograd(options):
            preserved_arg_indices = None
            if tracing_context := torch._guards.TracingContext.try_get():
                fw_metadata = tracing_context.fw_metadata
                params_flat = tracing_context.params_flat
                if not params_flat:
                    # torch._inductor.freezing._freeze reads this one.
                    params_flat = getattr(
                        tracing_context, "params_flat_unwrap_subclasses", None) or []
                assert fw_metadata is not None and params_flat is not None
                # static_input_indices marks the weights on every torch
                # version we support, and on the ones where params_flat is
                # also populated the two agree exactly, so this single path
                # covers both. replace_params_with_constants below stays as
                # the fallback for the case static_input_indices is empty.
                preserved_arg_indices = _freeze_static_inputs(
                    subgraph, example_inputs, fw_metadata)
            if preserved_arg_indices is None:
                preserved_arg_indices = replace_params_with_constants(subgraph, params_flat, fw_metadata)
            example_inputs = [example_inputs[ind] for ind in preserved_arg_indices]
            model = subgraph
        else:
            decompositions = _get_decompositions(options) + get_inf_decomposition_list()
            model = make_fx(
                subgraph,
                decomposition_table=get_decompositions(decompositions),
                tracing_mode="fake",
                _allow_non_fake_inputs=True,
            )(*example_inputs)
            with torch.no_grad():
                model.eval()
        partitioner = Partitioner(options)
        compiled_model = partitioner.make_partitions(model, options)

        if executor_parameters is not None and "model_hash_str" in executor_parameters:
            # Check if the model is fully supported.
            fully_supported = partitioner.check_fully_supported(compiled_model)
            if fully_supported:
                executor_parameters["model_hash_str"] += "_fs"

        def _call(*args):
            if _get_aot_autograd(options):
                args_list = args[0]
                args_new = [args_list[i] for i in preserved_arg_indices]
                args = args_new
            res = execute(compiled_model, *args, executor="openvino", executor_parameters=executor_parameters, options=options)
            return res

        if _get_aot_autograd(options):
            _call._boxed_call = True  # type: ignore[attr-defined]
        return _call
    except Exception as e:
        logger.debug(f"Failed in OpenVINO execution: {e}")
        return compile_fx(subgraph, example_inputs)


def reset():
    clear_caches()
