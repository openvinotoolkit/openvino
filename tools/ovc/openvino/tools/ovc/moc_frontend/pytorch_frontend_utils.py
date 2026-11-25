# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import pathlib
import re
import sys
from collections import Counter, deque

import numpy as np

# pylint: disable=no-name-in-module,import-error
from openvino import Tensor, PartialShape
from openvino.tools.ovc.cli_parser import input_to_input_cut_info, single_input_to_input_cut_info, _InputCutInfo
from openvino.tools.ovc.error import Error


def extract_module_extensions(args):
    from openvino.frontend.pytorch.module_extension import ModuleExtension
    extensions = args.get('extension', []) or []
    if not isinstance(extensions, (list, tuple)):
        extensions = [extensions]
    return {extension.module: extension for extension in extensions if isinstance(extension, ModuleExtension)}


def _build_dynamic_shapes(inputs, input_specs=None, input_names=None, input_aliases=None):
    """Build dynamic_shapes for torch.export.export.

    If input_specs (list of _InputCutInfo from the 'input' parameter) is provided
    and contains shapes, dimensions marked as -1 (fully dynamic) get Dim.AUTO,
    dimensions with min/max constraints (e.g. Dimension(1, 10)) get
    Dim.AUTO with the same bounds, and fixed dimensions stay static.
    When no specs are given returns None so that torch.export.export produces
    a fully static graph.

    The input_specs list is flat (one spec per leaf tensor), while inputs may
    contain nested tuples/lists (e.g. past_key_values). pytree is used to
    flatten inputs, pair each leaf with its spec, and then unflatten the
    result back into the original structure that torch.export expects.
    """
    import torch
    from torch.export import Dim
    from torch.utils._pytree import GetAttrKey, MappingKey, SequenceKey, tree_flatten_with_path, tree_unflatten

    if input_specs is None:
        return None

    def _dims_for_tensor(tensor, spec_shape):
        """Return a per-dim dict mapping dim index to Dim hint.

        - Fully dynamic dims (-1) → Dim.AUTO
        - Constrained dims (min, max) → Dim.AUTO(min=..., max=...)
        - Static dims → omitted (stays static)
        """
        if not isinstance(tensor, torch.Tensor) or tensor.ndim == 0:
            return None
        if spec_shape is None:
            return None
        dims = {}
        for d in range(tensor.ndim):
            if d >= len(spec_shape):
                continue
            sd = spec_shape[d]
            if isinstance(sd, tuple):
                # Constrained dimension with (min, max) bounds
                min_val, max_val = sd
                kwargs = {}
                if min_val > 0:
                    kwargs['min'] = min_val
                if max_val != -1:  # -1 means unbounded
                    kwargs['max'] = max_val
                dims[d] = Dim.AUTO(**kwargs)
            elif sd == -1:
                dims[d] = Dim.AUTO
        # If no dynamic dims, return None
        if not dims:
            return None
        return dims

    # Resolve spec shapes (list of dim descriptors aligned to flat leaves).
    # Each dim is: int (static), -1 (fully dynamic), or (min, max) tuple.
    def _get_spec_shapes():
        shapes = []
        for spec in input_specs:
            if spec is None or getattr(spec, 'shape', None) is None or not spec.shape.rank.is_static:
                shapes.append(None)
                continue
            dims = []
            for i in range(spec.shape.rank.get_length()):
                dim = spec.shape.get_dimension(i)
                if dim.is_static:
                    dims.append(dim.get_length())
                elif dim.get_min_length() == 0 and dim.get_max_length() == -1:
                    dims.append(-1)  # Fully dynamic (unbounded)
                else:
                    dims.append((dim.get_min_length(), dim.get_max_length()))  # Constrained
            shapes.append(dims)
        return shapes

    spec_shapes = _get_spec_shapes()

    # Flatten the (possibly nested) inputs to get leaf tensors and the
    # tree structure.  Build a flat list of per-leaf dim specs, then
    # unflatten back so that dynamic_shapes mirrors the input structure
    # exactly as torch.export.export requires.
    if isinstance(inputs, torch.Tensor):
        inputs = (inputs,)

    flat_paths, tree_spec = tree_flatten_with_path(inputs)
    flat_leaves = [leaf for _, leaf in flat_paths]
    if isinstance(inputs, dict):
        examples = flatten_inputs(inputs)
    elif isinstance(inputs, (tuple, list)):
        examples = []
        for i, value in enumerate(inputs):
            name = input_names[i] if input_names and i < len(input_names) else None
            examples.extend(flatten_inputs({name: value} if name else [value]))
    else:
        examples = [inputs]
    names = [example[0] if isinstance(example, tuple) and len(example) == 2 else None
             for example in examples]

    def canonical_name(path):
        if not path:
            return None
        if isinstance(path[0], MappingKey):
            root = path[0].key
        elif isinstance(path[0], SequenceKey) and input_names and path[0].idx < len(input_names):
            root = input_names[path[0].idx]
        else:
            return None
        if len(path) == 1:
            return root
        parts = [root]
        for key in path[1:]:
            if isinstance(key, MappingKey):
                parts.append(re.sub(r"[^a-zA-Z0-9]", "_", str(key.key)))
            elif isinstance(key, SequenceKey):
                parts.append(str(key.idx))
            elif isinstance(key, GetAttrKey):
                parts.append(key.name)
            else:
                return None
        return "_".join(part.lower() for part in parts)

    from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
    canonical_names = TorchFXPythonDecoder._disambiguate_leaf_names(
        [canonical_name(path) for path, _ in flat_paths], [len(path) > 1 for path, _ in flat_paths])
    assigned = [None] * len(flat_leaves)
    remaining = set(range(len(input_specs)))
    for j, spec in enumerate(input_specs):
        if spec is None or spec.name is None or not spec.name.isdecimal():
            continue
        leaf_idx = (input_aliases or {}).get(spec.name)
        if leaf_idx is None:
            raise Error(f"Cannot match input '{spec.name}' to an exported PyTorch input.")
        assigned[leaf_idx] = j
        remaining.remove(j)
    for candidate_names in (canonical_names, names):
        counts = Counter(name for name, leaf in zip(candidate_names, flat_leaves)
                         if name is not None and isinstance(leaf, torch.Tensor))
        for i, name in enumerate(candidate_names):
            if name is None or counts[name] != 1 or not isinstance(flat_leaves[i], torch.Tensor):
                continue
            matches = [j for j in remaining if input_specs[j] is not None and input_specs[j].name == name]
            if len(matches) == 1:
                if assigned[i] is not None:
                    raise Error(f"Input '{name}' is specified more than once.")
                assigned[i] = matches[0]
                remaining.remove(matches[0])
    unnamed = deque(j for j, spec in enumerate(input_specs) if spec is None or spec.name is None)
    for i, leaf in enumerate(flat_leaves):
        if unnamed and assigned[i] is None and isinstance(leaf, torch.Tensor):
            assigned[i] = unnamed.popleft()
    flat_dim_specs = []
    for i, leaf in enumerate(flat_leaves):
        spec = input_specs[assigned[i]] if assigned[i] is not None else None
        if isinstance(leaf, torch.Tensor) and spec is not None and spec.shape is not None:
            if not spec.shape.rank.is_static:
                raise Error("torch.export requires an input shape with a static rank; use dynamo=False otherwise.")
            if not spec.shape.compatible(PartialShape(list(leaf.shape))):
                raise Error(f"Input shape {spec.shape} is incompatible with example_input shape {list(leaf.shape)}. "
                            "Provide a matching example_input or use dynamo=False.")
        ss = spec_shapes[assigned[i]] if assigned[i] is not None else None
        flat_dim_specs.append(_dims_for_tensor(leaf, ss))

    return tree_unflatten(flat_dim_specs, tree_spec)


def get_pytorch_decoder(model, example_inputs, args):
    try:
        from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
        import torch
        TorchFXPythonDecoder = None
        if hasattr(torch, "export") and hasattr(torch.export, "ExportedProgram"):
            from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e

    if 'nncf' in sys.modules:
        is_good_version = True
        try:
            from nncf.torch.nncf_network import NNCFNetwork

            if isinstance(model, NNCFNetwork):
                from packaging import version
                if version.parse(sys.modules['nncf'].__version__) < version.parse("2.6"):
                    is_good_version = False
        except:
            pass
        if not is_good_version:
            raise RuntimeError("NNCF models produced by nncf<2.6 are not "
                               "supported directly. Please upgrade nncf or "
                               "export to ONNX first.")
    dynamo = args.get("dynamo", False)
    if dynamo and isinstance(model, getattr(torch.jit, "ScriptFunction", ())):
        raise Error("torch.export does not support torch.jit.ScriptFunction; pass dynamo=False to use TorchScript.")
    use_dynamo = dynamo and isinstance(model, torch.nn.Module)
    is_exported = TorchFXPythonDecoder is not None and isinstance(model, torch.export.ExportedProgram)
    inputs = prepare_torch_inputs(example_inputs, preserve_lists=use_dynamo or is_exported)
    if use_dynamo and inputs is None:
        raise Error("example_input is required when dynamo=True; pass dynamo=False to use TorchScript.")
    elif use_dynamo and (TorchFXPythonDecoder is None or not hasattr(getattr(torch.export, "Dim", None), "AUTO")):
        raise Error("torch.export conversion requires PyTorch >= 2.6; pass dynamo=False to use TorchScript.")

    decoder_types = (TorchScriptPythonDecoder,)
    if TorchFXPythonDecoder is not None:
        decoder_types += (TorchFXPythonDecoder,)
    if not isinstance(model, decoder_types):
        if is_exported:
            decoder = TorchFXPythonDecoder.from_exported_program(model, shared_memory=args.get("share_weights", True))
            if args.get("input") is not None and example_inputs is None:
                _validate_exported_input_shapes(decoder, _input_specs_for_validation(args["input"]))
        elif use_dynamo:
            import inspect
            from torch.utils._pytree import tree_flatten
            # Extract user-provided input shape specs to constrain dynamic dims.
            # Normalize raw user input (PartialShape, list, etc.) to list of _InputCutInfo
            # because _build_dynamic_shapes expects _InputCutInfo objects with .shape attribute.
            raw_input = args.get('input', None) or None
            input_specs = input_to_input_cut_info(raw_input) if raw_input is not None else None
            signature = inspect.signature(model.forward) if input_specs or isinstance(inputs, dict) else None
            if isinstance(inputs, dict):
                ordered_inputs = {name: inputs[name] for name in signature.parameters if name in inputs}
                ordered_inputs.update({name: value for name, value in inputs.items() if name not in ordered_inputs})
                inputs = ordered_inputs
            leaves, _ = tree_flatten(inputs)
            tensor_ids = [id(leaf) for leaf in leaves if isinstance(leaf, torch.Tensor)]
            if len(tensor_ids) != len(set(tensor_ids)):
                raise Error("example_input contains the same tensor in multiple positions; torch.export can "
                            "specialize these inputs as aliases. Provide separate tensors or use dynamo=False.")
            input_names = None
            shape_inputs = inputs
            if input_specs and any(spec.name for spec in input_specs):
                input_names = [param.name for param in signature.parameters.values()
                               if param.kind in (inspect.Parameter.POSITIONAL_ONLY,
                                                 inspect.Parameter.POSITIONAL_OR_KEYWORD)]
            if input_specs and not isinstance(inputs, dict):
                if any(param.kind == inspect.Parameter.VAR_POSITIONAL for param in signature.parameters.values()):
                    positional_inputs = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
                    bound = signature.bind(*positional_inputs)
                    shape_inputs = tuple(bound.arguments.values())
                    input_names = list(bound.arguments)
            module_extensions = extract_module_extensions(args)
            shared_memory = args.get("share_weights", True)
            numeric_names = [spec.name for spec in input_specs or []
                             if spec.name is not None and spec.name.isdecimal()]

            def exported_aliases(export_decoder):
                if len(leaves) != len(export_decoder._flat_input_is_constant):
                    raise Error("The number of example inputs does not match the exported PyTorch model inputs.")
                leaf_indices = [i for i, is_constant in enumerate(export_decoder._flat_input_is_constant)
                                if not is_constant]
                input_ids = export_decoder.inputs()
                if len(leaf_indices) != len(input_ids):
                    raise Error("The exported PyTorch input IDs do not match its input metadata.")
                return {str(tensor_id): index for tensor_id, index in zip(input_ids, leaf_indices)}

            preview = None
            input_aliases = None
            if numeric_names:
                # Numeric aliases are FX node IDs, including state and specialized placeholders.
                preview = TorchFXPythonDecoder.from_model(
                    model, example_inputs=inputs, module_extensions=module_extensions, shared_memory=shared_memory)
                input_aliases = exported_aliases(preview)
            dynamic_shapes = _build_dynamic_shapes(
                shape_inputs, input_specs=input_specs, input_names=input_names, input_aliases=input_aliases)
            if dynamic_shapes is not None and not any(dim is not None for dim in tree_flatten(dynamic_shapes)[0]):
                dynamic_shapes = None
            if preview is not None and dynamic_shapes is None:
                decoder = preview
            else:
                preview = None
                decoder = TorchFXPythonDecoder.from_model(
                    model,
                    example_inputs=inputs,
                    dynamic_shapes=dynamic_shapes,
                    module_extensions=module_extensions,
                    shared_memory=shared_memory)
                if input_aliases is not None:
                    final_aliases = exported_aliases(decoder)
                    if any(final_aliases.get(name) != input_aliases[name] for name in numeric_names):
                        raise Error("Numeric input aliases changed during torch.export. Use input names instead.")
        else:
            decoder = TorchScriptPythonDecoder(
                model,
                example_input=inputs,
                shared_memory=args.get("share_weights", True),
                module_extensions=extract_module_extensions(args))
    else:
        decoder = model
    args['input_model'] = decoder
    ei = getattr(decoder, "_example_input", None)
    if ei is not None:
        args["example_input"] = ei
    else:
        args["example_input"] = inputs

    return args


def _is_pytorch_zip(path):
    """Check if a file looks like a PyTorch archive (TorchScript or ExportedProgram).

    Both formats are ZIP archives with specific internal entries.
    This lightweight check avoids expensive torch.jit.load / torch.export.load
    calls (and their warnings/side effects) on non-PyTorch files.
    """
    import zipfile
    # Maximum bytes to read from the "archive_format" entry. The expected
    # content is the literal b"pt2" (3 bytes); cap at a small constant so a
    # crafted ZIP cannot trigger large/expensive decompression during
    # auto-detection.
    _ARCHIVE_FORMAT_MAX_BYTES = 16
    try:
        if not zipfile.is_zipfile(path):
            return False
        with zipfile.ZipFile(path, 'r') as zf:
            for info in zf.infolist():
                name = info.filename
                basename = name.rsplit('/', 1)[-1] if '/' in name else name
                # TorchScript archives contain data.pkl and constants.pkl
                if basename == 'data.pkl' or basename == 'constants.pkl':
                    return True
                # Older ExportedProgram format (PyTorch ≤2.6)
                if basename.startswith('serialized_') and basename.endswith('.json'):
                    return True
                # Newer PT2 archive format (PyTorch ≥2.7): contains an
                # "archive_format" entry whose content is b"pt2". Reject
                # entries with unexpected uncompressed size up-front and
                # only read a small bounded prefix to avoid CPU/memory
                # blow-ups on crafted archives.
                if basename == 'archive_format':
                    if info.file_size > _ARCHIVE_FORMAT_MAX_BYTES:
                        continue
                    try:
                        with zf.open(info, 'r') as fh:
                            data = fh.read(_ARCHIVE_FORMAT_MAX_BYTES)
                        if data == b'pt2':
                            return True
                    except Exception:
                        pass
            return False
    except Exception:
        return False


def get_pytorch_decoder_for_model_on_disk(argv, args):
    example_inputs = None
    if 'example_input' in args and args['example_input'] is not None:
        example_inputs = args['example_input']

    if isinstance(argv.input_model, (tuple, list)) and len(argv.input_model) == 1:
        input_model = argv.input_model[0]
    else:
        input_model = argv.input_model

    if not isinstance(input_model, (str, pathlib.Path)):
        return False

    # Quick structural check: only attempt PyTorch loading for ZIP archives
    # that contain PyTorch-specific entries. This avoids importing torch,
    # emitting misleading warnings, and running pickle-based deserialization
    # on non-PyTorch files (e.g. ONNX, TF, TFLite).
    if not _is_pytorch_zip(input_model):
        return False

    try:
        from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
        import torch
    except Exception:
        # Auto-detection must not fail hard if the PyTorch frontend or torch
        # itself cannot be imported (missing shared libs, init errors, etc.).
        # Mirror the broad-except pattern used in convert_impl.py.
        return False

    inputs = prepare_torch_inputs(example_inputs)
    # SECURITY: torch.jit.load / torch.export.load unpickle, so a malicious
    # archive can execute arbitrary code. Mitigations:
    #   1. _is_pytorch_zip() above rejects anything that isn't a ZIP with
    #      PyTorch-specific entries, shrinking the attack surface.
    #   2. input_model comes from the OVC CLI / Python API; the trust
    #      boundary is the caller, who already has code-exec on this host.
    #   3. neither torch.jit.load nor torch.export.load supports a
    #      weights_only / safe mode for TorchScript / older PT2 archives.
    log.warning("Loading PyTorch model from '%s'. PyTorch archives are deserialized via pickle; only load model files you trust.", input_model)

    # attempt to load scripted model
    try:
        model = torch.jit.load(input_model) # nosec B614 - see note above
        model.eval()
        decoder = TorchScriptPythonDecoder(
            model,
            example_input=inputs,
            shared_memory=args.get("share_weights", True),
            module_extensions=extract_module_extensions(args))
        argv.input_model = decoder
        argv.framework = 'pytorch'
        return True
    except Exception:
        pass

    # attempt to load exported model
    try:
        from openvino.frontend.pytorch.fx_decoder import TorchFXPythonDecoder
        exported_program = torch.export.load(input_model) # nosec B614 - see note above
    except Exception:
        return False

    if isinstance(exported_program, torch.export.ExportedProgram):
        argv.input_model = TorchFXPythonDecoder.from_exported_program(
            exported_program, shared_memory=args.get("share_weights", True))
        if example_inputs is not None:
            argv.example_input = prepare_torch_inputs(example_inputs, preserve_lists=True)
        else:
            input_specs = getattr(argv, "input", None)
            if input_specs is None:
                input_specs = args.get("input")
            if input_specs is not None:
                _validate_exported_input_shapes(argv.input_model, _input_specs_for_validation(input_specs))
        argv.framework = 'pytorch'
        return True
    return False


def update_list_or_dict(container, name, idx, value):
    if isinstance(container, dict):
        if name is None:
            name = list(container)[idx]
        container[name] = value
        return
    if idx == len(container):
        container.append(value)
    elif idx > len(container):
        raise Error(f"Wrong {idx}")
    else:
        container[idx] = value
    return


def get_value_from_list_or_dict(container, name, idx):
    if isinstance(container, dict):
        if name is None:
            if idx < len(container):
                name = list(container)[idx]
            return None
        return container.get(name)
    if idx < len(container):
        return container[idx]
    return None


def flatten_inputs(inputs, names=None):
    flattened = []
    if isinstance(inputs, dict):
        # if names are provided we need to unpack in the same order
        if names:
            for name in names:
                if isinstance(inputs[name], (list, tuple, dict)):
                    flattened.extend(flatten_inputs(inputs[name]))
                else:
                    flattened.append((name, inputs[name]))
        else:
            for name, input_data in inputs.items():
                if isinstance(input_data, (list, tuple, dict)):
                    flattened.extend(flatten_inputs(input_data))
                else:
                    flattened.append((name, input_data))
    else:
        for input_data in inputs:
            if isinstance(input_data, (list, tuple, dict)):
                flattened.extend(flatten_inputs(input_data))
            else:
                flattened.append(input_data)
    return flattened


def _torchscript_input_aliases(model):
    from openvino.frontend.pytorch.ts_decoder import TorchScriptPythonDecoder
    if not isinstance(model, TorchScriptPythonDecoder):
        return {}
    schema = getattr(getattr(model.pt_module, "forward", model.pt_module), "schema", None)
    if schema is None or len(schema.arguments) != len(model.raw_inputs):
        return {}
    # TorchScript schema names and graph debug names identify the same inputs.
    return {model.get_input_debug_name(i): argument.name for i, argument in enumerate(schema.arguments)}


def can_suggest_dynamo(model, args):
    torch = sys.modules.get("torch")
    return (torch is not None and not args.get("dynamo", False) and isinstance(model, torch.nn.Module)
            and not isinstance(model, getattr(torch.jit, "ScriptModule", ())))


def add_dynamo_hint(error):
    hint = ("PyTorch model conversion with TorchScript (dynamo=False) failed. "
            "Try convert_model(model, example_input=..., dynamo=True) to convert it with torch.export.")
    try:
        hinted = type(error)(f"{error}\n{hint}")
    except Exception:
        return error
    hinted.__cause__ = error.__cause__
    return hinted.with_traceback(error.__traceback__)


def _validate_exported_input_shapes(model, input_specs):
    input_ids = model.inputs()
    shapes = [model.get_input_shape(i) for i in range(len(input_ids))]
    by_name = {}
    for i, tensor_id in enumerate(input_ids):
        by_name.setdefault(str(tensor_id), shapes[i])
        by_name.setdefault(model.get_input_signature_name(i), shapes[i])
    for i, desc in enumerate(input_specs):
        if desc.name is not None:
            exported_shape = by_name.get(desc.name)
        else:
            exported_shape = shapes[i] if i < len(shapes) else None
        if desc.shape is None:
            continue
        if exported_shape is None:
            raise Error(f"Cannot match input '{desc.name}' to an exported PyTorch input.")
        if not desc.shape.refines(exported_shape):
            raise Error(f"Input shape {desc.shape} is not contained in the exported shape {exported_shape}. "
                        "Export the model with the requested dynamic dimensions instead.")


def _input_specs_for_validation(input_specs):
    if input_specs is None:
        return []
    if isinstance(input_specs, (list, tuple)) and all(isinstance(spec, _InputCutInfo) for spec in input_specs):
        return list(input_specs)
    return input_to_input_cut_info(input_specs)


def extract_input_info_from_example(args, inputs):
    try:
        from openvino.frontend.pytorch.utils import pt_to_ov_type_map  # pylint: disable=no-name-in-module,import-error
    except Exception as e:
        log.error("PyTorch frontend loading failed")
        raise e
    example_inputs = args.input_model._example_input if args.input_model._example_input is not None else args.example_input
    if example_inputs is None:
        return
    is_dict_input = isinstance(example_inputs, dict)
    if not isinstance(example_inputs, (list, tuple, dict)):
        example_inputs = [example_inputs]
    input_names = args.input_model._input_signature or None
    if input_names and input_names[0] == "self":
        input_names = input_names[1:]
    is_exported = getattr(args.input_model, "_is_exported_program", False)
    if is_exported and not is_dict_input and len(example_inputs) != len(input_names or []):
        raise Error("example_input does not match the exported PyTorch model's input arguments.")
    if (input_names or is_exported) and not is_dict_input:
        example_inputs = dict(zip(input_names or [], example_inputs))
    input_names = input_names or []
    if is_exported:
        from torch.utils._pytree import tree_unflatten
        from openvino.frontend.pytorch.fx_decoder import spec_children

        spec = args.input_model.pt_module._in_spec
        args_spec, kwargs_spec = spec_children(spec)
        positional_names = input_names[:args_spec.num_children]
        keyword_names = kwargs_spec.context
        expected_names = positional_names + list(keyword_names)
        if not isinstance(example_inputs, dict) or set(example_inputs) != set(expected_names):
            raise Error("example_input does not match the exported PyTorch model's input arguments.")
        try:
            ordered_leaves = spec.flatten_up_to((
                tuple(example_inputs[name] for name in positional_names),
                {name: example_inputs[name] for name in keyword_names},
            ))
        except (ValueError, TypeError) as error:
            raise Error(f"example_input does not match the exported PyTorch model's input structure: {error}") from error
        positional, keyword = tree_unflatten(ordered_leaves, spec)
        ordered_inputs = dict(zip(positional_names, positional))
        ordered_inputs.update(keyword)
        example_inputs = flatten_inputs(ordered_inputs, expected_names)
    else:
        example_inputs = flatten_inputs(example_inputs, input_names)
    input_arg = []
    if is_exported and len(example_inputs) != len(args.input_model._flat_input_is_constant):
        raise Error("The number of example inputs does not match the exported PyTorch model inputs.")
    example_aliases = {}
    for i, example in enumerate(example_inputs):
        name = None
        if isinstance(example, tuple) and len(example) == 2:
            name = example[0]
            example = example[1]
        shape = PartialShape([-1] * example.ndim) if hasattr(example, "ndim") else PartialShape.dynamic()
        if getattr(args, 'dynamo', False):
            # For dynamo mode, use concrete shapes from example; dynamic dims
            # are only introduced when the user explicitly provides 'input'.
            shape = PartialShape(list(example.shape)) if hasattr(example, "shape") else shape
        if is_exported:
            if args.input_model._flat_input_is_constant[i]:
                continue
            exported_name = args.input_model.get_input_signature_name(len(input_arg))
            if name is not None:
                example_aliases[name] = None if name in example_aliases else exported_name
            name = exported_name
            shape = args.input_model.get_input_shape(len(input_arg))
        dtype = getattr(example, "dtype", type(example))
        dtype = pt_to_ov_type_map.get(str(dtype))
        if name:
            input_arg.append(single_input_to_input_cut_info((name, shape, dtype)))
        else:
            input_arg.append(single_input_to_input_cut_info((shape, dtype)))
    if inputs is not None and len(inputs) != 0:
        if len(inputs) == len(input_arg):
            # Match explicit names and unambiguous aliases before positional inputs.
            by_name = {ex.name: ex for ex in input_arg if ex.name is not None}
            matched = [by_name.get(desc.name) for desc in args.input]
            used = {id(ex) for ex in matched if ex is not None}
            aliases = _torchscript_input_aliases(args.input_model)
            aliases.update(example_aliases)
            for i, desc in enumerate(args.input):
                ex = by_name.get(aliases.get(desc.name))
                if matched[i] is None and ex is not None and id(ex) not in used:
                    matched[i] = ex
                    used.add(id(ex))
            leftover = [ex for ex in input_arg if id(ex) not in used]
            new_input = []
            for desc, ex in zip(args.input, matched):
                if ex is None:
                    ex = next((e for e in leftover if desc.name is None or e.name is None), None)
                    if ex is not None:
                        leftover.remove(ex)
                name, shape, dtype = (ex.name, ex.shape, ex.type) if ex is not None else (None, None, None)
                requested_name = desc.name if desc.name is not None else name
                if is_exported and ex is not None and example_aliases.get(desc.name) == ex.name:
                    requested_name = ex.name
                new_input.append(_InputCutInfo(requested_name,
                                               desc.shape if desc.shape is not None else shape,
                                               desc.type if desc.type is not None else dtype,
                                               desc.value))
            input_arg = new_input
        else:
            if is_exported:
                exact = {ex.name for ex in input_arg} | {str(tensor_id) for tensor_id in args.input_model.inputs()}
                args.input = [_InputCutInfo(example_aliases[desc.name], desc.shape, desc.type, desc.value)
                              if desc.name not in exact and example_aliases.get(desc.name) else desc
                              for desc in args.input]
                _validate_exported_input_shapes(args.input_model, args.input)
            return
    if is_exported:
        _validate_exported_input_shapes(args.input_model, input_arg)
    args.input = input_arg


# pylint: disable=no-member
def to_torch_tensor(tensor, preserve_lists=False):
    import torch  # pylint: disable=import-error
    if isinstance(tensor, torch.Tensor):
        return tensor
    if isinstance(tensor, np.ndarray):
        return torch.tensor(tensor)
    if isinstance(tensor, Tensor):
        return torch.tensor(tensor.data)
    if isinstance(tensor, (float, int, bool)):
        return tensor
    if isinstance(tensor, (tuple, list)):
        # TODO: Function to_torch_tensor should be renamed as it handles not only a tensor
        converted = (to_torch_tensor(x, preserve_lists=preserve_lists) for x in tensor)
        return list(converted) if preserve_lists and isinstance(tensor, list) else tuple(converted)
    if isinstance(tensor, dict) and all(isinstance(k, str) for k in tensor.keys()):
        return dict((k, to_torch_tensor(x, preserve_lists=preserve_lists)) for k, x in tensor.items())
    if tensor is None:
        return None
    else:
        raise Error("Unexpected type of example_input. Supported types torch.Tensor, np.array or ov.Tensor. "
                    "Got {}".format(type(tensor)))


def prepare_torch_inputs(example_inputs, preserve_lists=False):
    inputs = None
    if example_inputs is not None:
        inputs = example_inputs
        if isinstance(inputs, list):
            inputs = [to_torch_tensor(x, preserve_lists=preserve_lists) for x in inputs]
        elif isinstance(inputs, tuple):
            inputs = [to_torch_tensor(x, preserve_lists=preserve_lists) for x in inputs]
            inputs = tuple(inputs)
        elif isinstance(inputs, dict):
            for name in inputs:
                assert isinstance(name, str), "Expected dictionary where keys are input names of string type and" \
                                              " values are tensors. Got key of type {}".format(type(name))
            inputs = {name: to_torch_tensor(tensor, preserve_lists=preserve_lists) for name, tensor in inputs.items()}
        else:
            inputs = to_torch_tensor(inputs, preserve_lists=preserve_lists)
    else:
        # No example_input were provided, decoder will use scripting
        return None
    return inputs
