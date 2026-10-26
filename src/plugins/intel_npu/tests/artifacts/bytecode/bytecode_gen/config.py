#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""YAML parsing and config-driven generation of bytecode artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Optional

from bytecode_gen.artifact_builder import BytecodeCompatArtifactBuilder
from bytecode_gen.encoders import _u8, _u64
from bytecode_gen.model import (
    OpCode,
    SectionType,
    _FLOAT_TYPE_FORMATS,
    _METADATA_RECORD_KINDS,
    _SECTION_TYPE_ALIASES,
)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyYAML is required. Install it with: python3 -m pip install pyyaml") from exc

    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError("Top-level YAML node must be a mapping")
    return content


def _resolve_type_ref(ref: Any, type_indices: dict[str, int]) -> int:
    if isinstance(ref, int):
        return ref
    if not isinstance(ref, str):
        raise ValueError(f"Type reference must be int or str, got: {type(ref).__name__}")
    if ref not in type_indices:
        raise ValueError(f"Unknown type reference: '{ref}'")
    return type_indices[ref]


def _apply_instruction(fn: Any, instruction: dict[str, Any]) -> None:
    op_name = instruction.get("op")
    if not isinstance(op_name, (str, int)):
        raise ValueError("Each instruction must define field 'op' as string helper name or integer opcode")

    if isinstance(op_name, int):
        args = instruction.get("args", [])
        if not isinstance(args, list):
            raise ValueError("Instruction with integer 'op' expects list in 'args'")
        fn.instruction(op_name, *args)
        return

    if op_name == "instruction":
        opcode_name = instruction.get("opcode")
        args = instruction.get("args", [])
        if not isinstance(opcode_name, str):
            raise ValueError("Generic 'instruction' entry requires string field 'opcode'")
        if not isinstance(args, list):
            raise ValueError("Instruction 'args' must be a list")
        try:
            opcode = OpCode[opcode_name]
        except KeyError as exc:
            raise ValueError(f"Unknown opcode enum: '{opcode_name}'") from exc
        fn.instruction(opcode, *args)
        return

    method = getattr(fn, op_name, None)
    if method is None:
        raise ValueError(f"Unknown instruction helper: '{op_name}'")

    if "args" in instruction:
        args = instruction["args"]
        if not isinstance(args, list):
            raise ValueError(f"Instruction '{op_name}' expects list in 'args'")
        method(*args)
        return

    kwargs = {k: v for k, v in instruction.items() if k != "op"}
    method(**kwargs)


def _decode_section_payload(payload: Any, section_name: str) -> bytes:
    if isinstance(payload, list):
        if not all(isinstance(v, int) and 0 <= v <= 255 for v in payload):
            raise ValueError(
                f"Section '{section_name}' list entries must contain only byte values in range [0, 255]"
            )
        return bytes(payload)

    if isinstance(payload, str):
        # Accept plain UTF-8 text payloads or explicit 0x-prefixed hex payloads.
        if payload.startswith("0x") or payload.startswith("0X"):
            hex_digits = payload[2:]
            if len(hex_digits) % 2 != 0:
                raise ValueError(
                    f"Section '{section_name}' has odd-length hex payload '{payload}'"
                )
            try:
                return bytes.fromhex(hex_digits)
            except ValueError as exc:
                raise ValueError(
                    f"Section '{section_name}' has invalid hex payload '{payload}'"
                ) from exc
        return payload.encode("utf-8")

    raise ValueError(
        f"Unsupported payload in section '{section_name}'; expected list[int] or str"
    )


def _get_section_entries(config: dict[str, Any], key: str) -> list[Any]:
    entries = config.get(key, [])
    if not isinstance(entries, list):
        raise ValueError(f"Section '{key}' must be a list")
    return entries


def _add_payload_entries(config: dict[str, Any], key: str, add_fn: Any) -> None:
    """Decode each entry of a raw payload section and forward it to the builder."""
    for entry in _get_section_entries(config, key):
        add_fn(_decode_section_payload(entry, key))


def _resolve_constant_ref(ref: Any) -> int:
    if not isinstance(ref, int):
        raise ValueError(f"Constant reference must be int (constant index), got: {type(ref).__name__}")
    if ref < 0:
        raise ValueError(f"Constant reference must be >= 0, got: {ref}")
    return ref


def _encode_metadata_network_entry(entry: dict[str, Any], builder: BytecodeCompatArtifactBuilder) -> bytes:
    name = entry.get("name")
    num_streams = entry.get("num_streams", 1)
    num_cmdlists = entry.get("num_cmdlists", 1)

    if not isinstance(name, str):
        raise ValueError("Metadata 'network' entry requires string field 'name'")
    if not isinstance(num_streams, int) or num_streams < 0:
        raise ValueError("Metadata 'network' entry field 'num_streams' must be a non-negative integer")
    if not isinstance(num_cmdlists, int) or num_cmdlists < 0:
        raise ValueError("Metadata 'network' entry field 'num_cmdlists' must be a non-negative integer")

    out = bytearray()
    out.extend(_u8(_METADATA_RECORD_KINDS["network"]))
    out.extend(_u64(builder.intern_string(name)))
    out.extend(_u64(num_streams))
    out.extend(_u64(num_cmdlists))
    return bytes(out)


def _encode_metadata_descriptor_entry(
    entry: dict[str, Any],
    kind: str,
    builder: BytecodeCompatArtifactBuilder,
    type_indices: dict[str, int],
) -> bytes:
    name = entry.get("name")
    precision_type_ref = entry.get("precision_type")
    shape_ref = entry.get("shape")
    index_used_by_driver = entry.get("index_used_by_driver")
    has_dynamic_strides = entry.get("has_dynamic_strides")
    output_tensor_names = entry.get("output_tensor_names", [])
    shape_from_ir_model_ref = entry.get("shape_from_ir_model")
    node_friendly_name = entry.get("node_friendly_name")

    if not isinstance(name, str):
        raise ValueError(f"Metadata '{kind}' entry requires string field 'name'")
    if precision_type_ref is None:
        raise ValueError(f"Metadata '{kind}' entry requires field 'precision_type'")
    if shape_ref is None:
        raise ValueError(f"Metadata '{kind}' entry requires field 'shape'")
    if not isinstance(index_used_by_driver, int) or index_used_by_driver < 0 or index_used_by_driver > 0xFFFFFFFF:
        raise ValueError(
            f"Metadata '{kind}' entry field 'index_used_by_driver' must be a uint32 integer"
        )
    if not isinstance(has_dynamic_strides, bool):
        raise ValueError(f"Metadata '{kind}' entry field 'has_dynamic_strides' must be a bool")
    if not isinstance(output_tensor_names, list):
        raise ValueError(f"Metadata '{kind}' entry field 'output_tensor_names' must be a list when provided")
    if len(output_tensor_names) > 255:
        raise ValueError(f"Metadata '{kind}' entry field 'output_tensor_names' must contain at most 255 values")

    precision_type_index = _resolve_type_ref(precision_type_ref, type_indices)
    shape_index = _resolve_constant_ref(shape_ref)

    out = bytearray()
    out.extend(_u8(_METADATA_RECORD_KINDS[kind]))
    out.extend(_u64(builder.intern_string(name)))
    out.extend(_u64(precision_type_index))
    out.extend(_u64(shape_index))
    out.extend(index_used_by_driver.to_bytes(4, "little", signed=False))
    out.extend(_u8(1 if has_dynamic_strides else 0))

    out.extend(_u8(len(output_tensor_names)))
    for tensor_name in output_tensor_names:
        if not isinstance(tensor_name, str):
            raise ValueError(f"Metadata '{kind}' output tensor names must be strings")
        out.extend(_u64(builder.intern_string(tensor_name)))

    has_shape_from_ir_model = shape_from_ir_model_ref is not None
    out.extend(_u8(1 if has_shape_from_ir_model else 0))
    if has_shape_from_ir_model:
        out.extend(_u64(_resolve_constant_ref(shape_from_ir_model_ref)))

    has_node_friendly_name = node_friendly_name is not None
    out.extend(_u8(1 if has_node_friendly_name else 0))
    if has_node_friendly_name:
        if not isinstance(node_friendly_name, str):
            raise ValueError(f"Metadata '{kind}' field 'node_friendly_name' must be a string when provided")
        out.extend(_u64(builder.intern_string(node_friendly_name)))

    return bytes(out)


def _add_metadata_entries(
    config: dict[str, Any],
    builder: BytecodeCompatArtifactBuilder,
    type_indices: dict[str, int],
) -> None:
    entries = _get_section_entries(config, "metadata_section")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each metadata entry in 'metadata_section' must be a mapping")

        kind = entry.get("kind")
        if not isinstance(kind, str) or kind not in _METADATA_RECORD_KINDS:
            supported = ", ".join(sorted(_METADATA_RECORD_KINDS.keys()))
            raise ValueError(f"Metadata entry requires field 'kind' with one of: {supported}")

        if kind == "network":
            builder.add_metadata_data(_encode_metadata_network_entry(entry, builder))
        else:
            builder.add_metadata_data(_encode_metadata_descriptor_entry(entry, kind, builder, type_indices))


_SECTION_KEY_TO_TYPE: dict[str, SectionType] = {
    "function_section": SectionType.FUNCTION,
    "constant_section": SectionType.CONSTANT,
    "string_section": SectionType.STRING,
    "kernel_section": SectionType.KERNEL,
    "type_section": SectionType.TYPE,
    "metadata_section": SectionType.METADATA,
}


def _parse_magic_number(config: dict[str, Any]) -> bytes:
    magic_number = config.get("magic_number", b"NPUByte\x00")
    if isinstance(magic_number, str):
        return magic_number.encode("utf-8")
    return magic_number


def _parse_version(config: dict[str, Any]) -> tuple[int, int, int]:
    version = config.get("version", [1, 0, 0])
    if not (isinstance(version, list) and len(version) == 3 and all(isinstance(v, int) for v in version)):
        raise ValueError("'version' must be a list of three integers, e.g. [1, 0, 0]")
    return tuple(version)


def _collect_explicit_empty_sections(config: dict[str, Any]) -> set[SectionType]:
    """Return section types that appear in the config but carry no entries."""
    explicit_empty: set[SectionType] = set()
    for section_key, section_type in _SECTION_KEY_TO_TYPE.items():
        if section_key in config and len(_get_section_entries(config, section_key)) == 0:
            explicit_empty.add(section_type)
    return explicit_empty


def _apply_explicit_string_section(builder: BytecodeCompatArtifactBuilder, entries: list[Any]) -> None:
    if not all(isinstance(v, str) for v in entries):
        raise ValueError("Section 'string_section' entries must be strings")
    builder.set_explicit_strings(list(entries))


# Sections whose content is otherwise auto-derived by the builder. Declaring such a
# section in the config (even as an empty list) makes the config authoritative and
# overrides the derived content, mirroring how every other section is sourced
# directly from the config. This allows crafting intentionally malformed artifacts,
# for example an empty string section still referenced by a function.
_EXPLICIT_SECTION_APPLIERS: dict[str, Callable[[BytecodeCompatArtifactBuilder, list[Any]], None]] = {
    "string_section": _apply_explicit_string_section,
}


def _apply_explicit_sections(config: dict[str, Any], builder: BytecodeCompatArtifactBuilder) -> None:
    """Override auto-derived section content with explicit entries from the config."""
    for section_key, apply_fn in _EXPLICIT_SECTION_APPLIERS.items():
        if section_key in config:
            apply_fn(builder, _get_section_entries(config, section_key))


def _build_type_section(config: dict[str, Any], builder: BytecodeCompatArtifactBuilder) -> dict[str, int]:
    """Add type entries to the builder and return a name -> index map.

    Function types are deferred until all scalar types are registered so they can
    reference parameter and result types by name regardless of declaration order.
    """
    type_indices: dict[str, int] = {}
    pending_function_types: list[tuple[str, list[Any], list[Any]]] = []
    for type_cfg in _get_section_entries(config, "type_section"):
        if not isinstance(type_cfg, dict):
            raise ValueError("Each type entry must be a mapping")
        type_id = type_cfg.get("id")
        type_kind = type_cfg.get("kind", "integer")
        if not isinstance(type_id, str):
            raise ValueError("Type entry requires string field 'id'")
        if not isinstance(type_kind, str):
            raise ValueError("Type entry field 'kind' must be a string when provided")

        if type_kind == "integer":
            width = type_cfg.get("width")
            is_signed = type_cfg.get("is_signed")
            if not isinstance(width, int) or not isinstance(is_signed, bool):
                raise ValueError("Integer type requires 'width' (int) and 'is_signed' (bool)")
            type_indices[type_id] = builder.add_integer_type(width=width, is_signed=is_signed)
        elif type_kind == "float":
            width = type_cfg.get("width")
            fmt_name = type_cfg.get("format", "ieee754")
            if not isinstance(width, int):
                raise ValueError("Float type requires 'width' (int)")
            if not isinstance(fmt_name, str):
                raise ValueError("Float type field 'format' must be a string")
            fmt = _FLOAT_TYPE_FORMATS.get(fmt_name.lower())
            if fmt is None:
                supported = ", ".join(sorted(_FLOAT_TYPE_FORMATS.keys()))
                raise ValueError(f"Unsupported float format '{fmt_name}'. Supported values: {supported}")
            type_indices[type_id] = builder.add_float_type(width=width, fmt=fmt)
        elif type_kind == "function":
            params = type_cfg.get("params", [])
            results = type_cfg.get("results", [])
            if not isinstance(params, list) or not isinstance(results, list):
                raise ValueError("Function type requires list fields 'params' and 'results'")
            pending_function_types.append((type_id, params, results))
        else:
            raise ValueError(f"Unsupported type kind: '{type_kind}'")

    for type_id, params, results in pending_function_types:
        param_indices = [_resolve_type_ref(ref, type_indices) for ref in params]
        result_indices = [_resolve_type_ref(ref, type_indices) for ref in results]
        type_indices[type_id] = builder.add_function_type(
            param_type_indices=param_indices,
            result_type_indices=result_indices,
        )

    return type_indices


def _build_unknown_sections(config: dict[str, Any], builder: BytecodeCompatArtifactBuilder) -> None:
    for raw_sec in config.get("unknown_sections", []):
        if not isinstance(raw_sec, dict):
            raise ValueError("Each 'unknown_sections' entry must be a mapping")
        section_type_byte = raw_sec.get("section_type")
        data_payload = raw_sec.get("data", [])
        if not isinstance(section_type_byte, int):
            raise ValueError("'unknown_sections' entry requires integer field 'section_type'")
        if not isinstance(data_payload, list) or not all(isinstance(b, int) and 0 <= b <= 255 for b in data_payload):
            raise ValueError("'unknown_sections' entry 'data' must be a list of byte values (0-255)")
        builder.add_unknown_section(section_type_byte, bytes(data_payload))


def _resolve_override_section_type(section_type: Any) -> int:
    if isinstance(section_type, str):
        resolved = _SECTION_TYPE_ALIASES.get(section_type.lower())
        if resolved is None:
            supported = ", ".join(sorted(_SECTION_TYPE_ALIASES.keys()))
            raise ValueError(
                f"'section_header_overrides.section_type' unsupported value '{section_type}'. "
                f"Supported values: {supported}"
            )
        return int(resolved)
    if isinstance(section_type, int):
        if not (0 <= section_type <= 255):
            raise ValueError("'section_header_overrides.section_type' integer must be in range [0, 255]")
        return section_type
    raise ValueError("'section_header_overrides.section_type' must be a string alias or integer byte")


def _build_section_header_overrides(config: dict[str, Any], builder: BytecodeCompatArtifactBuilder) -> None:
    section_header_overrides = config.get("section_header_overrides", [])
    if not isinstance(section_header_overrides, list):
        raise ValueError("'section_header_overrides' must be a list when provided")
    for override in section_header_overrides:
        if not isinstance(override, dict):
            raise ValueError("Each 'section_header_overrides' entry must be a mapping")
        if "section_index" in override:
            raise ValueError("'section_header_overrides.section_index' is no longer supported; use 'section_type'")
        if override.get("section_type") is None:
            raise ValueError("'section_header_overrides.section_type' is required")

        builder.add_section_header_override(
            section_type=_resolve_override_section_type(override["section_type"]),
            section_occurrence=override.get("section_occurrence", 0),
            name_index=override.get("name_index"),
            offset=override.get("offset"),
            size=override.get("size"),
        )


def _build_functions(
    config: dict[str, Any],
    builder: BytecodeCompatArtifactBuilder,
    type_indices: dict[str, int],
    allow_zero_sections: bool,
) -> None:
    functions = _get_section_entries(config, "function_section")
    if not functions and not allow_zero_sections:
        raise ValueError("'function_section' must be a non-empty list unless 'allow_zero_sections: true' is set")

    for fn_cfg in functions:
        if not isinstance(fn_cfg, dict):
            raise ValueError("Each function entry must be a mapping")
        name = fn_cfg.get("name")
        if not isinstance(name, str):
            raise ValueError("Function requires string field 'name'")

        fn_type_ref = fn_cfg.get("type")
        if fn_type_ref is None:
            raise ValueError(
                f"Function '{name}' must define field 'type' referencing a function type in the type section"
            )
        if not isinstance(fn_type_ref, (int, str)):
            raise ValueError("Function field 'type' must be int or str")

        num_registers = fn_cfg.get("num_registers")
        if num_registers is not None and not isinstance(num_registers, int):
            raise ValueError("'num_registers' must be integer when provided")

        force_num_registers = fn_cfg.get("force_num_registers", False)
        if not isinstance(force_num_registers, bool):
            raise ValueError("'force_num_registers' must be a bool when provided")

        fn = builder.begin_function(
            name=name,
            function_type_index=_resolve_type_ref(fn_type_ref, type_indices),
            num_registers=num_registers,
            force_num_registers=force_num_registers,
        )

        instructions = fn_cfg.get("instructions", [])
        if not isinstance(instructions, list):
            raise ValueError(f"Function '{name}' field 'instructions' must be a list")
        for ins in instructions:
            if not isinstance(ins, dict):
                raise ValueError(f"Function '{name}' instruction entries must be mappings")
            _apply_instruction(fn, ins)


def _parse_truncation_config(config: dict[str, Any]) -> Optional[tuple[str, int]]:
    """Parse truncate_bytes_from_end config and return (section_name, num_bytes) or None."""
    truncate_config = config.get("truncate_bytes_from_end")
    if truncate_config is None:
        return None

    if not isinstance(truncate_config, dict):
        raise ValueError("'truncate_bytes_from_end' must be a dict with 'section' and 'bytes' keys")

    section_name = truncate_config.get("section")
    num_bytes = truncate_config.get("bytes")

    if not isinstance(section_name, str):
        raise ValueError("'truncate_bytes_from_end.section' must be a string (section name)")
    if not isinstance(num_bytes, int) or num_bytes <= 0:
        raise ValueError("'truncate_bytes_from_end.bytes' must be a positive integer")

    return section_name, num_bytes


def build_from_config(config: dict[str, Any], output_path: Path | None = None) -> Path:
    allow_zero_sections = config.get("allow_zero_sections", False)
    if not isinstance(allow_zero_sections, bool):
        raise ValueError("'allow_zero_sections' must be a bool when provided")

    # Parse truncation config early so it can be passed to the builder.
    truncation_config = _parse_truncation_config(config)

    builder = BytecodeCompatArtifactBuilder(
        magic_number=_parse_magic_number(config),
        version=_parse_version(config),
        allow_zero_sections=allow_zero_sections,
        emit_empty_sections=_collect_explicit_empty_sections(config),
        truncation_config=truncation_config,
    )

    # Section build order mirrors BytecodeWriter and fixes string interning order.
    _add_payload_entries(config, "constant_section", builder.add_constant_data)
    _add_payload_entries(config, "kernel_section", builder.add_kernel_data)
    type_indices = _build_type_section(config, builder)
    _add_metadata_entries(config, builder, type_indices)
    _build_unknown_sections(config, builder)
    _build_section_header_overrides(config, builder)
    _build_functions(config, builder, type_indices, allow_zero_sections)

    # Apply config-authoritative content for sections that are otherwise auto-derived.
    _apply_explicit_sections(config, builder)

    output_value = output_path if output_path is not None else config.get("output")
    if output_value is None:
        raise ValueError("Output path missing. Provide 'output' in YAML or --output-dir CLI argument")

    output_path = Path(output_value)
    builder.write(output_path)
    return output_path
