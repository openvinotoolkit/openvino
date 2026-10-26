#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""High-level builder that assembles bytecode compatibility binary artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from bytecode_gen.encoders import _u8, _u16, _u64
from bytecode_gen.function_builder import FunctionBuilder
from bytecode_gen.model import (
    SectionType,
    TypeCode,
    _DataEntryInfo,
    _FunctionDef,
    _FunctionInfo,
    _Section,
    _SectionHeader,
    _SectionHeaderOverride,
)


class BytecodeCompatArtifactBuilder:
    """High-level builder for bytecode compatibility binary artifacts."""

    def __init__(
        self,
        magic_number: bytes,
        version: tuple[int, int, int],
        allow_zero_sections: bool = False,
        emit_empty_sections: Optional[set[SectionType]] = None,
        truncation_config: Optional[tuple[str, int]] = None,
    ):
        self._magic_number = magic_number
        self._version = version
        self._allow_zero_sections = allow_zero_sections
        self._emit_empty_sections = set(emit_empty_sections or ())
        self._strings: list[str] = []
        self._string_to_index: dict[str, int] = {}
        self._type_entries: list[bytes] = []
        self._functions: list[_FunctionDef] = []
        self._constant_entries: list[bytes] = []
        self._kernel_entries: list[bytes] = []
        self._metadata_entries: list[bytes] = []
        self._unknown_sections: list[tuple[int, bytes]] = []
        self._section_header_overrides: list[_SectionHeaderOverride] = []
        # Maps section type (by name) to (start_offset, end_offset) in the last built binary.
        # Populated by build() for use with section-specific truncation.
        self._section_boundaries: dict[str, tuple[int, int]] = {}
        # Truncation config: (section_name, num_bytes) or None.
        # Truncates N bytes from end of specified section during build().
        self._truncation_config = truncation_config
        # Validate and store truncation config.
        if self._truncation_config is not None:
            if not isinstance(self._truncation_config, tuple) or len(self._truncation_config) != 2:
                raise ValueError("truncation_config must be a tuple of (section_name, num_bytes)")
            truncate_section, truncate_bytes = self._truncation_config
            if not isinstance(truncate_section, str):
                raise ValueError("truncation_config[0] (section_name) must be a string")
            if not isinstance(truncate_bytes, int) or truncate_bytes <= 0:
                raise ValueError("truncation_config[1] (num_bytes) must be a positive integer")

    def add_constant_data(self, data: bytes) -> int:
        self._constant_entries.append(data)
        return len(self._constant_entries) - 1

    def add_kernel_data(self, data: bytes) -> int:
        self._kernel_entries.append(data)
        return len(self._kernel_entries) - 1

    def add_metadata_data(self, data: bytes) -> int:
        self._metadata_entries.append(data)
        return len(self._metadata_entries) - 1

    def add_unknown_section(self, section_type_byte: int, data: bytes) -> None:
        """Append a section with an unrecognized type byte, for forward-compatibility testing."""
        if not isinstance(section_type_byte, int) or not (0 <= section_type_byte <= 255):
            raise ValueError("section_type_byte must be an integer in [0, 255]")
        self._unknown_sections.append((section_type_byte, data))

    def add_section_header_override(
        self,
        *,
        section_type: int,
        section_occurrence: int = 0,
        name_index: Optional[int] = None,
        offset: Optional[int] = None,
        size: Optional[int] = None,
    ) -> None:
        if not isinstance(section_type, int) or not (0 <= section_type <= 255):
            raise ValueError("section_type must be an integer in range [0, 255]")

        if not isinstance(section_occurrence, int) or section_occurrence < 0:
            raise ValueError("section_occurrence must be a non-negative integer")

        override: dict[str, int] = {}
        if name_index is not None:
            if not isinstance(name_index, int) or not (0 <= name_index <= 0xFFFFFFFFFFFFFFFF):
                raise ValueError("name_index override must be an integer in range [0, 2^64-1]")
            override["name_index"] = name_index
        if offset is not None:
            if not isinstance(offset, int) or not (0 <= offset <= 0xFFFFFFFFFFFFFFFF):
                raise ValueError("offset override must be an integer in range [0, 2^64-1]")
            override["offset"] = offset
        if size is not None:
            if not isinstance(size, int) or not (0 <= size <= 0xFFFFFFFFFFFFFFFF):
                raise ValueError("size override must be an integer in range [0, 2^64-1]")
            override["size"] = size

        if not override:
            raise ValueError("At least one of name_index, offset or size must be provided in section header override")

        self._section_header_overrides.append(
            _SectionHeaderOverride(
                section_type=section_type,
                section_occurrence=section_occurrence,
                override=override,
            )
        )

    def add_integer_type(self, *, width: int, is_signed: bool) -> int:
        if width <= 0 or width > 255:
            raise ValueError("integer type width must be in range [1, 255]")
        entry = bytes([int(TypeCode.INTEGER), width, 1 if is_signed else 0])
        self._type_entries.append(entry)
        return len(self._type_entries) - 1

    def add_float_type(self, *, width: int, fmt: int) -> int:
        if width <= 0 or width > 255:
            raise ValueError("float type width must be in range [1, 255]")
        if fmt < 0 or fmt > 255:
            raise ValueError("float type format must be in range [0, 255]")
        entry = bytes([int(TypeCode.FLOAT), width, fmt])
        self._type_entries.append(entry)
        return len(self._type_entries) - 1

    def add_function_type(self, *, param_type_indices: list[int], result_type_indices: list[int]) -> int:
        entry = self._encode_function_type(param_type_indices, result_type_indices)
        self._type_entries.append(entry)
        return len(self._type_entries) - 1

    def begin_function(
        self,
        *,
        name: str,
        function_type_index: int,
        num_registers: Optional[int] = None,
        force_num_registers: bool = False,
    ) -> FunctionBuilder:
        self.intern_string(name)
        fn = _FunctionDef(
            name=name,
            function_type_index=function_type_index,
            num_registers_override=num_registers,
            force_num_registers=force_num_registers,
        )
        self._functions.append(fn)
        return FunctionBuilder(fn)

    def build(self) -> bytes:
        sections_in_file_order = self._collect_sections_in_file_order()
        if not sections_in_file_order and not self._allow_zero_sections:
            raise ValueError("artifact must contain at least one non-empty section")

        header_prefix = self._encode_header_prefix()
        section_header_table = self._encode_section_header_table(
            sections_in_file_order,
            header_prefix_size=len(header_prefix),
        )

        out = bytearray()
        out.extend(header_prefix)
        out.extend(section_header_table)

        # Track section boundaries for potential truncation.
        self._section_boundaries.clear()
        for section in sections_in_file_order:
            section_start = len(out)
            out.extend(section.data)
            section_end = len(out)
            # Map known section type enum names to (start, end) offsets.
            # Unknown/raw section types (plain int) are not addressable by name.
            if isinstance(section.section_type, SectionType):
                section_name = section.section_type.name.lower()
                self._section_boundaries[section_name] = (section_start, section_end)

        # Apply truncation if configured.
        if self._truncation_config is not None:
            out = self._apply_truncation_to_buffer(bytes(out))

        return bytes(out)

    def _apply_truncation_to_buffer(self, data: bytes) -> bytes:
        """Truncate the binary buffer according to configured truncation parameters."""
        section_name, num_bytes = self._truncation_config

        if section_name not in self._section_boundaries:
            supported = ", ".join(sorted(self._section_boundaries.keys()))
            raise ValueError(
                f"truncate_section '{section_name}' not found in binary. "
                f"Available sections: {supported}"
            )

        section_start, section_end = self._section_boundaries[section_name]
        truncate_point = section_end - num_bytes
        if truncate_point <= section_start:
            raise ValueError(
                f"truncate_bytes ({num_bytes}) would remove entire section '{section_name}' "
                f"(section size: {section_end - section_start} bytes)"
            )
        return data[:truncate_point]

    def _collect_sections_in_file_order(self) -> list[_Section]:
        """Collect all non-empty sections in file order from the bytecode spec."""
        sections: list[_Section] = []

        # Keep section order aligned with BytecodeWriter::appendSections.
        if self._functions or SectionType.FUNCTION in self._emit_empty_sections:
            func_blob, func_info = self._build_function_section_with_function_info()
            sections.append(
                _Section(
                    section_type=SectionType.FUNCTION,
                    name_index=0,
                    info=func_info,
                    data=func_blob,
                )
            )

        sections.extend(
            self._build_raw_section(
                SectionType.CONSTANT,
                self._constant_entries,
                force_emit_empty=SectionType.CONSTANT in self._emit_empty_sections,
            )
        )
        sections.extend(
            self._build_raw_section(
                SectionType.KERNEL,
                self._kernel_entries,
                force_emit_empty=SectionType.KERNEL in self._emit_empty_sections,
            )
        )

        # Strings are stored NUL-terminated UTF-8, so they cannot reuse the generic
        # raw-section path that treats entries as opaque bytes.
        str_blob, str_offsets = self._pack_data_section(
            [value.encode("utf-8") + b"\x00" for value in self._strings]
        )
        if self._strings or SectionType.STRING in self._emit_empty_sections:
            sections.append(
                _Section(
                    section_type=SectionType.STRING,
                    name_index=0,
                    info=self._encode_data_section_info_entries(str_offsets),
                    data=str_blob,
                )
            )

        # Type entries are opaque byte blobs and use the generic data-section layout.
        sections.extend(
            self._build_raw_section(
                SectionType.TYPE,
                self._type_entries,
                force_emit_empty=SectionType.TYPE in self._emit_empty_sections,
            )
        )

        sections.extend(
            self._build_raw_section(
                SectionType.METADATA,
                self._metadata_entries,
                force_emit_empty=SectionType.METADATA in self._emit_empty_sections,
            )
        )

        for section_type_byte, data in self._unknown_sections:
            sections.append(
                _Section(
                    section_type=section_type_byte,
                    name_index=0,
                    info=_u64(0),
                    data=data,
                )
            )

        return sections

    def _build_raw_section(
        self,
        section_type: SectionType,
        entries: list[bytes],
        *,
        force_emit_empty: bool = False,
    ) -> list[_Section]:
        if not entries and not force_emit_empty:
            return []
        blob, offsets = self._pack_data_section(entries)
        return [
            _Section(
                section_type=section_type,
                name_index=0,
                info=self._encode_data_section_info_entries(offsets),
                data=blob,
            )
        ]

    def _encode_header_prefix(self) -> bytes:
        out = bytearray()
        out.extend(self._magic_number)
        out.extend(_u16(self._version[0]))
        out.extend(_u16(self._version[1]))
        out.extend(_u16(self._version[2]))
        return bytes(out)

    def _encode_section_header_table(self, sections: list[_Section], header_prefix_size: int) -> bytes:
        section_headers = self._build_section_headers(sections, header_prefix_size)

        out = bytearray()
        out.extend(_u64(len(section_headers)))

        for section_header in section_headers:
            out.extend(_u8(int(section_header.section_type)))
            out.extend(_u64(section_header.name_index))
            out.extend(_u64(section_header.offset))
            out.extend(_u64(section_header.size))
            out.extend(section_header.info)

        return bytes(out)

    def _build_section_headers(self, sections: list[_Section], header_prefix_size: int) -> list[_SectionHeader]:
        # section_header_table = num_sections + section_header[]
        # section_header = type + name_index + offset + size + info
        table_size = 8 + sum((1 + 8 + 8 + 8 + len(section.info)) for section in sections)
        next_section_offset = header_prefix_size + table_size

        headers: list[_SectionHeader] = []
        for section in sections:
            headers.append(
                _SectionHeader(
                    section_type=section.section_type,
                    name_index=section.name_index,
                    offset=next_section_offset,
                    size=len(section.data),
                    info=section.info,
                )
            )
            next_section_offset += len(section.data)

        for override_request in self._section_header_overrides:
            target_type = int(override_request.section_type)
            matching_indices = [idx for idx, header in enumerate(headers) if int(header.section_type) == target_type]
            if override_request.section_occurrence >= len(matching_indices):
                raise ValueError(
                    f"section_header_overrides type {target_type} occurrence "
                    f"{override_request.section_occurrence} is out of range for {len(matching_indices)} matches"
                )
            section_index = matching_indices[override_request.section_occurrence]

            current = headers[section_index]
            override = override_request.override
            headers[section_index] = _SectionHeader(
                section_type=current.section_type,
                name_index=override.get("name_index", current.name_index),
                offset=override.get("offset", current.offset),
                size=override.get("size", current.size),
                info=current.info,
            )

        return headers

    def write(self, output_path: str | Path) -> Path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.build())
        return path

    @staticmethod
    def _pack_data_section(blobs: list[bytes]) -> tuple[bytes, list[_DataEntryInfo]]:
        """Concatenate entries into one blob and record each entry's (offset, size).

        Every data-style section (string, type, constant, kernel, metadata) shares this
        layout: a packed blob plus an info table describing where each entry lives.
        """
        blob = bytearray()
        entries: list[_DataEntryInfo] = []
        for data in blobs:
            entries.append(_DataEntryInfo(offset=len(blob), size=len(data)))
            blob.extend(data)
        return bytes(blob), entries

    def _build_function_section_with_function_info(self) -> tuple[bytes, bytes]:
        body_blob = bytearray()
        function_info_list: list[_FunctionInfo] = []

        for fn in self._functions:
            body_offset = len(body_blob)
            fn_body = self._encode_function_body(fn)
            body_blob.extend(fn_body)

            inferred_regs = fn.max_register_index + 1 if fn.max_register_index >= 0 else 0
            num_regs = fn.num_registers_override if fn.num_registers_override is not None else inferred_regs
            if num_regs < inferred_regs and not fn.force_num_registers:
                raise ValueError(
                    f"function '{fn.name}' uses {inferred_regs} registers, but num_registers is set to {num_regs}"
                )

            function_info_list.append(
                _FunctionInfo(
                    name_index=self._string_to_index.get(fn.name, 0),
                    function_type_index=self._resolve_function_type_index(fn),
                    num_general_registers=num_regs,
                    body_offset=body_offset,
                    body_size=len(fn_body),
                )
            )

        info = bytearray()
        info.extend(_u64(len(function_info_list)))
        info.extend(_u64(0))
        for function_info in function_info_list:
            info.extend(_u64(function_info.name_index))
            info.extend(_u64(function_info.function_type_index))
            info.extend(_u64(function_info.num_general_registers))
            info.extend(_u64(function_info.body_offset))
            info.extend(_u64(function_info.body_size))

        return bytes(body_blob), bytes(info)

    def _encode_function_body(self, fn: _FunctionDef) -> bytes:
        encoded = bytearray()
        for ins in fn.instructions:
            encoded.extend(_u16(ins.opcode))
            encoded.extend(ins.operands)
        return bytes(encoded)

    def _encode_data_section_info_entries(self, entries: list[_DataEntryInfo]) -> bytes:
        out = bytearray()
        out.extend(_u64(len(entries)))
        for entry in entries:
            out.extend(_u64(entry.offset))
            out.extend(_u64(entry.size))
        return bytes(out)

    def _encode_function_type(self, params: list[int], results: list[int]) -> bytes:
        out = bytearray()
        out.extend(_u8(int(TypeCode.FUNCTION)))
        out.extend(_u16(len(params)))
        for p in params:
            out.extend(_u64(p))
        out.extend(_u16(len(results)))
        for r in results:
            out.extend(_u64(r))
        return bytes(out)

    def _resolve_function_type_index(self, fn: _FunctionDef) -> int:
        if fn.function_type_index < 0 or fn.function_type_index >= len(self._type_entries):
            raise ValueError(f"function '{fn.name}' has out-of-range function type index {fn.function_type_index}")
        type_entry = self._type_entries[fn.function_type_index]
        if not type_entry or type_entry[0] != int(TypeCode.FUNCTION):
            raise ValueError(
                f"function '{fn.name}' references non-function type at index {fn.function_type_index}"
            )
        return fn.function_type_index

    def intern_string(self, value: str) -> int:
        """Return the index of an interned string, adding it on first use."""
        if value in self._string_to_index:
            return self._string_to_index[value]
        idx = len(self._strings)
        self._strings.append(value)
        self._string_to_index[value] = idx
        return idx

    def set_explicit_strings(self, values: list[str]) -> None:
        """Replace auto-interned strings with an explicit list (for malformed-artifact tests)."""
        self._strings = list(values)
        self._string_to_index = {value: idx for idx, value in enumerate(self._strings)}
