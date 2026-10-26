#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Bytecode model: section/opcode/type enums and the dataclasses they describe."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional


class SectionType(IntEnum):
    """Bytecode section identifiers."""

    FUNCTION = 0x00
    CONSTANT = 0x01
    STRING = 0x02
    KERNEL = 0x03
    TYPE = 0x04
    METADATA = 0x05


class OpCode(IntEnum):
    """Instruction opcodes."""

    ABS_I64 = 0x01
    ADD_I64 = 0x02
    DIV_I64 = 0x03
    MAX_I64 = 0x04
    MIN_I64 = 0x05
    MUL_I64 = 0x06
    REM_I64 = 0x07
    SUB_I64 = 0x08
    ADD_U64 = 0x09
    DIV_U64 = 0x0A
    MAX_U64 = 0x0B
    MIN_U64 = 0x0C
    MUL_U64 = 0x0D
    REM_U64 = 0x0E
    SUB_U64 = 0x0F
    ABS_F64 = 0x10
    ADD_F64 = 0x11
    CEIL_F64 = 0x12
    DIV_F64 = 0x13
    FLOOR_F64 = 0x14
    MAX_F64 = 0x15
    MIN_F64 = 0x16
    MUL_F64 = 0x17
    NEG_F64 = 0x18
    REM_F64 = 0x19
    ROUND_F64 = 0x1A
    SUB_F64 = 0x1B
    AND_64 = 0x1C
    NOT_64 = 0x1D
    OR_64 = 0x1E
    XOR_64 = 0x1F
    SLL_64 = 0x20
    SRL_64 = 0x21
    SRA_64 = 0x22
    CMP_I64 = 0x23
    CMP_F64 = 0x24
    CONVERT_I8_TO_F32 = 0x25
    CONVERT_I8_TO_F64 = 0x26
    CONVERT_I16_TO_I8 = 0x27
    CONVERT_I16_TO_F32 = 0x28
    CONVERT_I16_TO_F64 = 0x29
    CONVERT_I32_TO_I8 = 0x2A
    CONVERT_I32_TO_I16 = 0x2B
    CONVERT_I32_TO_F32 = 0x2C
    CONVERT_I32_TO_F64 = 0x2D
    CONVERT_I64_TO_I8 = 0x2E
    CONVERT_I64_TO_I16 = 0x2F
    CONVERT_I64_TO_I32 = 0x30
    CONVERT_I64_TO_F32 = 0x31
    CONVERT_I64_TO_F64 = 0x32
    CONVERT_F32_TO_I8 = 0x33
    CONVERT_F32_TO_I16 = 0x34
    CONVERT_F32_TO_I32 = 0x35
    CONVERT_F32_TO_I64 = 0x36
    CONVERT_F32_TO_F64 = 0x37
    CONVERT_F64_TO_I8 = 0x38
    CONVERT_F64_TO_I16 = 0x39
    CONVERT_F64_TO_I32 = 0x3A
    CONVERT_F64_TO_I64 = 0x3B
    CONVERT_F64_TO_F32 = 0x3C
    SELECT = 0x3D
    CALL = 0x3E
    RET = 0x3F
    RETV = 0x40
    JMP = 0x41
    JE = 0x42
    JNE = 0x43
    ASSERT = 0x44
    KERNEL_CREATE = 0x45
    CMD_LIST_CREATE = 0x46
    CMD_LIST_ADD_KERNEL = 0x47
    CMD_LIST_CLOSE = 0x48
    CMD_LIST_EXEC = 0x49
    BUFFER_CREATE = 0x4A
    BUFFER_GET_DIM = 0x4B
    BUFFER_SUBVIEW = 0x4C
    BUFFER_VIEW = 0x4D
    BUFFER_STORE = 0x4E
    SET = 0x4F
    SET_IMM = 0x50

    # v1.1.0
    BUFFER_LOAD = 0x51

    # v1.2.0
    BUFFER_FROM_CONST = 0x52
    BUFFER_COPY = 0x53


class TypeCode(IntEnum):
    """Type section identifiers."""

    INTEGER = 0x01
    FLOAT = 0x02
    FUNCTION = 0x05


class FloatTypeFormat(IntEnum):
    """Float format identifiers used by float type entries."""

    IEEE754 = 0x00
    BFLOAT = 0x01
    TFLOAT = 0x02
    E4M3 = 0x03
    E5M2 = 0x04
    E2M1 = 0x05
    E8M0 = 0x06
    NF4 = 0x07


# Maps the lowercase enum names to their byte values, e.g. "ieee754" -> 0x00.
_FLOAT_TYPE_FORMATS: dict[str, int] = {fmt.name.lower(): int(fmt) for fmt in FloatTypeFormat}

# Maps metadata record kind names (used in YAML configs) to their on-disk tag bytes.
_METADATA_RECORD_KINDS: dict[str, int] = {
    "network": 0,
    "input": 1,
    "output": 2,
    "profiling_output": 3,
}

# Maps section name aliases (used in YAML configs) to their SectionType byte values.
_SECTION_TYPE_ALIASES: dict[str, SectionType] = {
    "function": SectionType.FUNCTION,
    "constant": SectionType.CONSTANT,
    "string": SectionType.STRING,
    "kernel": SectionType.KERNEL,
    "type": SectionType.TYPE,
    "metadata": SectionType.METADATA,
}


@dataclass
class _Instruction:
    opcode: int
    operands: bytes = b""  # operands already encoded little-endian


@dataclass
class _FunctionDef:
    name: str
    function_type_index: int
    num_registers_override: Optional[int] = None
    force_num_registers: bool = False
    instructions: list[_Instruction] = field(default_factory=list)
    max_register_index: int = -1


@dataclass
class _Section:
    """In-memory representation of a section body and its section-specific info payload."""

    section_type: SectionType
    name_index: int
    info: bytes
    data: bytes


@dataclass
class _DataEntryInfo:
    offset: int
    size: int


@dataclass
class _FunctionInfo:
    name_index: int
    function_type_index: int
    num_general_registers: int
    body_offset: int
    body_size: int


@dataclass
class _SectionHeader:
    section_type: SectionType
    name_index: int
    offset: int
    size: int
    info: bytes


@dataclass
class _SectionHeaderOverride:
    section_type: int
    section_occurrence: int
    override: dict[str, int]
