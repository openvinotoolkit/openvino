#
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

"""Function-level instruction builder for the bytecode artifact format."""

from __future__ import annotations

from typing import Iterable

from bytecode_gen.encoders import _i16, _i64
from bytecode_gen.model import OpCode, _FunctionDef, _Instruction


class FunctionBuilder:
    """Builds one function body as a stream of (opcode, operands) instructions.

    Operand encoding follows the bytecode format: register indices are signed 16-bit,
    immediates and jump offsets are signed 64-bit. Variable-length register lists are
    length-prefixed with an i16 count, except where the format omits the count because
    the length is implied by an earlier operand (for example buffer strides).
    """

    def __init__(self, fn: _FunctionDef):
        self._fn = fn

    def instruction(self, opcode: OpCode, *registers: int) -> FunctionBuilder:
        """Append an instruction whose operands are all register indices."""
        return self._emit(opcode, self._regs(registers))

    def set_imm(self, dst: int, value: int) -> FunctionBuilder:
        return self._emit(OpCode.SET_IMM, self._reg(dst) + _i64(value))

    def jmp(self, offset: int) -> FunctionBuilder:
        return self._emit(OpCode.JMP, _i64(offset))

    def je(self, offset: int, lhs: int, rhs: int) -> FunctionBuilder:
        return self._emit(OpCode.JE, _i64(offset) + self._reg(lhs) + self._reg(rhs))

    def jne(self, offset: int, lhs: int, rhs: int) -> FunctionBuilder:
        return self._emit(OpCode.JNE, _i64(offset) + self._reg(lhs) + self._reg(rhs))

    def retv(self, *registers: int) -> FunctionBuilder:
        if not registers:
            raise ValueError("retv expects at least one return register")
        return self._emit(OpCode.RETV, self._counted_regs(registers))

    def call(self, func_idx_reg: int, dst_registers: list[int], arg_registers: list[int]) -> FunctionBuilder:
        operands = self._reg(func_idx_reg) + self._counted_regs(dst_registers) + self._counted_regs(arg_registers)
        return self._emit(OpCode.CALL, operands)

    def kernel_create(
        self,
        dst: int,
        kernel_index: int,
        kernel_name_index: int,
        input_registers: list[int],
        output_registers: list[int],
    ) -> FunctionBuilder:
        # kernel_index and kernel_name_index are literal i16 table indices, not registers.
        operands = (
            self._reg(dst)
            + _i16(kernel_index)
            + _i16(kernel_name_index)
            + self._counted_regs(input_registers)
            + self._counted_regs(output_registers)
        )
        return self._emit(OpCode.KERNEL_CREATE, operands)

    def cmd_list_add_kernel(
        self,
        cmd_list: int,
        kernel: int,
        signal_events: list[int],
        wait_events: list[int],
    ) -> FunctionBuilder:
        operands = (
            self._reg(cmd_list)
            + self._reg(kernel)
            + self._counted_regs(signal_events)
            + self._counted_regs(wait_events)
        )
        return self._emit(OpCode.CMD_LIST_ADD_KERNEL, operands)

    def buffer_create(self, dst: int, elem_type: int, shape_regs: list[int], stride_regs: list[int]) -> FunctionBuilder:
        if len(shape_regs) != len(stride_regs):
            raise ValueError("buffer_create expects shape_regs and stride_regs with equal length")
        # Shape is length-prefixed; strides reuse that length and are stored uncounted.
        operands = self._reg(dst) + self._reg(elem_type) + self._counted_regs(shape_regs) + self._regs(stride_regs)
        return self._emit(OpCode.BUFFER_CREATE, operands)

    def buffer_subview(
        self,
        dst: int,
        src: int,
        offsets: list[int],
        sizes: list[int],
        strides: list[int],
    ) -> FunctionBuilder:
        if not (len(offsets) == len(sizes) == len(strides)):
            raise ValueError("buffer_subview expects offsets, sizes and strides with equal length")
        # Offsets are length-prefixed; sizes and strides reuse that length, stored uncounted.
        operands = (
            self._reg(dst)
            + self._reg(src)
            + self._counted_regs(offsets)
            + self._regs(sizes)
            + self._regs(strides)
        )
        return self._emit(OpCode.BUFFER_SUBVIEW, operands)

    def buffer_view(
        self,
        dst: int,
        src: int,
        byte_offset: int,
        elem_type: int,
        shape_regs: list[int],
        stride_regs: list[int],
    ) -> FunctionBuilder:
        if len(shape_regs) != len(stride_regs):
            raise ValueError("buffer_view expects shape_regs and stride_regs with equal length")
        operands = (
            self._reg(dst)
            + self._reg(src)
            + self._reg(byte_offset)
            + self._reg(elem_type)
            + self._counted_regs(shape_regs)
            + self._regs(stride_regs)
        )
        return self._emit(OpCode.BUFFER_VIEW, operands)

    def buffer_store(self, buffer: int, value: int, index_regs: list[int]) -> FunctionBuilder:
        operands = self._reg(buffer) + self._reg(value) + self._counted_regs(index_regs)
        return self._emit(OpCode.BUFFER_STORE, operands)

    def buffer_load(self, dst: int, buffer: int, index_regs: list[int]) -> FunctionBuilder:
        operands = self._reg(dst) + self._reg(buffer) + self._counted_regs(index_regs)
        return self._emit(OpCode.BUFFER_LOAD, operands)

    def buffer_from_const(
        self,
        dst: int,
        constant_index: int,
        elem_type: int,
        shape_regs: list[int],
        stride_regs: list[int],
    ) -> FunctionBuilder:
        if len(shape_regs) != len(stride_regs):
            raise ValueError("buffer_from_const expects shape_regs and stride_regs with equal length")
        operands = (
            self._reg(dst)
            + _i16(constant_index)
            + _i16(elem_type)
            + self._counted_regs(shape_regs)
            + self._regs(stride_regs)
        )
        return self._emit(OpCode.BUFFER_FROM_CONST, operands)

    def buffer_copy(self, dst: int, src: int) -> FunctionBuilder:
        return self._emit(OpCode.BUFFER_COPY, self._reg(dst) + self._reg(src))

    def _emit(self, opcode: OpCode, operands: bytes = b"") -> FunctionBuilder:
        self._fn.instructions.append(_Instruction(opcode=int(opcode), operands=operands))
        return self

    def _reg(self, register: int) -> bytes:
        """Encode one register index as an i16 operand, tracking the highest index used."""
        return _i16(self._resolve_operand(register))

    def _regs(self, registers: Iterable[int]) -> bytes:
        """Encode an uncounted sequence of register operands."""
        return b"".join(self._reg(register) for register in registers)

    def _counted_regs(self, registers: Iterable[int]) -> bytes:
        """Encode a register list prefixed by its i16 length."""
        registers = list(registers)
        return _i16(len(registers)) + self._regs(registers)

    def _resolve_operand(self, operand: int) -> int:
        if not isinstance(operand, int):
            raise ValueError(f"register operand must be an integer index, got {type(operand).__name__}")
        if operand < 0:
            raise ValueError(f"register index must be >= 0, got {operand}")
        if operand > self._fn.max_register_index:
            self._fn.max_register_index = operand
        return operand


_FIXED_ARITY: dict[OpCode, int] = {
    OpCode.RET: 0,
    OpCode.CMD_LIST_CREATE: 1,
    OpCode.CMD_LIST_CLOSE: 1,
    OpCode.ABS_I64: 2,
    OpCode.NOT_64: 2,
    OpCode.ABS_F64: 2,
    OpCode.NEG_F64: 2,
    OpCode.CEIL_F64: 2,
    OpCode.FLOOR_F64: 2,
    OpCode.SET: 2,
    OpCode.CONVERT_I8_TO_F32: 2,
    OpCode.CONVERT_I8_TO_F64: 2,
    OpCode.CONVERT_I16_TO_I8: 2,
    OpCode.CONVERT_I16_TO_F32: 2,
    OpCode.CONVERT_I16_TO_F64: 2,
    OpCode.CONVERT_I32_TO_I8: 2,
    OpCode.CONVERT_I32_TO_I16: 2,
    OpCode.CONVERT_I32_TO_F32: 2,
    OpCode.CONVERT_I32_TO_F64: 2,
    OpCode.CONVERT_I64_TO_I8: 2,
    OpCode.CONVERT_I64_TO_I16: 2,
    OpCode.CONVERT_I64_TO_I32: 2,
    OpCode.CONVERT_I64_TO_F32: 2,
    OpCode.CONVERT_I64_TO_F64: 2,
    OpCode.CONVERT_F32_TO_I8: 2,
    OpCode.CONVERT_F32_TO_I16: 2,
    OpCode.CONVERT_F32_TO_I32: 2,
    OpCode.CONVERT_F32_TO_I64: 2,
    OpCode.CONVERT_F32_TO_F64: 2,
    OpCode.CONVERT_F64_TO_I8: 2,
    OpCode.CONVERT_F64_TO_I16: 2,
    OpCode.CONVERT_F64_TO_I32: 2,
    OpCode.CONVERT_F64_TO_I64: 2,
    OpCode.CONVERT_F64_TO_F32: 2,
    OpCode.ADD_I64: 3,
    OpCode.DIV_I64: 3,
    OpCode.MAX_I64: 3,
    OpCode.MIN_I64: 3,
    OpCode.MUL_I64: 3,
    OpCode.REM_I64: 3,
    OpCode.SUB_I64: 3,
    OpCode.ADD_U64: 3,
    OpCode.DIV_U64: 3,
    OpCode.MAX_U64: 3,
    OpCode.MIN_U64: 3,
    OpCode.MUL_U64: 3,
    OpCode.REM_U64: 3,
    OpCode.SUB_U64: 3,
    OpCode.ADD_F64: 3,
    OpCode.DIV_F64: 3,
    OpCode.MAX_F64: 3,
    OpCode.MIN_F64: 3,
    OpCode.MUL_F64: 3,
    OpCode.REM_F64: 3,
    OpCode.SUB_F64: 3,
    OpCode.AND_64: 3,
    OpCode.OR_64: 3,
    OpCode.XOR_64: 3,
    OpCode.SLL_64: 3,
    OpCode.SRL_64: 3,
    OpCode.SRA_64: 3,
    OpCode.BUFFER_GET_DIM: 3,
    OpCode.CMD_LIST_EXEC: 2,
    OpCode.CMP_I64: 4,
    OpCode.CMP_F64: 4,
    OpCode.ROUND_F64: 3,
    OpCode.SELECT: 4,
    OpCode.ASSERT: 2,
}

# Generate a FunctionBuilder method for every fixed-arity opcode (all operands are
# register indices). Opcodes with variable arity or non-register operands have explicit
# methods above and are deliberately omitted from _FIXED_ARITY.


def _install_fixed_helpers() -> None:
    for opcode, operand_count in _FIXED_ARITY.items():
        method_name = opcode.name.lower()

        def _helper(self: FunctionBuilder, *operands: int, _opcode: OpCode = opcode,
                    _operand_count: int = operand_count) -> FunctionBuilder:
            if len(operands) != _operand_count:
                raise ValueError(f"{_opcode.name.lower()} expects {_operand_count} operands, got {len(operands)}")
            return self.instruction(_opcode, *operands)

        _helper.__name__ = method_name
        _helper.__qualname__ = f"FunctionBuilder.{method_name}"
        _helper.__doc__ = f"Append '{method_name.replace('_', '.')}' instruction."
        setattr(FunctionBuilder, method_name, _helper)


_install_fixed_helpers()
