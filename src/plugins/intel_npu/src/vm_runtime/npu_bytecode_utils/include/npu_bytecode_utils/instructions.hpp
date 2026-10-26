//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <optional>
#include <type_traits>

namespace intel_npu::vm {

// Instruction opcodes recognized by the bytecode interpreter
enum class OpCode : uint16_t {
    // v1.0.0
    ABS_I64 = 0x01,
    ADD_I64 = 0x02,
    DIV_I64 = 0x03,
    MAX_I64 = 0x04,
    MIN_I64 = 0x05,
    MUL_I64 = 0x06,
    REM_I64 = 0x07,
    SUB_I64 = 0x08,
    ADD_U64 = 0x09,
    DIV_U64 = 0x0A,
    MAX_U64 = 0x0B,
    MIN_U64 = 0x0C,
    MUL_U64 = 0x0D,
    REM_U64 = 0x0E,
    SUB_U64 = 0x0F,
    ABS_F64 = 0x10,
    ADD_F64 = 0x11,
    CEIL_F64 = 0x12,
    DIV_F64 = 0x13,
    FLOOR_F64 = 0x14,
    MAX_F64 = 0x15,
    MIN_F64 = 0x16,
    MUL_F64 = 0x17,
    NEG_F64 = 0x18,
    REM_F64 = 0x19,
    ROUND_F64 = 0x1A,
    SUB_F64 = 0x1B,
    AND_64 = 0x1C,
    NOT_64 = 0x1D,
    OR_64 = 0x1E,
    XOR_64 = 0x1F,
    SLL_64 = 0x20,
    SRL_64 = 0x21,
    SRA_64 = 0x22,
    CMP_I64 = 0x23,
    CMP_F64 = 0x24,
    CONVERT_I8_TO_F32 = 0x25,
    CONVERT_I8_TO_F64 = 0x26,
    CONVERT_I16_TO_I8 = 0x27,
    CONVERT_I16_TO_F32 = 0x28,
    CONVERT_I16_TO_F64 = 0x29,
    CONVERT_I32_TO_I8 = 0x2A,
    CONVERT_I32_TO_I16 = 0x2B,
    CONVERT_I32_TO_F32 = 0x2C,
    CONVERT_I32_TO_F64 = 0x2D,
    CONVERT_I64_TO_I8 = 0x2E,
    CONVERT_I64_TO_I16 = 0x2F,
    CONVERT_I64_TO_I32 = 0x30,
    CONVERT_I64_TO_F32 = 0x31,
    CONVERT_I64_TO_F64 = 0x32,
    CONVERT_F32_TO_I8 = 0x33,
    CONVERT_F32_TO_I16 = 0x34,
    CONVERT_F32_TO_I32 = 0x35,
    CONVERT_F32_TO_I64 = 0x36,
    CONVERT_F32_TO_F64 = 0x37,
    CONVERT_F64_TO_I8 = 0x38,
    CONVERT_F64_TO_I16 = 0x39,
    CONVERT_F64_TO_I32 = 0x3A,
    CONVERT_F64_TO_I64 = 0x3B,
    CONVERT_F64_TO_F32 = 0x3C,
    SELECT = 0x3D,
    CALL = 0x3E,
    RET = 0x3F,
    RETV = 0x40,
    JMP = 0x41,
    JE = 0x42,
    JNE = 0x43,
    ASSERT = 0x44,
    KERNEL_CREATE = 0x45,
    CMD_LIST_CREATE = 0x46,
    CMD_LIST_ADD_KERNEL = 0x47,
    CMD_LIST_CLOSE = 0x48,
    CMD_LIST_EXEC = 0x49,
    BUFFER_CREATE = 0x4A,
    BUFFER_GET_DIM = 0x4B,
    BUFFER_SUBVIEW = 0x4C,
    BUFFER_VIEW = 0x4D,
    BUFFER_STORE = 0x4E,
    SET = 0x4F,
    SET_IMM = 0x50,

    // v1.1.0
    BUFFER_LOAD = 0x51,

    // v1.2.0
    BUFFER_FROM_CONST = 0x52,
    BUFFER_COPY = 0x53,
};

enum class RoundingMode : int16_t {
    RNE = 0x0,
    RNA = 0x1,
    RDN = 0x2,
    RUP = 0x3,
    RTZ = 0x4,
};

enum class CmpPredicate : uint8_t {
    EQ = 0x00,
    NE = 0x01,
    GT = 0x02,
    GTE = 0x03,
    LT = 0x04,
    LTE = 0x05,
};

inline constexpr size_t OPCODE_SIZE = sizeof(OpCode);
inline constexpr size_t OPERAND_SIZE = sizeof(int16_t);

inline constexpr uint16_t CMP_SIGN_BIT = 0x100;

inline constexpr uint16_t makeCmpFlag(CmpPredicate pred, bool isSigned) {
    return (isSigned ? CMP_SIGN_BIT : 0u) | static_cast<uint16_t>(pred);
}

// Get the byte size of an instruction that has a static number of operands
std::optional<size_t> getStaticInstructionSize(OpCode opcode);

// Get the byte size of an instruction. This supports both instructions that have a static or variadic number of
// operands. For variadic opcodes, 'availableBytes' is validated internally against the decode header before any operand
// is read. Callers still have to verify that 'availableBytes' is at least equal to the returned size before advancing
// past the instruction.
std::optional<size_t> getInstructionSize(OpCode opcode, const uint8_t* instructionBegin, size_t availableBytes);

// Get the opcode of the instruction whose binary representation starts at 'instructionBegin'
inline OpCode getOpcode(const uint8_t* instructionBegin) {
    std::underlying_type_t<OpCode> value = 0;
    std::memcpy(&value, instructionBegin, sizeof(value));
    return static_cast<OpCode>(value);
}

// The value of the i-th operand for the instruction whose binary representation starts at 'begin'.
// This util is intended to be used only for instructions that have operands represented as 16-bit signed integers
inline int16_t getOperand(const uint8_t* instructionBegin, size_t operandIndex) {
    int16_t value = 0;
    const auto byteOffset = static_cast<std::ptrdiff_t>(OPCODE_SIZE + operandIndex * OPERAND_SIZE);
    std::memcpy(&value, std::next(instructionBegin, byteOffset), sizeof(value));
    return value;
}

inline int64_t get64BitImm(const uint8_t* instructionBegin, size_t operandIndex) {
    int64_t value = 0;
    const auto byteOffset = static_cast<std::ptrdiff_t>(OPCODE_SIZE + operandIndex * OPERAND_SIZE);
    std::memcpy(&value, std::next(instructionBegin, byteOffset), sizeof(value));
    return value;
}

template <typename T>
inline T getAsOperand(const uint8_t* instructionBegin, size_t byteOffset) {
    static_assert(std::is_trivially_copyable_v<T>, "getAsOperand requires a trivially copyable type");
    T value = {};
    std::memcpy(&value, std::next(instructionBegin, static_cast<std::ptrdiff_t>(byteOffset)), sizeof(value));
    return value;
}

}  // namespace intel_npu::vm
