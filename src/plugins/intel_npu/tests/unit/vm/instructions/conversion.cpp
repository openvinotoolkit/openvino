//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "utils/common.hpp"
#include "vm_runtime/virtual_machine.h"
#include "vm_value_helpers.hpp"

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

using namespace ov::intel_npu::vm;
using namespace utils;

namespace {

std::string typeToStr(const ov::intel_npu::vm::Type& type) {
    const auto typeCode = ov::intel_npu::vm::getTypeCode(type);
    const auto bitWidth = ov::intel_npu::vm::getBitWidth(type);
    switch (typeCode) {
    case ov::intel_npu::vm::TypeCode::INTEGER: {
        if (bitWidth == sizeof(int8_t) * CHAR_BIT) {
            return "INT8";
        } else if (bitWidth == sizeof(int16_t) * CHAR_BIT) {
            return "INT16";
        } else if (bitWidth == sizeof(int32_t) * CHAR_BIT) {
            return "INT32";
        } else if (bitWidth == sizeof(int64_t) * CHAR_BIT) {
            return "INT64";
        } else {
            return "INT";
        }
    }
    case ov::intel_npu::vm::TypeCode::FLOAT: {
        if (bitWidth == sizeof(float) * CHAR_BIT) {
            return "FP32";
        } else if (bitWidth == sizeof(double) * CHAR_BIT) {
            return "FP64";
        } else {
            return "FP";
        }
    }
    case ov::intel_npu::vm::TypeCode::OPAQUE:
        return "OPAQUE" + std::to_string(bitWidth) + "b";
    default:
        return "UNKNOWN";
    }
}

std::string valueToStr(const npu_vm_value& value, const ov::intel_npu::vm::Type& type) {
    std::ostringstream ss;
    const auto typeCode = ov::intel_npu::vm::getTypeCode(type);
    if (typeCode == ov::intel_npu::vm::TypeCode::INTEGER) {
        ss << value.i64;
    } else if (typeCode == ov::intel_npu::vm::TypeCode::FLOAT) {
        const auto bitWidth = ov::intel_npu::vm::getBitWidth(type);
        if (bitWidth == sizeof(float) * CHAR_BIT) {
            ss << value.f32;
        } else {
            ss << value.f64;
        }
    } else {
        ss << "<unsupported>";
    }
    return ss.str();
}

struct ConvertInstructionParams {
    OpCode opcode;
    ov::intel_npu::vm::Type srcType;
    ov::intel_npu::vm::Type dstType;
    npu_vm_value src;
    npu_vm_value expected;

    [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os, const ConvertInstructionParams& params) {
        os << "{ opcode=" << static_cast<uint16_t>(params.opcode) << ", srcType=" << typeToStr(params.srcType)
           << ", dstType=" << typeToStr(params.dstType) << ", src=" << valueToStr(params.src, params.srcType)
           << ", expected=" << valueToStr(params.expected, params.dstType) << " }";
        return os;
    }
};

class VirtualMachineConvertInstructionTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<ConvertInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertInstructionParams>& obj) {
        std::ostringstream result;
        result << "opcode_" << static_cast<uint16_t>(obj.param.opcode) << "_srcType_" << typeToStr(obj.param.srcType)
               << "_dstType_" << typeToStr(obj.param.dstType) << "_src_" << valueToStr(obj.param.src, obj.param.srcType)
               << "_expected_" << valueToStr(obj.param.expected, obj.param.dstType) << "_idx_" << obj.index;
        return result.str();
    }
};

ConvertInstructionParams intToInt(OpCode op, int64_t src, int64_t expected) {
    return {op, getInt64Type(), getInt64Type(), ov::intel_npu::vm::makeI64(src), ov::intel_npu::vm::makeI64(expected)};
}
ConvertInstructionParams intToF32(OpCode op, int64_t src, float expected) {
    return {op, getInt64Type(), getF32Type(), ov::intel_npu::vm::makeI64(src), ov::intel_npu::vm::makeF32(expected)};
}
ConvertInstructionParams intToF64(OpCode op, int64_t src, double expected) {
    return {op, getInt64Type(), getF64Type(), ov::intel_npu::vm::makeI64(src), ov::intel_npu::vm::makeF64(expected)};
}
ConvertInstructionParams f32ToInt(OpCode op, float src, int64_t expected) {
    return {op, getF32Type(), getInt64Type(), ov::intel_npu::vm::makeF32(src), ov::intel_npu::vm::makeI64(expected)};
}
ConvertInstructionParams f32ToF64(OpCode op, float src, double expected) {
    return {op, getF32Type(), getF64Type(), ov::intel_npu::vm::makeF32(src), ov::intel_npu::vm::makeF64(expected)};
}
ConvertInstructionParams f64ToInt(OpCode op, double src, int64_t expected) {
    return {op, getF64Type(), getInt64Type(), ov::intel_npu::vm::makeF64(src), ov::intel_npu::vm::makeI64(expected)};
}
ConvertInstructionParams f64ToF32(OpCode op, double src, float expected) {
    return {op, getF64Type(), getF32Type(), ov::intel_npu::vm::makeF64(src), ov::intel_npu::vm::makeF32(expected)};
}

}  // namespace

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

// Execute a single two-operand conversion instruction and compare the result.
// The function signature uses srcType/dstType so that npu_vm_call_with_results
// marshals values correctly (i64, f32, or f64).
TEST_P(VirtualMachineConvertInstructionTest, Execute) {
    const auto& params = GetParam();

    const auto funcName = "test_convert";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2,
                                                                          {params.srcType}, {params.dstType},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(params.opcode, {/*dst=*/0, /*src=*/1})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {params.src};
    npu_vm_value resultInit{};
    std::vector<npu_vm_value> results = {resultInit};
    ASSERT_EQ(callWithResults(funcName, args, results), NPU_VM_SUCCESS);

    const auto dstTypeCode = ov::intel_npu::vm::getTypeCode(params.dstType);
    if (dstTypeCode == ov::intel_npu::vm::TypeCode::INTEGER) {
        EXPECT_EQ(results.at(0).i64, params.expected.i64);
    } else {
        const auto dstBitWidth = ov::intel_npu::vm::getBitWidth(params.dstType);
        if (dstBitWidth == sizeof(float) * CHAR_BIT) {
            expectF32Eq(results.at(0).f32, params.expected.f32);
        } else {
            expectF64Eq(results.at(0).f64, params.expected.f64);
        }
    }
}

// CONVERT_I8_TO_F32: sign-extend low 8 bits to i8, then convert to float32
INSTANTIATE_TEST_SUITE_P(ConvertI8ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I8_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, MAX_I8, static_cast<float>(MAX_I8)),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I8_TO_F32, MIN_I8, static_cast<float>(MIN_I8)),
                                         // 0xFF low byte → i8(-1) → -1.0f
                                         intToF32(OpCode::CONVERT_I8_TO_F32, static_cast<int64_t>(0xFF), -1.0f)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I8_TO_F64: sign-extend low 8 bits to i8, then convert to float64
INSTANTIATE_TEST_SUITE_P(ConvertI8ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I8_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, MAX_I8, static_cast<double>(MAX_I8)),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, MIN_I8, static_cast<double>(MIN_I8)),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, static_cast<int64_t>(0xFF), -1.0),
                                         intToF64(OpCode::CONVERT_I8_TO_F64, static_cast<int64_t>(0x1FF), -1.0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I16_TO_I8: truncate i16 to i8 (low 8 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(
        ConvertI16ToI8, VirtualMachineConvertInstructionTest,
        testing::Values(intToInt(OpCode::CONVERT_I16_TO_I8, 0, 0), intToInt(OpCode::CONVERT_I16_TO_I8, 1, 1),
                        intToInt(OpCode::CONVERT_I16_TO_I8, MAX_I8, MAX_I8),
                        // 128 as i16 → low 8 bits = 0x80 = i8(-128)
                        intToInt(OpCode::CONVERT_I16_TO_I8, 128, MIN_I8), intToInt(OpCode::CONVERT_I16_TO_I8, -1, -1),
                        // 256 → low 8 bits = 0x00 = 0
                        intToInt(OpCode::CONVERT_I16_TO_I8, 256, 0), intToInt(OpCode::CONVERT_I16_TO_I8, MAX_I16, -1)),
        VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I16_TO_F32: sign-extend low 16 bits to i16, then convert to float32
INSTANTIATE_TEST_SUITE_P(ConvertI16ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I16_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, MAX_I16, static_cast<float>(MAX_I16)),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, MIN_I16, static_cast<float>(MIN_I16)),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, static_cast<int64_t>(0xFFFF), -1.0f),
                                         intToF32(OpCode::CONVERT_I16_TO_F32, static_cast<int64_t>(0x8000), -32768.0f)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I16_TO_F64: sign-extend low 16 bits to i16, then convert to float64
INSTANTIATE_TEST_SUITE_P(ConvertI16ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I16_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, MAX_I16, static_cast<double>(MAX_I16)),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, MIN_I16, static_cast<double>(MIN_I16)),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, static_cast<int64_t>(0xFFFF), -1.0),
                                         intToF64(OpCode::CONVERT_I16_TO_F64, static_cast<int64_t>(0x18000), -32768.0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_I8: truncate i32 to i8 (low 8 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI32ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I32_TO_I8, 0, 0),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, 1, 1),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, MAX_I8, MAX_I8),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, 128, MIN_I8),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, -1, -1),
                                         intToInt(OpCode::CONVERT_I32_TO_I8, 256, 0),
                                         // INT32_MAX low 8 bits = 0xFF → i8(-1)
                                         intToInt(OpCode::CONVERT_I32_TO_I8, MAX_I32, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_I16: truncate i32 to i16 (low 16 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI32ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I32_TO_I16, 0, 0),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, 1, 1),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, MAX_I16, MAX_I16),
                                         // 32768 = 0x8000 as i16 → i16(-32768)
                                         intToInt(OpCode::CONVERT_I32_TO_I16, 32768, MIN_I16),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, -1, -1),
                                         intToInt(OpCode::CONVERT_I32_TO_I16, 65536, 0),
                                         // INT32_MAX low 16 bits = 0xFFFF → i16(-1)
                                         intToInt(OpCode::CONVERT_I32_TO_I16, MAX_I32, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_F32: sign-extend low 32 bits to i32, then convert to float32
INSTANTIATE_TEST_SUITE_P(ConvertI32ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I32_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, MAX_I32, static_cast<float>(MAX_I32)),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, MIN_I32, static_cast<float>(MIN_I32)),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, static_cast<int64_t>(0xFFFFFFFF), -1.0f),
                                         intToF32(OpCode::CONVERT_I32_TO_F32, static_cast<int64_t>(0x80000000),
                                                  static_cast<float>(MIN_I32))),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I32_TO_F64: sign-extend low 32 bits to i32, then convert to float64
INSTANTIATE_TEST_SUITE_P(ConvertI32ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I32_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, MAX_I32, static_cast<double>(MAX_I32)),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, MIN_I32, static_cast<double>(MIN_I32)),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, static_cast<int64_t>(0xFFFFFFFF), -1.0),
                                         intToF64(OpCode::CONVERT_I32_TO_F64, static_cast<int64_t>(0x1FFFFFFFF), -1.0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_I8: truncate i64 to i8 (low 8 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I64_TO_I8, 0, 0),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, 1, 1),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, MAX_I8, MAX_I8),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, 128, MIN_I8),
                                         intToInt(OpCode::CONVERT_I64_TO_I8, -1, -1),
                                         // INT64_MAX low 8 bits = 0xFF → i8(-1)
                                         intToInt(OpCode::CONVERT_I64_TO_I8, MAX_I64, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_I16: truncate i64 to i16 (low 16 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I64_TO_I16, 0, 0),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, 1, 1),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, MAX_I16, MAX_I16),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, 32768, MIN_I16),
                                         intToInt(OpCode::CONVERT_I64_TO_I16, -1, -1),
                                         // INT64_MAX low 16 bits = 0xFFFF → i16(-1)
                                         intToInt(OpCode::CONVERT_I64_TO_I16, MAX_I64, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_I32: truncate i64 to i32 (low 32 bits), sign-extend to i64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToI32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToInt(OpCode::CONVERT_I64_TO_I32, 0, 0),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, 1, 1),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, MAX_I32, MAX_I32),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, MIN_I32, MIN_I32),
                                         intToInt(OpCode::CONVERT_I64_TO_I32, -1, -1),
                                         // INT64_MAX low 32 bits = 0xFFFFFFFF → i32(-1)
                                         intToInt(OpCode::CONVERT_I64_TO_I32, MAX_I64, -1)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_F32: i64 to float32
INSTANTIATE_TEST_SUITE_P(ConvertI64ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF32(OpCode::CONVERT_I64_TO_F32, 0, 0.0f),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, 1, 1.0f),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, -1, -1.0f),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, MAX_I64, static_cast<float>(MAX_I64)),
                                         intToF32(OpCode::CONVERT_I64_TO_F32, MIN_I64, static_cast<float>(MIN_I64))),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_I64_TO_F64: i64 to float64
INSTANTIATE_TEST_SUITE_P(ConvertI64ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(intToF64(OpCode::CONVERT_I64_TO_F64, 0, 0.0),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, 1, 1.0),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, -1, -1.0),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, MAX_I64, static_cast<double>(MAX_I64)),
                                         intToF64(OpCode::CONVERT_I64_TO_F64, MIN_I64, static_cast<double>(MIN_I64))),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// Float-to-integer conversions use saturating semantics: NaN → 0, overflow → INT_MAX/MIN.

// CONVERT_F32_TO_I8
INSTANTIATE_TEST_SUITE_P(ConvertF32ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I8, 0.0f, 0),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, 1.0f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -1.0f, -1),
                                         // Truncation toward zero
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, 1.9f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -1.9f, -1),
                                         // In-range boundaries
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, static_cast<float>(MAX_I8), MAX_I8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, static_cast<float>(MIN_I8), MIN_I8),
                                         // Overflow saturation
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, 128.0f, MAX_I8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -129.0f, MIN_I8),
                                         // Special values
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, INF_F32, MAX_I8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, -INF_F32, MIN_I8),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I8, NAN_F32, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_I16
INSTANTIATE_TEST_SUITE_P(ConvertF32ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I16, 0.0f, 0),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, 1.0f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -1.0f, -1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, 1.9f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -1.9f, -1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, static_cast<float>(MAX_I16), MAX_I16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, static_cast<float>(MIN_I16), MIN_I16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, 32768.0f, MAX_I16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -32769.0f, MIN_I16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, INF_F32, MAX_I16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, -INF_F32, MIN_I16),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I16, NAN_F32, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_I32
INSTANTIATE_TEST_SUITE_P(
        ConvertF32ToI32, VirtualMachineConvertInstructionTest,
        testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I32, 0.0f, 0), f32ToInt(OpCode::CONVERT_F32_TO_I32, 1.5f, 1),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, -1.5f, -1),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, 1.0e9f, 1000000000),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, 1.0e10f, MAX_I32),   // overflow saturation
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, -1.0e10f, MIN_I32),  // overflow saturation
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, INF_F32, MAX_I32),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, -INF_F32, MIN_I32),
                        f32ToInt(OpCode::CONVERT_F32_TO_I32, NAN_F32, 0)),
        VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_I64
INSTANTIATE_TEST_SUITE_P(ConvertF32ToI64, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToInt(OpCode::CONVERT_F32_TO_I64, 0.0f, 0),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, 1.0f, 1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, -1.0f, -1),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, 1.0e18f, static_cast<int64_t>(1.0e18f)),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, INF_F32, MAX_I64),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, -INF_F32, MIN_I64),
                                         f32ToInt(OpCode::CONVERT_F32_TO_I64, NAN_F32, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F32_TO_F64: widen float32 to float64
INSTANTIATE_TEST_SUITE_P(ConvertF32ToF64, VirtualMachineConvertInstructionTest,
                         testing::Values(f32ToF64(OpCode::CONVERT_F32_TO_F64, 0.0f, 0.0),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, 1.5f, static_cast<double>(1.5f)),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, -3.14f, static_cast<double>(-3.14f)),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, INF_F32, INF_F64),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, -INF_F32, -INF_F64),
                                         f32ToF64(OpCode::CONVERT_F32_TO_F64, NAN_F32, NAN_F64)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I8
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI8, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I8, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, 1.0, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -1.0, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, 1.9, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -1.9, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, static_cast<double>(MAX_I8), MAX_I8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, static_cast<double>(MIN_I8), MIN_I8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, 128.0, MAX_I8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -129.0, MIN_I8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, INF_F64, MAX_I8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, -INF_F64, MIN_I8),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I8, NAN_F64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I16
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI16, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I16, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, 1.0, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, -1.0, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, static_cast<double>(MAX_I16), MAX_I16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, static_cast<double>(MIN_I16), MIN_I16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, 32768.0, MAX_I16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, -32769.0, MIN_I16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, INF_F64, MAX_I16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, -INF_F64, MIN_I16),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I16, NAN_F64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I32
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI32, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I32, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, 1.5, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, -1.5, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, 1.0e9, 1000000000),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, 1.0e10, MAX_I32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, -1.0e10, MIN_I32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, INF_F64, MAX_I32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, -INF_F64, MIN_I32),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I32, NAN_F64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_I64
INSTANTIATE_TEST_SUITE_P(ConvertF64ToI64, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToInt(OpCode::CONVERT_F64_TO_I64, 0.0, 0),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, 1.0, 1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, -1.0, -1),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, 1.0e18, static_cast<int64_t>(1.0e18)),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, INF_F64, MAX_I64),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, -INF_F64, MIN_I64),
                                         f64ToInt(OpCode::CONVERT_F64_TO_I64, NAN_F64, 0)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// CONVERT_F64_TO_F32: narrow float64 to float32
INSTANTIATE_TEST_SUITE_P(ConvertF64ToF32, VirtualMachineConvertInstructionTest,
                         testing::Values(f64ToF32(OpCode::CONVERT_F64_TO_F32, 0.0, 0.0f),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, 1.5, 1.5f),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, -3.14, static_cast<float>(-3.14)),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, INF_F64, INF_F32),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, -INF_F64, -INF_F32),
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, NAN_F64, NAN_F32),
                                         // Rounding: midpoint between 1.0f and nextafterf(1.0f, 2.0f)
                                         // ties-to-even rounds to 1.0f (even mantissa LSB)
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, 0x1.0000010000000p+0, 1.0f),
                                         // Midpoint between 0x1.000002p+0 and 0x1.000004p+0
                                         // ties-to-even rounds to 0x1.000004p+0 (even mantissa LSB)
                                         f64ToF32(OpCode::CONVERT_F64_TO_F32, 0x1.0000030000000p+0, 0x1.000004p+0f)),
                         VirtualMachineConvertInstructionTest::getTestCaseName);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
