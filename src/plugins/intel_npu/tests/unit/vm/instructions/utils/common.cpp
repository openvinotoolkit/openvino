//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common.hpp"
#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vm_runtime/virtual_machine.h"
#include "vm_value_helpers.hpp"

#include <gtest/gtest.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <ostream>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

namespace {

void expectResultEq(const npu_vm_value& result, const std::variant<int64_t, double>& expected) {
    if (std::holds_alternative<double>(expected)) {
        utils::expectF64Eq(result.f64, std::get<double>(expected));
    } else {
        EXPECT_EQ(result.i64, std::get<int64_t>(expected));
    }
}

std::string expectedToString(const std::variant<int64_t, double>& expected) {
    std::ostringstream result;
    std::visit(
            [&result](auto value) {
                result << value;
            },
            expected);
    return result.str();
}

}  // namespace

namespace utils {

std::ostream& operator<<(std::ostream& os, const TwoOperandInstructionParams& params) {
    os << "{ opcode=" << static_cast<uint16_t>(params.opcode) << ", src=" << params.src
       << ", expected=" << expectedToString(params.expected) << " }";
    return os;
}

std::ostream& operator<<(std::ostream& os, const ThreeOperandInstructionParams& params) {
    os << "{ opcode=" << static_cast<uint16_t>(params.opcode) << ", lhs=" << params.lhs << ", rhs=" << params.rhs
       << ", expected=" << expectedToString(params.expected) << " }";
    return os;
}

}  // namespace utils

int64_t utils::f64Bits(double v) {
    int64_t bits{};
    std::memcpy(&bits, &v, sizeof(v));
    return bits;
}

ov::intel_npu::vm::Type utils::getInt64Type() {
    return ov::intel_npu::vm::Type{ov::intel_npu::vm::IntegerType{sizeof(int64_t) * CHAR_BIT, true}};
}

ov::intel_npu::vm::Type utils::getF32Type() {
    return ov::intel_npu::vm::Type{
            ov::intel_npu::vm::FloatType{sizeof(float) * CHAR_BIT, ov::intel_npu::vm::FloatTypeFormat::IEEE754}};
}

ov::intel_npu::vm::Type utils::getF64Type() {
    return ov::intel_npu::vm::Type{
            ov::intel_npu::vm::FloatType{sizeof(double) * CHAR_BIT, ov::intel_npu::vm::FloatTypeFormat::IEEE754}};
}

void utils::expectF64Eq(double actual, double expected) {
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(actual));
        EXPECT_EQ(std::signbit(actual), std::signbit(expected));
    } else {
        EXPECT_DOUBLE_EQ(actual, expected);
    }
}

void utils::expectF32Eq(float actual, float expected) {
    if (std::isnan(expected)) {
        EXPECT_TRUE(std::isnan(actual));
    } else if (std::isinf(expected)) {
        EXPECT_TRUE(std::isinf(actual));
        EXPECT_EQ(std::signbit(actual), std::signbit(expected));
    } else {
        EXPECT_FLOAT_EQ(actual, expected);
    }
}

npu_vm_result utils::VirtualMachineInstructionTest::loadEntrypoint(const std::vector<uint8_t>& bytecode,
                                                                   const char* funcName) {
    if (const auto status = npu_vm_parse_module(bytecode.data(), static_cast<uint32_t>(bytecode.size()), &module); status != NPU_VM_SUCCESS) {
        return status;
    }
    if (const auto status = npu_vm_get_function_info(module, funcName, &funcInfo); status != NPU_VM_SUCCESS) {
        return status;
    }
    if (const auto status = npu_vm_new_engine(&engine); status != NPU_VM_SUCCESS) {
        return status;
    }
    return npu_vm_load_module(engine, /*moduleParams*/ nullptr, module);
}

npu_vm_result utils::VirtualMachineInstructionTest::callWithResults(const char* funcName,
                                                                    const std::vector<npu_vm_value>& args,
                                                                    std::vector<npu_vm_value>& results) {
    return npu_vm_call_with_results(engine, funcName, static_cast<uint32_t>(args.size()), args.data(),
                                    static_cast<uint32_t>(results.size()), results.data());
}

std::string utils::VirtualMachineThreeOperandInstructionTest::getTestCaseName(
        const testing::TestParamInfo<ThreeOperandInstructionParams>& obj) {
    std::ostringstream result;
    result << "opcode=" << static_cast<uint16_t>(obj.param.opcode) << "_";
    result << "lhs=" << obj.param.lhs << "_";
    result << "rhs=" << obj.param.rhs << "_";
    result << "expected=" << expectedToString(obj.param.expected);
    return result.str();
}

std::string utils::VirtualMachineTwoOperandInstructionTest::getTestCaseName(
        const testing::TestParamInfo<TwoOperandInstructionParams>& obj) {
    std::ostringstream result;
    result << "opcode=" << static_cast<uint16_t>(obj.param.opcode) << "_";
    result << "src=" << obj.param.src << "_";
    result << "expected=" << expectedToString(obj.param.expected);
    return result.str();
}

namespace utils {

TEST_P(VirtualMachineTwoOperandInstructionTest, Execute) {
    const auto& params = GetParam();
    const bool isFp = std::holds_alternative<double>(params.expected);

    const auto paramType = getInt64Type();
    const auto resultType = isFp ? getF64Type() : getInt64Type();

    const auto funcName = "test";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2,
                                                                          {paramType}, {resultType},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(params.opcode, {/*dst=*/0, /*src=*/1})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {ov::intel_npu::vm::makeI64(params.src)};
    std::vector<npu_vm_value> results = {isFp ? ov::intel_npu::vm::makeF64(0.0) : ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, args, results), NPU_VM_SUCCESS);

    expectResultEq(results.at(0), params.expected);
}

// Program layout (3 registers):
//   opcode   %0, %1, %2
//   retv     1, %0
TEST_P(VirtualMachineThreeOperandInstructionTest, Execute) {
    const auto& params = GetParam();
    const bool isFp = std::holds_alternative<double>(params.expected);

    const auto paramType = getInt64Type();
    const auto resultType = isFp ? getF64Type() : getInt64Type();

    const auto funcName = "test";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                          {paramType, paramType}, {resultType},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {ov::intel_npu::vm::makeI64(params.lhs), ov::intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {isFp ? ov::intel_npu::vm::makeF64(0.0) : ov::intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, args, results), NPU_VM_SUCCESS);

    expectResultEq(results.at(0), params.expected);
}

}  // namespace utils
