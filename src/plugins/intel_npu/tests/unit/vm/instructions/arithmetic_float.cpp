// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/instructions.hpp"
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

using namespace ov;
using namespace ov::intel_npu::vm;
using namespace utils;

namespace {

struct RoundF64Params {
    int64_t src;
    RoundingMode flag;
    double expected;

    [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os, const RoundF64Params& params) {
        os << "{ src=" << params.src << ", flag=" << static_cast<int16_t>(params.flag)
           << ", expected=" << params.expected << " }";
        return os;
    }
};

class VirtualMachineRoundF64Test :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<RoundF64Params> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<RoundF64Params>& obj) {
        std::ostringstream result;
        result << "flag=" << static_cast<int16_t>(obj.param.flag) << "_src=" << obj.param.src
               << "_expected=" << obj.param.expected;
        return result.str();
    }
};

}  // namespace

TEST_P(VirtualMachineRoundF64Test, Execute) {
    const auto& params = GetParam();

    const auto funcName = "test";
    auto bytecode = BytecodeBuilder{}
                            .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2,
                                                                          {getF64Type()}, {getF64Type()},
                                                                          /*isEntrypoint=*/true)
                                                 .instruction(OpCode::ROUND_F64,
                                                              {/*dst=*/0, /*src=*/1, static_cast<int16_t>(params.flag)})
                                                 .retv(/*resultRegs=*/{0}))
                            .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    double srcDouble{};
    std::memcpy(&srcDouble, &params.src, sizeof(srcDouble));
    std::vector<npu_vm_value> args = {ov::intel_npu::vm::makeF64(srcDouble)};
    std::vector<npu_vm_value> results = {ov::intel_npu::vm::makeF64(0.0)};
    ASSERT_EQ(callWithResults(funcName, args, results), NPU_VM_SUCCESS);

    expectF64Eq(results.at(0).f64, params.expected);
}

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

INSTANTIATE_TEST_SUITE_P(
        AddF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-0.0), f64Bits(-0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-0.0), f64Bits(0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.0), f64Bits(0.0), double{1.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.0), f64Bits(2.0), double{3.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.5), f64Bits(2.5), double{4.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-1.0), f64Bits(2.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-1.0), f64Bits(1.0), double{0.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-3.0), f64Bits(-4.0), double{-7.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(INF_F64), f64Bits(1.0), INF_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-INF_F64), f64Bits(1.0), -INF_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(INF_F64), f64Bits(INF_F64), INF_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), -INF_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(INF_F64), f64Bits(-INF_F64), NAN_F64},
                // NaN
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(NAN_F64), f64Bits(1.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(1.0), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(-INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(NAN_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::ADD_F64, f64Bits(NAN_F64), f64Bits(-INF_F64), NAN_F64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        SubF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-0.0), f64Bits(-0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-0.0), f64Bits(0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(0.0), double{1.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(5.0), f64Bits(3.0), double{2.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(1.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(3.0), double{-2.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-1.0), f64Bits(2.0), double{-3.0}},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(-2.0), double{3.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-2.0), f64Bits(-5.0), double{3.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(INF_F64), f64Bits(1.0), INF_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-INF_F64), f64Bits(1.0), -INF_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(INF_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(INF_F64), f64Bits(-INF_F64), INF_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-INF_F64), f64Bits(INF_F64), -INF_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), NAN_F64},
                // NaN
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(NAN_F64), f64Bits(1.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(1.0), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(-INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(NAN_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::SUB_F64, f64Bits(NAN_F64), f64Bits(-INF_F64), NAN_F64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MulF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(0.0), f64Bits(-0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-0.0), f64Bits(0.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-0.0), f64Bits(-0.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(0.0), f64Bits(5.0), double{0.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(1.0), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(2.0), f64Bits(3.0), double{6.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-2.0), f64Bits(3.0), double{-6.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-2.0), f64Bits(-3.0), double{6.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(INF_F64), f64Bits(1.0), INF_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-INF_F64), f64Bits(1.0), -INF_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(INF_F64), f64Bits(INF_F64), INF_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(INF_F64), f64Bits(-INF_F64), -INF_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-INF_F64), f64Bits(INF_F64), -INF_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), INF_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(INF_F64), f64Bits(0.0), NAN_F64},
                // NaN
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(NAN_F64), f64Bits(1.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(1.0), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(-INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(NAN_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::MUL_F64, f64Bits(NAN_F64), f64Bits(-INF_F64), NAN_F64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DivF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(0.0), f64Bits(1.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(0.0), f64Bits(-1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-0.0), f64Bits(1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-0.0), f64Bits(-1.0), double{0.0}},
                // Division by zero (not a trap in floating point)
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(0.0), f64Bits(0.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(1.0), f64Bits(0.0), INF_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-1.0), f64Bits(0.0), -INF_F64},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(6.0), f64Bits(2.0), double{3.0}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(1.0), f64Bits(4.0), double{0.25}},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(7.0), f64Bits(7.0), double{1.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-6.0), f64Bits(2.0), double{-3.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-6.0), f64Bits(-2.0), double{3.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(INF_F64), f64Bits(2.0), INF_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(INF_F64), f64Bits(-2.0), -INF_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(INF_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(INF_F64), f64Bits(-INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-INF_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(INF_F64), f64Bits(0.0), INF_F64},
                // NaN
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(NAN_F64), f64Bits(1.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(1.0), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(-INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(NAN_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::DIV_F64, f64Bits(NAN_F64), f64Bits(-INF_F64), NAN_F64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        RemF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(0.0), f64Bits(1.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(0.0), f64Bits(-1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-0.0), f64Bits(1.0), double{-0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-0.0), f64Bits(-1.0), double{0.0}},
                // Division by zero (not a trap in floating point)
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(0.0), f64Bits(0.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(1.0), f64Bits(0.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-1.0), f64Bits(0.0), NAN_F64},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(6.0), f64Bits(2.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(1.0), f64Bits(4.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(7.0), f64Bits(7.0), double{0.0}},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(2.5), f64Bits(1.0), double{0.5}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-7.0), f64Bits(3.0), double{-1.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-7.0), f64Bits(-3.0), double{-1.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(INF_F64), f64Bits(2.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(INF_F64), f64Bits(-2.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(INF_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(INF_F64), f64Bits(-INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-INF_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(INF_F64), f64Bits(0.0), NAN_F64},
                // NaN
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(NAN_F64), f64Bits(1.0), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(1.0), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(-INF_F64), f64Bits(NAN_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(NAN_F64), f64Bits(INF_F64), NAN_F64},
                ThreeOperandInstructionParams{OpCode::REM_F64, f64Bits(NAN_F64), f64Bits(-INF_F64), NAN_F64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MaxF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(3.0), f64Bits(5.0), double{5.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(5.0), f64Bits(3.0), double{5.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(-1.0), f64Bits(1.0), double{1.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(-5.0), f64Bits(-3.0), double{-3.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(INF_F64), f64Bits(1.0), INF_F64},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(-INF_F64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(1.0), f64Bits(INF_F64), INF_F64},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(1.0), f64Bits(-INF_F64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(INF_F64), f64Bits(INF_F64), INF_F64},
                // NaN
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(NAN_F64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(1.0), f64Bits(NAN_F64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MAX_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), NAN_F64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MinF64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(0.0), f64Bits(0.0), double{0.0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(3.0), f64Bits(5.0), double{3.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(5.0), f64Bits(3.0), double{3.0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-1.0), f64Bits(1.0), double{-1.0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-5.0), f64Bits(-3.0), double{-5.0}},
                // Infinity
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(INF_F64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-INF_F64), f64Bits(1.0), -INF_F64},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(1.0), f64Bits(INF_F64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(1.0), f64Bits(-INF_F64), -INF_F64},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(-INF_F64), f64Bits(-INF_F64), -INF_F64},
                // NaN
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(NAN_F64), f64Bits(1.0), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(1.0), f64Bits(NAN_F64), double{1.0}},
                ThreeOperandInstructionParams{OpCode::MIN_F64, f64Bits(NAN_F64), f64Bits(NAN_F64), NAN_F64}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(AbsF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(0.0), double{0.0}},
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(3.0), double{3.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-3.0), double{3.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(INF_F64), double{INF_F64}},
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-INF_F64), double{INF_F64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(NAN_F64), double{NAN_F64}},
                                 TwoOperandInstructionParams{OpCode::ABS_F64, f64Bits(-NAN_F64), double{NAN_F64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(NegF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(0.0), double{-0.0}},
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(3.0), double{-3.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-3.0), double{3.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(INF_F64), double{-INF_F64}},
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-INF_F64), double{INF_F64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(NAN_F64), double{NAN_F64}},
                                 TwoOperandInstructionParams{OpCode::NEG_F64, f64Bits(-NAN_F64), double{NAN_F64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(CeilF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(1.1), double{2.0}},
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(2.0), double{2.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(-1.1), double{-1.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(INF_F64), double{INF_F64}},
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(-INF_F64), double{-INF_F64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(NAN_F64), double{NAN_F64}},
                                 TwoOperandInstructionParams{OpCode::CEIL_F64, f64Bits(-NAN_F64), double{NAN_F64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(FloorF64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(0.0), double{0.0}},
                                 // Basic positive
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(1.9), double{1.0}},
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(2.0), double{2.0}},
                                 // Basic negative
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(-1.1), double{-2.0}},
                                 // Infinity
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(INF_F64), double{INF_F64}},
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(-INF_F64), double{-INF_F64}},
                                 // NaN
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(NAN_F64), double{NAN_F64}},
                                 TwoOperandInstructionParams{OpCode::FLOOR_F64, f64Bits(-NAN_F64), double{NAN_F64}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RNE, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.5), RoundingMode::RNE, 2.0},
                                         RoundF64Params{f64Bits(2.5), RoundingMode::RNE, 2.0},
                                         RoundF64Params{f64Bits(-1.5), RoundingMode::RNE, -2.0},
                                         RoundF64Params{f64Bits(-2.5), RoundingMode::RNE, -2.0},
                                         RoundF64Params{f64Bits(1.4), RoundingMode::RNE, 1.0},
                                         RoundF64Params{f64Bits(1.0), RoundingMode::RNE, 1.0},
                                         RoundF64Params{f64Bits(INF_F64), RoundingMode::RNE, INF_F64},
                                         RoundF64Params{f64Bits(NAN_F64), RoundingMode::RNE, NAN_F64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RNA, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.5), RoundingMode::RNA, 2.0},
                                         RoundF64Params{f64Bits(2.5), RoundingMode::RNA, 3.0},
                                         RoundF64Params{f64Bits(-1.5), RoundingMode::RNA, -2.0},
                                         RoundF64Params{f64Bits(-2.5), RoundingMode::RNA, -3.0},
                                         RoundF64Params{f64Bits(1.4), RoundingMode::RNA, 1.0},
                                         RoundF64Params{f64Bits(1.0), RoundingMode::RNA, 1.0},
                                         RoundF64Params{f64Bits(INF_F64), RoundingMode::RNA, INF_F64},
                                         RoundF64Params{f64Bits(NAN_F64), RoundingMode::RNA, NAN_F64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RDN, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.5), RoundingMode::RDN, 1.0},
                                         RoundF64Params{f64Bits(-1.1), RoundingMode::RDN, -2.0},
                                         RoundF64Params{f64Bits(2.0), RoundingMode::RDN, 2.0},
                                         RoundF64Params{f64Bits(INF_F64), RoundingMode::RDN, INF_F64},
                                         RoundF64Params{f64Bits(NAN_F64), RoundingMode::RDN, NAN_F64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RUP, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.1), RoundingMode::RUP, 2.0},
                                         RoundF64Params{f64Bits(-1.5), RoundingMode::RUP, -1.0},
                                         RoundF64Params{f64Bits(2.0), RoundingMode::RUP, 2.0},
                                         RoundF64Params{f64Bits(INF_F64), RoundingMode::RUP, INF_F64},
                                         RoundF64Params{f64Bits(NAN_F64), RoundingMode::RUP, NAN_F64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RoundF64_RTZ, VirtualMachineRoundF64Test,
                         testing::Values(RoundF64Params{f64Bits(1.9), RoundingMode::RTZ, 1.0},
                                         RoundF64Params{f64Bits(-1.9), RoundingMode::RTZ, -1.0},
                                         RoundF64Params{f64Bits(2.0), RoundingMode::RTZ, 2.0},
                                         RoundF64Params{f64Bits(INF_F64), RoundingMode::RTZ, INF_F64},
                                         RoundF64Params{f64Bits(NAN_F64), RoundingMode::RTZ, NAN_F64}),
                         VirtualMachineRoundF64Test::getTestCaseName);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
