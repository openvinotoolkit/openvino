//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "bytecode_builder.hpp"
#include "npu_bytecode_utils/instructions.hpp"
#include "utils/common.hpp"
#include "vm_runtime/virtual_machine.h"
#include "vm_value_helpers.hpp"

#include <llvm/Support/FormatVariadic.h>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <optional>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

using namespace intel_npu::vm;
using namespace utils;

namespace {

class VirtualMachineCallInstructionTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<ThreeOperandInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ThreeOperandInstructionParams>& obj) {
        return VirtualMachineThreeOperandInstructionTest::getTestCaseName(obj);
    }
};

using VirtualMachineJmpInstructionTest = VirtualMachineInstructionTest;

struct JmpSelectTestParams {
    int64_t input;
    int64_t expected;

    [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os, const JmpSelectTestParams& params) {
        os << llvm::formatv("{ input={0}, expected={1} }", params.input, params.expected).str();
        return os;
    }
};

// Shared base for the jump/branch fixtures. Each derived fixture keeps a distinct type so that
// its own TEST_P body is instantiated independently, while reusing a single test-name formatter.
class JmpSelectTestBase :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<JmpSelectTestParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<JmpSelectTestParams>& obj) {
        std::ostringstream result;
        result << "input=" << obj.param.input << "_expected=" << obj.param.expected;
        return result.str();
    }
};

class VirtualMachineJmpTest : public JmpSelectTestBase {};

class VirtualMachineJmpBackwardTest : public JmpSelectTestBase {};

// Registers (3 total: 2 value + 1 param):
//   firstParamRegIndex = numRegisters - numParams = 3 - 1 = 2
//   0=rZero  1=rResult  2=rA (input param, last reg)
// Byte layout:
//   [ 0] set_imm rZero, 0      (12B)
//   [12] je 36, rA, rZero      (14B) if rA == 0: jump to [48]
//   [26] set_imm rResult, 2    (12B) not-taken (rA != 0)
//   [38] jmp 22                (10B) -> [60] (retv)
//   [48] set_imm rResult, 1    (12B) taken (rA == 0)
//   [60] retv rResult           (6B)
class VirtualMachineJETest : public JmpSelectTestBase {};

class VirtualMachineJNETest : public JmpSelectTestBase {};

struct JumpBoundsCheckParams {
    std::string testName;
    std::function<std::vector<uint8_t>()> buildBytecode;
};

class VirtualMachineJumpBoundsCheckTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<JumpBoundsCheckParams> {};

struct AssertInstructionParams {
    int64_t condition;
    int16_t messageIndex;
    npu_vm_result expectedStatus;
    std::optional<int64_t> expectedResult;

    [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os, const AssertInstructionParams& params) {
        os << llvm::formatv("{ condition={0}, messageIndex={1}, expectedStatus={2}, expectedResult={3} }",
                            params.condition, params.messageIndex, static_cast<uint32_t>(params.expectedStatus),
                            params.expectedResult.has_value() ? std::to_string(*params.expectedResult) : "none")
                        .str();
        return os;
    }
};

class VirtualMachineAssertInstructionTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<AssertInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<AssertInstructionParams>& obj) {
        std::ostringstream result;
        result << "condition=" << obj.param.condition << "_";
        result << "messageIndex=" << obj.param.messageIndex << "_";
        result << "status=" << static_cast<uint32_t>(obj.param.expectedStatus) << "_";
        result << "result=";
        if (obj.param.expectedResult.has_value()) {
            result << *obj.param.expectedResult;
        } else {
            result << "none";
        }
        return result.str();
    }
};

}  // namespace

// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)

// Program layout (3 registers):
// func(%0 = opcode(%1, %2)):
//   opcode   %0, %1, %2
// Entrypoint that calls func:
//   set.imm  0, callee_idx
//   call     0, 1, {dst_regs}, 2, {arg_regs} (callee index is held in register 0)
//   retv     1, %0
TEST_P(VirtualMachineCallInstructionTest, Execute) {
    const auto& params = GetParam();

    auto builder = BytecodeBuilder{}.addFunction(BytecodeBuilder::FunctionBuilder("callee", /*numGeneralRegisters=*/3,
                                                                                  {getInt64Type(), getInt64Type()},
                                                                                  {getInt64Type()})
                                                         .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                         .retv(/*resultRegs=*/{0}));
    const auto calleeIndex = builder.getFunctionIndex("callee");
    ASSERT_TRUE(calleeIndex.has_value());

    const auto entryFuncName = "entry";
    const auto functionIndexReg = 0;
    const auto bytecode =
            builder.addFunction(
                           BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/3,
                                                            {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                            /*isEntrypoint=*/true)
                                   .setImm(/*dst=*/functionIndexReg, /*imm=*/static_cast<int64_t>(calleeIndex.value()))
                                   .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{1, 2})
                                   .retv(/*resultRegs=*/{0}))
                    .build();

    ASSERT_EQ(loadEntrypoint(bytecode, entryFuncName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(entryFuncName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, std::get<int64_t>(params.expected));
}

// Program layout (4 registers):
// func:
//   opcode   %0, %1, %2
// Entrypoint that calls func:
//   set.imm  1, callee_idx
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Add])
//   set.imm  1, callee_idx
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Add] - result of the
//   previous add and second attr: add(dst, %1) to test caller state across multiple calls) retv 1, %0
TEST_P(VirtualMachineCallInstructionTest, ExecuteTwice) {
    const auto& params = GetParam();

    auto builder = BytecodeBuilder{}.addFunction(BytecodeBuilder::FunctionBuilder("callee", /*numGeneralRegisters=*/3,
                                                                                  {getInt64Type(), getInt64Type()},
                                                                                  {getInt64Type()})
                                                         .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                         .retv(/*resultRegs=*/{0}));
    const auto calleeIndex = builder.getFunctionIndex("callee");
    ASSERT_TRUE(calleeIndex.has_value());

    const auto entryFuncName = "entry_twice";
    const auto functionIndexReg = 1;
    const auto bytecode =
            builder.addFunction(
                           BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/4,
                                                            {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                            /*isEntrypoint=*/true)
                                   .setImm(/*dst=*/functionIndexReg, /*imm=*/static_cast<int64_t>(calleeIndex.value()))
                                   .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                   .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{0, 2})
                                   .retv(/*resultRegs=*/{0}))
                    .build();
    ASSERT_EQ(loadEntrypoint(bytecode, entryFuncName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(entryFuncName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, std::get<int64_t>(params.expected) + params.lhs);
}

// Program layout (4 registers):
// func_add:
//   opcode   %0, %1, %2
// func_mul:
//   opcode   %0, %1, %2
// Entrypoint that calls func:
//   set.imm  1, add_callee_idx
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Add])
//   set.imm  1, mul_callee_idx
//   call     1, 1, {dst_regs}, 2, {arg_regs} (function that contains an arith op [like Mul] - result of the
//   previous add and second attr: mul(dst, %1) to test caller state across multiple calls) retv 1, %0
TEST_P(VirtualMachineCallInstructionTest, ExecuteTwiceDifferentFunctions) {
    const auto& params = GetParam();

    auto builder = BytecodeBuilder{}
                           .addFunction(BytecodeBuilder::FunctionBuilder("callee", /*numGeneralRegisters=*/3,
                                                                         {getInt64Type(), getInt64Type()},
                                                                         {getInt64Type()}, /*isEntrypoint=*/false)
                                                .instruction(params.opcode, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                .retv(/*resultRegs=*/{0}))
                           .addFunction(BytecodeBuilder::FunctionBuilder("callee2", /*numGeneralRegisters=*/3,
                                                                         {getInt64Type(), getInt64Type()},
                                                                         {getInt64Type()}, /*isEntrypoint=*/false)
                                                .instruction(OpCode::MUL_I64, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                                                .retv(/*resultRegs=*/{0}));
    const auto firstCalleeIndex = builder.getFunctionIndex("callee");
    ASSERT_TRUE(firstCalleeIndex.has_value());
    const auto secondCalleeIndex = builder.getFunctionIndex("callee2");
    ASSERT_TRUE(secondCalleeIndex.has_value());

    const auto entryFuncName = "entry_twice_different_callees";
    const auto functionIndexReg = 1;
    const auto bytecode = builder.addFunction(BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/4,
                                                                               {getInt64Type(), getInt64Type()},
                                                                               {getInt64Type()}, /*isEntrypoint=*/true)
                                                      .setImm(/*dst=*/functionIndexReg,
                                                              /*imm=*/static_cast<int64_t>(firstCalleeIndex.value()))
                                                      .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                                      .setImm(/*dst=*/functionIndexReg,
                                                              /*imm=*/static_cast<int64_t>(secondCalleeIndex.value()))
                                                      .call(functionIndexReg, /*dstRegs=*/{0}, /*argRegs=*/{0, 2})
                                                      .retv(/*resultRegs=*/{0}))
                                  .build();
    ASSERT_EQ(loadEntrypoint(bytecode, entryFuncName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.lhs), intel_npu::vm::makeI64(params.rhs)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(entryFuncName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, std::get<int64_t>(params.expected) * params.lhs);
}

INSTANTIATE_TEST_SUITE_P(CallOpTests, VirtualMachineCallInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 1, 1, int64_t{2}},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 10, 32, int64_t{42}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, -5, 5, int64_t{0}},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, 100, -1, int64_t{99}},
                                 // Both negative
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, -10, -20, int64_t{-30}}),
                         VirtualMachineCallInstructionTest::getTestCaseName);

TEST_F(VirtualMachineCallInstructionTest, CallChainThreeLevelsDeepPropagatesResult) {
    auto builder = BytecodeBuilder{}.addFunction(
            BytecodeBuilder::FunctionBuilder("inner", /*numGeneralRegisters=*/3, {getInt64Type(), getInt64Type()},
                                             {getInt64Type()})
                    .instruction(OpCode::ADD_I64, {/*dst=*/0, /*lhs=*/1, /*rhs=*/2})
                    .retv(/*resultRegs=*/{0}));
    const auto innerIndex = builder.getFunctionIndex("inner");
    ASSERT_TRUE(innerIndex.has_value());

    // middle has 4 registers: params occupy regs [2, 3], general regs are [0, 1]. Reg 1 holds the callee index
    builder.addFunction(BytecodeBuilder::FunctionBuilder("middle", /*numGeneralRegisters=*/4,
                                                         {getInt64Type(), getInt64Type()}, {getInt64Type()})
                                .setImm(/*dst=*/1, /*imm=*/static_cast<int64_t>(innerIndex.value()))
                                .call(/*functionIndexReg=*/1, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                .retv(/*resultRegs=*/{0}));
    const auto middleIndex = builder.getFunctionIndex("middle");
    ASSERT_TRUE(middleIndex.has_value());

    const auto entryFuncName = "entry";
    const auto bytecode =
            builder.addFunction(BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/4,
                                                                 {getInt64Type(), getInt64Type()}, {getInt64Type()},
                                                                 /*isEntrypoint=*/true)
                                        .setImm(/*dst=*/1, /*imm=*/static_cast<int64_t>(middleIndex.value()))
                                        .call(/*functionIndexReg=*/1, /*dstRegs=*/{0}, /*argRegs=*/{2, 3})
                                        .retv(/*resultRegs=*/{0}))
                    .build();

    ASSERT_EQ(loadEntrypoint(bytecode, entryFuncName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(20), intel_npu::vm::makeI64(22)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(entryFuncName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, 42);
}

TEST_F(VirtualMachineCallInstructionTest, RetvRoutesMultipleResultsToCallerDestinationRegisters) {
    // swap has 2 registers, both parameters [0, 1]; retv returns them reversed
    auto builder = BytecodeBuilder{}.addFunction(BytecodeBuilder::FunctionBuilder("swap", /*numGeneralRegisters=*/2,
                                                                                  {getInt64Type(), getInt64Type()},
                                                                                  {getInt64Type(), getInt64Type()})
                                                         .retv(/*resultRegs=*/{1, 0}));
    const auto swapIndex = builder.getFunctionIndex("swap");
    ASSERT_TRUE(swapIndex.has_value());

    // entry has 5 registers: params occupy regs [3, 4], general regs are [0, 1, 2]. Reg 2 holds the callee index
    const auto entryFuncName = "entry";
    const auto bytecode =
            builder.addFunction(BytecodeBuilder::FunctionBuilder(entryFuncName, /*numGeneralRegisters=*/5,
                                                                 {getInt64Type(), getInt64Type()},
                                                                 {getInt64Type(), getInt64Type()},
                                                                 /*isEntrypoint=*/true)
                                        .setImm(/*dst=*/2, /*imm=*/static_cast<int64_t>(swapIndex.value()))
                                        .call(/*functionIndexReg=*/2, /*dstRegs=*/{0, 1},
                                              /*argRegs=*/{3, 4})
                                        .retv(/*resultRegs=*/{0, 1}))
                    .build();

    ASSERT_EQ(loadEntrypoint(bytecode, entryFuncName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(7), intel_npu::vm::makeI64(9)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0), intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(entryFuncName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, 9);
    EXPECT_EQ(results.at(1).i64, 7);
}

TEST_F(VirtualMachineCallInstructionTest, SelfRecursiveFunctionComputesTriangularNumber) {
    // The `sum` function computes sum(n) = n + sum(n - 1) with base case sum(0) = 0, pushing one frame per recursion
    // level.
    // Byte layout (offsets are PC-relative from the function body start):
    //   [ 0] set_imm rZero, 0                       (12B)
    //   [12] je 72, rN, rZero        -> base case   (14B)
    //   [26] set_imm rOne, 1                        (12B)
    //   [38] sub.i64 rRec, rN, rOne                  (8B)
    //   [46] set_imm rFunc, sumFnIndex              (12B)
    //   [58] call rFunc, {rRec}, {rRec}             (12B)
    //   [70] add.i64 rResult, rN, rRec               (8B)
    //   [78] retv {rResult}                          (6B)
    //   [84] set_imm rResult, 0  (base case)        (12B)
    //   [96] retv {rResult}                          (6B)

    constexpr int64_t sumFnIndex = 0;       // The index of the sum function within the function section
    constexpr int64_t jumpToBaseCase = 72;  // The relative offset to the base case
    constexpr int64_t numRegs = 6;  // Total number of registers used by the function (general: [0-4], params: [5])

    constexpr int16_t rResult = 0;  // The result register
    constexpr int16_t rFunc = 1;    // The register to hold the function index for the recursive call
    constexpr int16_t rZero = 2;    // The register to hold the constant zero for the base case comparison
    constexpr int16_t rOne = 3;     // The register to hold the constant one for decrementing n in the recursive case
    constexpr int16_t rRec = 4;     // The register to hold the result of the recursive call sum(n - 1)
    constexpr int16_t rN = 5;       // The register to hold the input parameter n

    const auto funcName = "sum";
    auto builder = BytecodeBuilder{};
    builder.addFunction(BytecodeBuilder::FunctionBuilder(funcName, numRegs, {getInt64Type()}, {getInt64Type()},
                                                         /*isEntrypoint=*/true)
                                .setImm(rZero, 0)
                                .je(jumpToBaseCase, rN, rZero)
                                .setImm(rOne, 1)
                                .instruction(OpCode::SUB_I64, {rRec, rN, rOne})
                                .setImm(rFunc, sumFnIndex)
                                .call(/*functionIndexReg=*/rFunc, /*dstRegs=*/{rRec}, /*argRegs=*/{rRec})
                                .instruction(OpCode::ADD_I64, {rResult, rN, rRec})
                                .retv(/*resultRegs=*/{rResult})
                                .setImm(rResult, 0)
                                .retv(/*resultRegs=*/{rResult}));
    const auto resolvedIndex = builder.getFunctionIndex(funcName);
    ASSERT_TRUE(resolvedIndex.has_value());
    ASSERT_EQ(resolvedIndex.value(), static_cast<uint64_t>(sumFnIndex));

    const auto bytecode = builder.build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    struct RecursionCase {
        int64_t n;
        int64_t expected;
    };
    for (const auto& testCase : {RecursionCase{0, 0}, RecursionCase{1, 1}, RecursionCase{5, 15}, RecursionCase{10, 55},
                                 RecursionCase{1000, 500500}}) {
        std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(testCase.n)};
        std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(-1)};
        ASSERT_EQ(callWithResults(funcName, parameters, results), NPU_VM_SUCCESS) << "n=" << testCase.n;
        EXPECT_EQ(results.at(0).i64, testCase.expected) << "n=" << testCase.n;
        ASSERT_EQ(npu_vm_reset_state(engine, /*resetExecutionContext=*/false), NPU_VM_SUCCESS);
    }

    const auto overMaxDepthTestCase = 1001;
    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(overMaxDepthTestCase)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(-1)};
    ASSERT_NE(callWithResults(funcName, parameters, results), NPU_VM_SUCCESS) << "n=" << overMaxDepthTestCase;
}

// Registers (3 total: 2 value + 1 param):
//   firstParamRegIndex = numRegisters - numParams = 3 - 1 = 2
//   0=rZero  1=rResult  2=rA (input param, last reg)
// Byte layout:
//   [ 0] set_imm rZero, 0      (12B)
//   [12] jne 36, rA, rZero     (14B) if rA != 0: jump to [48]
//   [26] set_imm rResult, 1    (12B) not-taken (rA == 0)
//   [38] jmp 22                (10B) -> [60] (retv)
//   [48] set_imm rResult, 2    (12B) taken (rA != 0)
//   [60] retv rResult           (6B)
TEST_P(VirtualMachineJmpTest, SelectJmp) {
    constexpr int16_t rZero = 0;
    constexpr int16_t rResult = 1;
    constexpr int16_t rA = 2;  // input param: firstParamRegIndex = numRegs(3) - numParams(1) = 2

    const auto funcName = "jmp_select_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .setImm(rZero, 0)
                                                       .jne(36, rA, rZero)
                                                       .setImm(rResult, 1)
                                                       .jmp(22)
                                                       .setImm(rResult, 2)
                                                       .retv({rResult}))
                                  .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(JmpSelect, VirtualMachineJmpTest,
                         testing::Values(JmpSelectTestParams{0, 1}, JmpSelectTestParams{1, 2}),
                         VirtualMachineJmpTest::getTestCaseName);

TEST_P(VirtualMachineJETest, ConditionalJump) {
    constexpr int16_t rZero = 0;
    constexpr int16_t rResult = 1;
    constexpr int16_t rA = 2;  // input param: firstParamRegIndex = numRegs(3) - numParams(1) = 2

    const auto funcName = "je_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .setImm(rZero, 0)
                                                       .je(36, rA, rZero)
                                                       .setImm(rResult, 2)
                                                       .jmp(22)
                                                       .setImm(rResult, 1)
                                                       .retv({rResult}))
                                  .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, GetParam().expected);
}

// Same layout as JE test, but jne instead of je.
// When rA != 0: jne taken → result = 1
// When rA == 0: jne not taken → result = 2
TEST_P(VirtualMachineJNETest, ConditionalJump) {
    constexpr int16_t rZero = 0;
    constexpr int16_t rResult = 1;
    constexpr int16_t rA = 2;  // input param: firstParamRegIndex = numRegs(3) - numParams(1) = 2

    const auto funcName = "jne_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/3,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .setImm(rZero, 0)
                                                       .jne(36, rA, rZero)
                                                       .setImm(rResult, 2)
                                                       .jmp(22)
                                                       .setImm(rResult, 1)
                                                       .retv({rResult}))
                                  .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(JEConditionalJump, VirtualMachineJETest,
                         testing::Values(JmpSelectTestParams{0, 1}, JmpSelectTestParams{5, 2},
                                         JmpSelectTestParams{-1, 2}),
                         VirtualMachineJETest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(JNEConditionalJump, VirtualMachineJNETest,
                         testing::Values(JmpSelectTestParams{5, 1}, JmpSelectTestParams{-1, 1},
                                         JmpSelectTestParams{0, 2}),
                         VirtualMachineJNETest::getTestCaseName);

// Registers (4 total: 3 value + 1 param):
//   firstParamRegIndex = numRegisters - numParams = 4 - 1 = 3
//   0=rCounter  1=rOne  2=rZero  3=rA (input param, last reg)
// Byte layout:
//   [ 0] set     rCounter, rA              copy param into working register  ( 6B)
//   [ 6] set_imm rOne,  1                                                    (12B)
//   [18] set_imm rZero, 0                                                    (12B)
//   [30] sub_i64 rCounter, rCounter, rOne                                    ( 8B)
//   [38] jne -8, rCounter, rZero           if != 0: _pc += -8 → [30]        (14B)
//   [52] retv rCounter                     returns 0 when loop exits         ( 6B)
TEST_P(VirtualMachineJmpBackwardTest, BackwardJump) {
    constexpr int16_t rCounter = 0;
    constexpr int16_t rOne = 1;
    constexpr int16_t rZero = 2;
    constexpr int16_t rA = 3;  // input param: firstParamRegIndex = numRegs(4) - numParams(1) = 3

    const auto funcName = "jmp_backward_test";
    const auto bytecode = BytecodeBuilder{}
                                  .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/4,
                                                                                {getInt64Type()}, {getInt64Type()},
                                                                                /*isEntrypoint=*/true)
                                                       .instruction(OpCode::SET, {rCounter, rA})
                                                       .setImm(rOne, 1)
                                                       .setImm(rZero, 0)
                                                       .instruction(OpCode::SUB_I64, {rCounter, rCounter, rOne})
                                                       .jne(-8, rCounter, rZero)
                                                       .retv({rCounter}))
                                  .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(GetParam().input)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, parameters, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(JmpBackward, VirtualMachineJmpBackwardTest,
                         testing::Values(JmpSelectTestParams{1, 0}, JmpSelectTestParams{3, 0},
                                         JmpSelectTestParams{5, 0}),
                         VirtualMachineJmpBackwardTest::getTestCaseName);

// Regression test: before the 64-bit offset fix, getRelativeOffset() used checked_cast<int16_t>
// which would abort on any jump offset > INT16_MAX (32767).
// Layout:
//   [     0] jmp 32788            (10B) — forward jump past all padding
//   [    10] set rResult, rResult × 5463 (32778B) — unreachable padding
//   [32788]  set_imm rResult, 1   (12B) — jump lands here
//   [32800]  retv {rResult}        (6B)
TEST_F(VirtualMachineJmpInstructionTest, JmpForwardOffsetExceedsInt16Range) {
    constexpr int16_t rResult = 0;
    const auto funcName = "large_jmp_test";

    constexpr int64_t padCount = 5463;  // 5463 * 6 = 32778 bytes > INT16_MAX
    constexpr int64_t setInstrSize = 6;
    constexpr int64_t jmpInstrSize = 10;
    const int64_t jmpOffset = jmpInstrSize + padCount * setInstrSize;  // 32788

    auto fb = BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/1, {}, {getInt64Type()},
                                               /*isEntrypoint=*/true);
    fb.jmp(jmpOffset);
    for (int64_t i = 0; i < padCount; ++i) {
        fb.instruction(OpCode::SET, {rResult, rResult});
    }
    fb.setImm(rResult, 1);
    fb.retv({rResult});

    const auto bytecode = BytecodeBuilder{}.addFunction(fb).build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, /*args=*/{}, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, 1);
}

TEST_P(VirtualMachineJumpBoundsCheckTest, Execute) {
    const auto& p = GetParam();
    const auto bytecode = p.buildBytecode();
    const auto funcName = "test";
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);
    EXPECT_FALSE(npu_vm_call(engine, funcName, /*num_arguments=*/0, /*arguments=*/nullptr) == NPU_VM_SUCCESS);
}

INSTANTIATE_TEST_SUITE_P(
        JumpBoundsCheck, VirtualMachineJumpBoundsCheckTest,
        testing::Values(JumpBoundsCheckParams{"JmpPositiveOffsetBeyondEnd",
                                              [] {
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/0, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .jmp(10000)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JmpNegativeOffsetBeforeStart",
                                              [] {
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/0, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .jmp(-10000)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JmpZeroOffsetInfiniteLoopGuard",
                                              [] {
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/0, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .jmp(0)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JETakenOffsetBeyondEnd",
                                              [] {
                                                  constexpr int16_t r0 = 0;
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/1, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .je(10000, r0, r0)
                                                                               .ret())
                                                          .build();
                                              }},
                        JumpBoundsCheckParams{"JNETakenOffsetBeyondEnd",
                                              [] {
                                                  constexpr int16_t rZero = 0;
                                                  constexpr int16_t rOne = 1;
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/2, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .setImm(rOne, 1)
                                                                               .jne(10000, rZero, rOne)
                                                                               .ret())
                                                          .build();
                                              }},
                        // Body layout: [0] jmp +3 (10B), [10] ret (2B). Valid instruction offsets are {0, 10};
                        // offset 3 lands inside the jmp instruction and must be rejected.
                        JumpBoundsCheckParams{"JmpForwardMidInstruction",
                                              [] {
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/0, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .jmp(3)
                                                                               .ret())
                                                          .build();
                                              }},
                        // Body layout: [0] set_imm r0, 0 (12B), [12] jmp -3 (10B), [22] ret (2B). Target = 12 - 3 = 9,
                        // which lands inside the set_imm at offset 0 and must be rejected.
                        JumpBoundsCheckParams{"JmpBackwardMidInstruction",
                                              [] {
                                                  constexpr int16_t r0 = 0;
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/1, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .setImm(r0, 0)
                                                                               .jmp(-3)
                                                                               .ret())
                                                          .build();
                                              }},
                        // Body layout: [0] je +3, r0, r0 (14B), [14] ret (2B). Predicate is always taken, target = 3
                        // lands inside the je instruction and must be rejected.
                        JumpBoundsCheckParams{"JETakenMidInstruction",
                                              [] {
                                                  constexpr int16_t r0 = 0;
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/1, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .je(3, r0, r0)
                                                                               .ret())
                                                          .build();
                                              }},
                        // Body layout: [0] set_imm r1, 1 (12B), [12] jne +3, r0, r1 (14B), [26] ret (2B). Target =
                        // 12 + 3 = 15, which lands inside the jne instruction and must be rejected.
                        JumpBoundsCheckParams{"JNETakenMidInstruction",
                                              [] {
                                                  constexpr int16_t rZero = 0;
                                                  constexpr int16_t rOne = 1;
                                                  return BytecodeBuilder{}
                                                          .addFunction(BytecodeBuilder::FunctionBuilder(
                                                                               "test", /*numGeneralRegisters=*/2, {},
                                                                               {}, /*isEntrypoint=*/true)
                                                                               .setImm(rOne, 1)
                                                                               .jne(3, rZero, rOne)
                                                                               .ret())
                                                          .build();
                                              }}),
        [](const testing::TestParamInfo<JumpBoundsCheckParams>& info) {
            return info.param.testName;
        });

TEST_P(VirtualMachineAssertInstructionTest, Execute) {
    const auto& params = GetParam();

    const auto funcName = "assert_test";
    const auto bytecode =
            BytecodeBuilder{}
                    .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/2, {getInt64Type()},
                                                                  {getInt64Type()},
                                                                  /*isEntrypoint=*/true)
                                         .instruction(OpCode::ASSERT, {/*condition=*/1, params.messageIndex})
                                         .setImm(/*dst=*/0, /*imm=*/42)
                                         .retv({0}))
                    .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> parameters = {intel_npu::vm::makeI64(params.condition)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(-1)};
    const auto status = callWithResults(funcName, parameters, results);

    EXPECT_EQ(status, params.expectedStatus);
    if (params.expectedResult.has_value()) {
        EXPECT_EQ(results.at(0).i64, *params.expectedResult);
    }
}

INSTANTIATE_TEST_SUITE_P(
        Assert, VirtualMachineAssertInstructionTest,
        testing::Values(AssertInstructionParams{/*condition=*/1, /*messageIndex=*/0, NPU_VM_SUCCESS,
                                                /*expectedResult=*/42},
                        AssertInstructionParams{/*condition=*/-1, /*messageIndex=*/0, NPU_VM_SUCCESS,
                                                /*expectedResult=*/42},
                        AssertInstructionParams{/*condition=*/0, /*messageIndex=*/0, NPU_VM_ERROR_FUNCTION_CALL_FAILED,
                                                /*expectedResult=*/std::nullopt},
                        AssertInstructionParams{/*condition=*/0, /*messageIndex=*/7, NPU_VM_ERROR_FUNCTION_CALL_FAILED,
                                                /*expectedResult=*/std::nullopt}),
        VirtualMachineAssertInstructionTest::getTestCaseName);

// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
