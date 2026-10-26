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
#include <initializer_list>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

using namespace intel_npu::vm;
using namespace utils;

namespace {

struct SelectInstructionParams {
    int64_t condition;
    int64_t trueValue;
    int64_t falseValue;
    int64_t expected;

    SelectInstructionParams(int64_t condition, int64_t trueValue, int64_t falseValue, int64_t expected)
            : condition(condition), trueValue(trueValue), falseValue(falseValue), expected(expected) {
    }

    [[maybe_unused]] friend std::ostream& operator<<(std::ostream& os, const SelectInstructionParams& params) {
        os << llvm::formatv("{ condition={0}, trueValue={1}, falseValue={2}, expected={3} }", params.condition,
                            params.trueValue, params.falseValue, params.expected)
                        .str();
        return os;
    }
};

class VirtualMachineSelectInstructionTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<SelectInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<SelectInstructionParams>& obj) {
        std::ostringstream result;
        result << "condition=" << obj.param.condition << "_";
        result << "trueValue=" << obj.param.trueValue << "_";
        result << "falseValue=" << obj.param.falseValue << "_";
        result << "expected=" << obj.param.expected;
        return result.str();
    }
};

}  // namespace

TEST_P(VirtualMachineSelectInstructionTest, Execute) {
    const auto& params = GetParam();

    const auto funcName = "test";
    auto bytecode =
            BytecodeBuilder{}
                    .addFunction(BytecodeBuilder::FunctionBuilder(funcName, /*numGeneralRegisters=*/4,
                                                                  {getInt64Type(), getInt64Type(), getInt64Type()},
                                                                  {getInt64Type()},
                                                                  /*isEntrypoint=*/true)
                                         .instruction(OpCode::SELECT, {/*dst=*/0, /*cond=*/1, /*true=*/2, /*false=*/3})
                                         .retv(/*resultRegs=*/{0}))
                    .build();
    ASSERT_EQ(loadEntrypoint(bytecode, funcName), NPU_VM_SUCCESS);

    std::vector<npu_vm_value> args = {intel_npu::vm::makeI64(params.condition),
                                      intel_npu::vm::makeI64(params.trueValue),
                                      intel_npu::vm::makeI64(params.falseValue)};
    std::vector<npu_vm_value> results = {intel_npu::vm::makeI64(0)};
    ASSERT_EQ(callWithResults(funcName, args, results), NPU_VM_SUCCESS);
    EXPECT_EQ(results.at(0).i64, params.expected);
}

INSTANTIATE_TEST_SUITE_P(
        Select, VirtualMachineSelectInstructionTest,
        testing::Values(SelectInstructionParams{/*condition=*/0, /*trueValue=*/42, /*falseValue=*/84, /*expected=*/84},
                        SelectInstructionParams{/*condition=*/1, /*trueValue=*/42, /*falseValue=*/84, /*expected=*/42},
                        SelectInstructionParams{/*condition=*/-1, /*trueValue=*/42, /*falseValue=*/84,
                                                /*expected=*/42}),
        VirtualMachineSelectInstructionTest::getTestCaseName);
