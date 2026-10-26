//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "npu_bytecode_utils/instructions.hpp"
#include "npu_bytecode_utils/type_section.hpp"
#include "vm_runtime/virtual_machine.h"

#include <llvm/Support/FormatVariadic.h>

#include <gtest/gtest.h>

#include <climits>
#include <cstdint>
#include <limits>
#include <ostream>
#include <string>
#include <variant>
#include <vector>

namespace utils {

constexpr int64_t MIN_I8 = std::numeric_limits<int8_t>::min();
constexpr int64_t MAX_I8 = std::numeric_limits<int8_t>::max();
constexpr int64_t MIN_I16 = std::numeric_limits<int16_t>::min();
constexpr int64_t MAX_I16 = std::numeric_limits<int16_t>::max();
constexpr int64_t MIN_I32 = std::numeric_limits<int32_t>::min();
constexpr int64_t MAX_I32 = std::numeric_limits<int32_t>::max();
constexpr int64_t MIN_I64 = std::numeric_limits<int64_t>::min();
constexpr int64_t MAX_I64 = std::numeric_limits<int64_t>::max();

constexpr float INF_F32 = std::numeric_limits<float>::infinity();
constexpr float NAN_F32 = std::numeric_limits<float>::quiet_NaN();
constexpr double INF_F64 = std::numeric_limits<double>::infinity();
constexpr double NAN_F64 = std::numeric_limits<double>::quiet_NaN();

int64_t f64Bits(double v);

intel_npu::vm::Type getInt64Type();
intel_npu::vm::Type getF32Type();
intel_npu::vm::Type getF64Type();

// Compares a floating-point result against an expected value, tolerating NaN payload differences
// and checking the sign of infinities
void expectF32Eq(float actual, float expected);
void expectF64Eq(double actual, double expected);

// Base fixture for VM instruction execution tests
// Each test case receives a freshly default-constructed VirtualMachine
class VirtualMachineInstructionTest : public testing::Test {
public:
    npu_vm_module* module{nullptr};
    npu_vm_engine* engine{nullptr};
    npu_vm_function_info* funcInfo{nullptr};

    // Parses the bytecode, resolves the named function, and creates and loads an engine
    // Returns NPU_VM_SUCCESS on full success, otherwise the first failing status
    npu_vm_result loadEntrypoint(const std::vector<uint8_t>& bytecode, const char* funcName);

    // Marshals the arguments and results through the VM call API and returns the raw status
    npu_vm_result callWithResults(const char* funcName, const std::vector<npu_vm_value>& args,
                                  std::vector<npu_vm_value>& results);

protected:
    void TearDown() override {
        if (funcInfo != nullptr) {
            npu_vm_destroy_function_info(funcInfo);
        }
        if (engine != nullptr) {
            npu_vm_destroy_engine(engine);
        }
        if (module != nullptr) {
            npu_vm_destroy_module(module);
        }
    }
};

struct TwoOperandInstructionParams {
    intel_npu::vm::OpCode opcode;
    int64_t src;
    std::variant<int64_t, double> expected;

    friend std::ostream& operator<<(std::ostream& os, const TwoOperandInstructionParams& params);
};

struct ThreeOperandInstructionParams {
    intel_npu::vm::OpCode opcode;
    int64_t lhs;
    int64_t rhs;
    std::variant<int64_t, double> expected;

    friend std::ostream& operator<<(std::ostream& os, const ThreeOperandInstructionParams& params);
};

class VirtualMachineThreeOperandInstructionTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<ThreeOperandInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ThreeOperandInstructionParams>& obj);
};

class VirtualMachineTwoOperandInstructionTest :
        public VirtualMachineInstructionTest,
        public testing::WithParamInterface<TwoOperandInstructionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<TwoOperandInstructionParams>& obj);
};

}  // namespace utils
