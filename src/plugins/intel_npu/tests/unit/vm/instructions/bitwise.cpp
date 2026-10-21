//
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "npu_bytecode_utils/instructions.hpp"
#include "utils/common.hpp"

#include <llvm/Support/FormatVariadic.h>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>
#include <common_test_utils/test_common.hpp>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

using namespace intel_npu::vm;
using namespace utils;

INSTANTIATE_TEST_SUITE_P(And64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::AND_64, 0b1100, 0b1010, int64_t{0b1000}},
                                         ThreeOperandInstructionParams{OpCode::AND_64, 0b1111, 0b0000, int64_t{0}},
                                         ThreeOperandInstructionParams{OpCode::AND_64, 0b1010, 0b0101, int64_t{0}},
                                         ThreeOperandInstructionParams{OpCode::AND_64, 0b1111'1111, 0b1010'1010,
                                                                       int64_t{0b1010'1010}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Not64, VirtualMachineTwoOperandInstructionTest,
                         testing::Values(TwoOperandInstructionParams{OpCode::NOT_64, 0b1100, int64_t{~0b1100}},
                                         TwoOperandInstructionParams{OpCode::NOT_64, 0b1010, int64_t{~0b1010}},
                                         TwoOperandInstructionParams{OpCode::NOT_64, 0b1111'0000,
                                                                     int64_t{~0b1111'0000}}),
                         VirtualMachineTwoOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Or64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::OR_64, 0b1100, 0b1010, int64_t{0b1110}},
                                         ThreeOperandInstructionParams{OpCode::OR_64, 0b1111, 0b0000, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::OR_64, 0b1010, 0b0101, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::OR_64, 0b1111'1111, 0b1010'1010,
                                                                       int64_t{0b1111'1111}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Xor64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::XOR_64, 0b1100, 0b1010, int64_t{0b0110}},
                                         ThreeOperandInstructionParams{OpCode::XOR_64, 0b1111, 0b0000, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::XOR_64, 0b1010, 0b0101, int64_t{0b1111}},
                                         ThreeOperandInstructionParams{OpCode::XOR_64, 0b1111'1111, 0b1010'1010,
                                                                       int64_t{0b0101'0101}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Sll64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::SLL_64, 1, 1, int64_t{2}},
                                         ThreeOperandInstructionParams{OpCode::SLL_64, 1, 63, INT64_MIN},
                                         ThreeOperandInstructionParams{OpCode::SLL_64, 1, 64,
                                                                       int64_t{1}},  // shift count masked by 63
                                         ThreeOperandInstructionParams{OpCode::SLL_64, -1, 1, int64_t{-2}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Srl64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::SRL_64, 2, 1, int64_t{1}},
                                         ThreeOperandInstructionParams{OpCode::SRL_64, INT64_MIN, 63, int64_t{1}},
                                         ThreeOperandInstructionParams{OpCode::SRL_64, INT64_MIN, 64,
                                                                       INT64_MIN},  // shift count masked by 63
                                         ThreeOperandInstructionParams{OpCode::SRL_64, -2, 1,
                                                                       static_cast<int64_t>(UINT64_MAX >> 1)}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(Sra64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(ThreeOperandInstructionParams{OpCode::SRA_64, 2, 1, int64_t{1}},
                                         ThreeOperandInstructionParams{OpCode::SRA_64, INT64_MIN, 63, int64_t{-1}},
                                         ThreeOperandInstructionParams{OpCode::SRA_64, INT64_MIN, 64,
                                                                       INT64_MIN},  // shift count masked by 63
                                         ThreeOperandInstructionParams{OpCode::SRA_64, -2, 1, int64_t{-1}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);
