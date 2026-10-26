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
#include <limits>
#include <string>

using namespace intel_npu::vm;
using namespace utils;

INSTANTIATE_TEST_SUITE_P(AddI64, VirtualMachineThreeOperandInstructionTest,
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
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, -10, -20, int64_t{-30}},
                                 // Identity: adding zero
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, std::numeric_limits<int64_t>::max(), 0,
                                                               std::numeric_limits<int64_t>::max()},
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, std::numeric_limits<int64_t>::min(), 0,
                                                               std::numeric_limits<int64_t>::min()},
                                 // Signed overflow wraps to INT64_MIN (two's complement)
                                 ThreeOperandInstructionParams{OpCode::ADD_I64, std::numeric_limits<int64_t>::max(), 1,
                                                               std::numeric_limits<int64_t>::min()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(SubI64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 1, 1, int64_t{0}},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 32, 10, int64_t{22}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, -5, 5, int64_t{-10}},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, 100, -1, int64_t{101}},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, -10, -20, int64_t{10}},
                                 // Identity: subtracting zero
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, std::numeric_limits<int64_t>::max(), 0,
                                                               std::numeric_limits<int64_t>::max()},
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, std::numeric_limits<int64_t>::min(), 0,
                                                               std::numeric_limits<int64_t>::min()},
                                 // Signed overflow wraps to INT64_MAX (two's complement)
                                 ThreeOperandInstructionParams{OpCode::SUB_I64, std::numeric_limits<int64_t>::min(), 1,
                                                               std::numeric_limits<int64_t>::max()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MulI64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MUL_I64, 0, 0, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, 12345, 0, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, -67890, 0, int64_t{0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MUL_I64, 2, 3, int64_t{6}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, 10, 32, int64_t{320}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::MUL_I64, -5, 5, int64_t{-25}},
                ThreeOperandInstructionParams{OpCode::MUL_I64, 100, -1, int64_t{-100}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::MUL_I64, -10, -20, int64_t{200}},
                // Identity: multiplying by one
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::max(), 1,
                                              std::numeric_limits<int64_t>::max()},
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::min(), 1,
                                              std::numeric_limits<int64_t>::min()},
                // Signed overflow wraps (two's complement)
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::max(), 2,
                                              static_cast<int64_t>(-2)},  // INT64_MAX * 2 -> -2 in two's complement
                ThreeOperandInstructionParams{OpCode::MUL_I64, std::numeric_limits<int64_t>::min(), 2,
                                              static_cast<int64_t>(0)}  // INT64_MIN * 2 -> 0 in two's complement
                ),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DivI64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::DIV_I64, 6, 3, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::DIV_I64, 10, 32, int64_t{0}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::DIV_I64, -25, 5, int64_t{-5}},
                ThreeOperandInstructionParams{OpCode::DIV_I64, 100, -1, int64_t{-100}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::DIV_I64, -200, -10, int64_t{20}},
                // Identity: dividing by one
                ThreeOperandInstructionParams{OpCode::DIV_I64, std::numeric_limits<int64_t>::max(), 1,
                                              std::numeric_limits<int64_t>::max()},
                ThreeOperandInstructionParams{OpCode::DIV_I64, std::numeric_limits<int64_t>::min(), 1,
                                              std::numeric_limits<int64_t>::min()},
                // Division overflow: INT64_MIN / -1 -> INT64_MIN (defined behavior to avoid undefined behavior)
                ThreeOperandInstructionParams{OpCode::DIV_I64, std::numeric_limits<int64_t>::min(), -1,
                                              std::numeric_limits<int64_t>::min()}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        RemI64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::REM_I64, 5, 3, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::REM_I64, 10, 32, int64_t{10}},
                // Mixed signs
                ThreeOperandInstructionParams{OpCode::REM_I64, -25, 5, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::REM_I64, 100, -1, int64_t{0}},
                // Both negative
                ThreeOperandInstructionParams{OpCode::REM_I64, -200, -10, int64_t{0}},
                // Identity: remainder by one is always zero
                ThreeOperandInstructionParams{OpCode::REM_I64, std::numeric_limits<int64_t>::max(), 1, int64_t{0}},
                ThreeOperandInstructionParams{OpCode::REM_I64, std::numeric_limits<int64_t>::min(), 1, int64_t{0}},
                // Division overflow: INT64_MIN / -1 -> INT64_MIN -> remainder is 0 (defined behavior to avoid undefined
                // behavior)
                ThreeOperandInstructionParams{OpCode::REM_I64, std::numeric_limits<int64_t>::min(), -1, int64_t{0}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        DivU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::DIV_U64, 6, 3, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::DIV_U64, 10, 32, int64_t{0}},
                // Large unsigned values (high bit set, interpreted as large positive when unsigned)
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 2,
                                              static_cast<int64_t>(0x7FFFFFFFFFFFFFFFULL)},
                // UINT64_MAX / 1 = UINT64_MAX
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // UINT64_MAX / UINT64_MAX = 1
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), int64_t{1}},
                // Value with high bit set divided by large divisor
                ThreeOperandInstructionParams{OpCode::DIV_U64, static_cast<int64_t>(0x8000000000000000ULL), 2,
                                              static_cast<int64_t>(0x4000000000000000ULL)},
                // Identity: dividing by one
                ThreeOperandInstructionParams{OpCode::DIV_U64, 100, 1, int64_t{100}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MinU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MIN_U64, 0, 0, int64_t{0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MIN_U64, 1, 2, int64_t{1}},
                ThreeOperandInstructionParams{OpCode::MIN_U64, 100, 50, int64_t{50}},
                // Large unsigned values (high bit set, interpreted as large positive when unsigned)
                ThreeOperandInstructionParams{OpCode::MIN_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              int64_t{1}},
                // Both large unsigned values
                ThreeOperandInstructionParams{OpCode::MIN_U64, static_cast<int64_t>(0x8000000000000000ULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0x8000000000000000ULL)},
                // UINT64_MAX vs UINT64_MAX
                ThreeOperandInstructionParams{OpCode::MIN_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Signed negative is large unsigned
                ThreeOperandInstructionParams{OpCode::MIN_U64, -1, 5, int64_t{5}},
                // Identity
                ThreeOperandInstructionParams{OpCode::MIN_U64, 42, 42, int64_t{42}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(AddU64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::ADD_U64, 3, 4, int64_t{7}},
                                 ThreeOperandInstructionParams{OpCode::ADD_U64, 0, 0, int64_t{0}},
                                 // Large unsigned values
                                 ThreeOperandInstructionParams{
                                         OpCode::ADD_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1, int64_t{0}},
                                 // Wrapping overflow
                                 ThreeOperandInstructionParams{OpCode::ADD_U64,
                                                               static_cast<int64_t>(0x8000000000000000ULL), 1,
                                                               static_cast<int64_t>(0x8000000000000001ULL)},
                                 // Identity: adding zero
                                 ThreeOperandInstructionParams{OpCode::ADD_U64, 100, 0, int64_t{100}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        MaxU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Zero
                ThreeOperandInstructionParams{OpCode::MAX_U64, 0, 0, int64_t{0}},
                // Basic positive
                ThreeOperandInstructionParams{OpCode::MAX_U64, 1, 2, int64_t{2}},
                ThreeOperandInstructionParams{OpCode::MAX_U64, 100, 50, int64_t{100}},
                // Large unsigned values (high bit set, interpreted as large positive when unsigned)
                ThreeOperandInstructionParams{OpCode::MAX_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Both large unsigned values
                ThreeOperandInstructionParams{OpCode::MAX_U64, static_cast<int64_t>(0x8000000000000000ULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL),
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Signed negative is large unsigned
                ThreeOperandInstructionParams{OpCode::MAX_U64, -1, 5, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Identity
                ThreeOperandInstructionParams{OpCode::MAX_U64, 42, 42, int64_t{42}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MulU64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::MUL_U64, 3, 4, int64_t{12}},
                                 ThreeOperandInstructionParams{OpCode::MUL_U64, 0, 100, int64_t{0}},
                                 // Identity: multiply by one
                                 ThreeOperandInstructionParams{OpCode::MUL_U64, 100, 1, int64_t{100}},
                                 // Large unsigned values with wrapping
                                 ThreeOperandInstructionParams{
                                         OpCode::MUL_U64, static_cast<int64_t>(0x8000000000000000ULL), 2, int64_t{0}},
                                 // UINT64_MAX * 1 = UINT64_MAX
                                 ThreeOperandInstructionParams{OpCode::MUL_U64,
                                                               static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                                               static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(RemU64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::REM_U64, 7, 3, int64_t{1}},
                                 ThreeOperandInstructionParams{OpCode::REM_U64, 10, 5, int64_t{0}},
                                 // Large unsigned values
                                 ThreeOperandInstructionParams{
                                         OpCode::REM_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 2, int64_t{1}},
                                 // Modulo by 1 = 0
                                 ThreeOperandInstructionParams{OpCode::REM_U64, 100, 1, int64_t{0}},
                                 // Value with high bit set
                                 ThreeOperandInstructionParams{
                                         OpCode::REM_U64, static_cast<int64_t>(0x8000000000000000ULL), 3, int64_t{2}}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        SubU64, VirtualMachineThreeOperandInstructionTest,
        testing::Values(
                // Basic positive
                ThreeOperandInstructionParams{OpCode::SUB_U64, 10, 3, int64_t{7}},
                ThreeOperandInstructionParams{OpCode::SUB_U64, 0, 0, int64_t{0}},
                // Wrapping underflow (0 - 1 = UINT64_MAX)
                ThreeOperandInstructionParams{OpCode::SUB_U64, 0, 1, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL)},
                // Large unsigned values
                ThreeOperandInstructionParams{OpCode::SUB_U64, static_cast<int64_t>(0xFFFFFFFFFFFFFFFFULL), 1,
                                              static_cast<int64_t>(0xFFFFFFFFFFFFFFFEULL)},
                // Identity: subtracting zero
                ThreeOperandInstructionParams{OpCode::SUB_U64, 100, 0, int64_t{100}}),
        VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MinI64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, 1, 2, int64_t{1}},
                                 // Basic negative
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, -10, -20, int64_t{-20}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, -5, 5, int64_t{-5}},
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, 100, -1, int64_t{-1}},
                                 // Boundary
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, std::numeric_limits<int64_t>::min(), 1,
                                                               std::numeric_limits<int64_t>::min()},
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, std::numeric_limits<int64_t>::max(), 1,
                                                               int64_t{1}},
                                 ThreeOperandInstructionParams{OpCode::MIN_I64, std::numeric_limits<int64_t>::max(),
                                                               std::numeric_limits<int64_t>::min(),
                                                               std::numeric_limits<int64_t>::min()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MaxI64, VirtualMachineThreeOperandInstructionTest,
                         testing::Values(
                                 // Zero
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, 0, 0, int64_t{0}},
                                 // Basic positive
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, 1, 2, int64_t{2}},
                                 // Basic negative
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, -10, -20, int64_t{-10}},
                                 // Mixed signs
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, -5, 5, int64_t{5}},
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, 100, -1, int64_t{100}},
                                 // Boundary
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, std::numeric_limits<int64_t>::min(), 1,
                                                               int64_t{1}},
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, std::numeric_limits<int64_t>::max(), 1,
                                                               std::numeric_limits<int64_t>::max()},
                                 ThreeOperandInstructionParams{OpCode::MAX_I64, std::numeric_limits<int64_t>::max(),
                                                               std::numeric_limits<int64_t>::min(),
                                                               std::numeric_limits<int64_t>::max()}),
                         VirtualMachineThreeOperandInstructionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
        AbsI64, VirtualMachineTwoOperandInstructionTest,
        testing::Values(TwoOperandInstructionParams{OpCode::ABS_I64, 0, int64_t{0}},
                        TwoOperandInstructionParams{OpCode::ABS_I64, 12345, int64_t{12345}},
                        TwoOperandInstructionParams{OpCode::ABS_I64, -67890, int64_t{67890}},
                        // Identity: abs of INT64_MIN is INT64_MIN (defined behavior to avoid undefined behavior)
                        TwoOperandInstructionParams{OpCode::ABS_I64, std::numeric_limits<int64_t>::min(),
                                                    std::numeric_limits<int64_t>::min()}),
        VirtualMachineTwoOperandInstructionTest::getTestCaseName);
