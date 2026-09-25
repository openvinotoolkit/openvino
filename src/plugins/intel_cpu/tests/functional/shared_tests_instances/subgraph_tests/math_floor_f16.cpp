// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/math_floor_f16.hpp"

#include "common_test_utils/test_constants.hpp"

namespace {

using ov::test::MathFloorF16Test;

INSTANTIATE_TEST_SUITE_P(smoke_MathFloorF16,
                         MathFloorF16Test,
                         ::testing::Combine(::testing::ValuesIn(MathFloorF16Test::all_cases()),
                                            ::testing::Values(ov::test::utils::DEVICE_CPU)),
                         MathFloorF16Test::getTestCaseName);

}  // namespace
