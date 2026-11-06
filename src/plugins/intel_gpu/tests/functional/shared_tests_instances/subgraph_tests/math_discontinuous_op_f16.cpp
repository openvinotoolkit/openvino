// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "subgraph_tests/math_discontinuous_op_f16.hpp"

#include "common_test_utils/test_constants.hpp"

namespace ov::test {

INSTANTIATE_TEST_SUITE_P(smoke_MathDiscontinuousOpF16,
                         MathDiscontinuousOpF16Test,
                         ::testing::Combine(::testing::ValuesIn(MathDiscontinuousOpF16Test::all_cases()),
                                            ::testing::Values(ov::element::f16, ov::element::dynamic),
                                            ::testing::Values(utils::DEVICE_GPU)),
                         MathDiscontinuousOpF16Test::getTestCaseName);

}  // namespace ov::test
