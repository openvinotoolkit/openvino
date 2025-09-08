// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/range.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::RangeLayerTest;

namespace {

const std::vector<float> start = { 1.0f, 1.2f };
const std::vector<float> stop = { 5.0f, 5.2f };
const std::vector<float> step = { 1.0f, 0.1f };

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

INSTANTIATE_TEST_SUITE_P(smoke_Basic, RangeLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(start),
                                ::testing::ValuesIn(stop),
                                ::testing::ValuesIn(step),
                                ::testing::ValuesIn(model_types),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        RangeLayerTest::getTestCaseName);
}  // namespace
