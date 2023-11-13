// Copyright (C) 2020-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/grn.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::GrnLayerTest;

std::vector<ov::element::Type> model_types = {
        ov::element::bf16,
        ov::element::f16,
        ov::element::f32,
};

std::vector<std::vector<ov::Shape>> input_shapes_static = {
    {{16, 24}},
    {{3, 16, 24}},
    {{1, 3, 30, 30}},
    {{2, 16, 15, 20}}};

std::vector<float> bias = {1e-6f, 0.33f, 1.1f, 2.25f, 100.25f};

const auto basicCases = ::testing::Combine(
    ::testing::ValuesIn(model_types),
    ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
    ::testing::ValuesIn(bias),
    ::testing::Values(ov::test::utils::DEVICE_CPU));

INSTANTIATE_TEST_SUITE_P(smoke_GRN_Basic, GrnLayerTest,
                        basicCases,
                        GrnLayerTest::getTestCaseName);
}  // namespace
