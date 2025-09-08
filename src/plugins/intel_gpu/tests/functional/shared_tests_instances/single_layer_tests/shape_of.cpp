// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/shape_of.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ShapeOfLayerTest;

const std::vector<ov::element::Type> model_precisions = {
        ov::element::f32,
        ov::element::i32
};

const std::vector<std::vector<ov::Shape>> input_shapes = {
    {{1, 2, 3, 4, 5}},
    {{1, 2, 3, 4}},
    {{1, 2}}
};

INSTANTIATE_TEST_SUITE_P(smoke_Check, ShapeOfLayerTest,
                        ::testing::Combine(
                                ::testing::ValuesIn(model_precisions),
                                ::testing::Values(ov::element::i64),
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes)),
                                ::testing::Values(ov::test::utils::DEVICE_GPU)),
                        ShapeOfLayerTest::getTestCaseName);
}  // namespace
