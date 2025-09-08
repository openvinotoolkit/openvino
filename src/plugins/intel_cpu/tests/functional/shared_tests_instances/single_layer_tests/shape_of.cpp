// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/shape_of.hpp"
#include "common_test_utils/test_constants.hpp"

using ov::test::ShapeOfLayerTest;

namespace {
    const std::vector<ov::element::Type> model_types = {
            ov::element::f32,
            ov::element::i32
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Check, ShapeOfLayerTest,
                            ::testing::Combine(
                                    ::testing::ValuesIn(model_types),
                                    ::testing::Values(ov::element::i64),
                                    ::testing::Values(ov::test::static_shapes_to_test_representation({{10, 10, 10}})),
                                    ::testing::Values(ov::test::utils::DEVICE_CPU)),
                            ShapeOfLayerTest::getTestCaseName);
}  // namespace
