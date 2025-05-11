// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/topk.hpp"

using ov::test::TopKLayerTest;

namespace {

const std::vector<ov::element::Type> model_types = {
        ov::element::f32,
        ov::element::f16
};

const std::vector<int64_t> axes = {
        0,
        1,
        2,
        3
};

const std::vector<int64_t> k = {
        1,
        5,
        7,
        18,
        21
};

const std::vector<ov::op::v1::TopK::Mode> modes = {
        ov::op::v1::TopK::Mode::MIN,
        ov::op::v1::TopK::Mode::MAX
};

const std::vector<ov::op::v1::TopK::SortType> sort_types = {
        ov::op::v1::TopK::SortType::SORT_INDICES,
        ov::op::v1::TopK::SortType::SORT_VALUES,
};

const std::vector<std::vector<ov::Shape>> input_shape_static = {
        {{21, 21, 21, 21}}
};

INSTANTIATE_TEST_SUITE_P(smoke_TopK, TopKLayerTest,
        ::testing::Combine(
                ::testing::ValuesIn(k),
                ::testing::ValuesIn(axes),
                ::testing::ValuesIn(modes),
                ::testing::ValuesIn(sort_types),
                ::testing::ValuesIn(model_types),
                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shape_static)),
                ::testing::Values(ov::test::utils::DEVICE_CPU)),
        TopKLayerTest::getTestCaseName);
}  // namespace
