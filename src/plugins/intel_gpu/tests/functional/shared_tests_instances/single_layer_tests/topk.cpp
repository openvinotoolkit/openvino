// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/topk.hpp"

namespace {
using ov::test::TopKLayerTest;

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<int64_t> axes = {
    0,
    1,
    2,
};

const std::vector<int64_t> k = {
    1,
    5,
    10,
};

const std::vector<ov::op::TopKMode> modes = {
    ov::op::TopKMode::MIN,
    ov::op::TopKMode::MAX,
};

const std::vector<ov::op::TopKSortType> sortTypes = {
    ov::op::TopKSortType::SORT_INDICES,
    ov::op::TopKSortType::SORT_VALUES,
};

INSTANTIATE_TEST_SUITE_P(smoke_TopK,
                         TopKLayerTest,
                         ::testing::Combine(::testing::ValuesIn(k),
                                            ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(std::vector<ov::Shape>({{10, 10, 10}}))),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         TopKLayerTest::getTestCaseName);
}  // namespace
