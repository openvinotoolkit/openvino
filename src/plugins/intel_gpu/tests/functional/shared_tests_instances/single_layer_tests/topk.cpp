// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/topk.hpp"

namespace {
using ov::test::TopK11LayerTest;
using ov::test::TopKLayerTest;

std::vector<ov::Shape> shapes = {{10, 10, 10}};

const std::vector<ov::element::Type> netPrecisions = {
    ov::element::f32,
    ov::element::f16,
};

const std::vector<int64_t> axes = {
    0,
    1,
    2,
};

const std::vector<int64_t> keep = {
    1,
    5,
    10,
};

const std::vector<ov::op::v1::TopK::Mode> modes = {
    ov::op::v1::TopK::Mode::MIN,
    ov::op::v1::TopK::Mode::MAX,
};

const std::vector<ov::op::v1::TopK::SortType> sortTypes = {
    ov::op::v1::TopK::SortType::SORT_INDICES,
    ov::op::v1::TopK::SortType::SORT_VALUES,
};

const std::vector<bool> stable = {
    false,
    true,
};

INSTANTIATE_TEST_SUITE_P(smoke_TopK11,
                         TopK11LayerTest,
                         ::testing::Combine(::testing::ValuesIn(keep),
                                            ::testing::ValuesIn(axes),
                                            ::testing::ValuesIn(modes),
                                            ::testing::ValuesIn(sortTypes),
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(ov::test::static_shapes_to_test_representation(shapes)),
                                            ::testing::ValuesIn(stable),
                                            ::testing::Values(ov::test::utils::DEVICE_GPU)),
                         TopK11LayerTest::getTestCaseName);

}  // namespace
