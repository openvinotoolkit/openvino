// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/concat_with_neighbors_graph_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> precisions = {
        ov::element::f32
};

const std::vector<ov::PartialShape> shapes = {
    { 1, 3, 16, 16 },
    { 4, 3, 16, 16 }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ConcatWithNeighborsGraphTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(precisions),
        ::testing::ValuesIn(shapes),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::Values("convolution_addition_original"),
        ::testing::Values("u8")),
    ConcatWithNeighborsGraphTransformation::getTestCaseName);
}  // namespace
