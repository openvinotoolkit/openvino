// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "single_op_tests/embedding_segments_sum.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::EmbeddingSegmentsSumLayerTest;

const std::vector<ov::element::Type> model_type = {
        ov::element::f32,
        ov::element::i32,
        ov::element::u8
};

const std::vector<ov::element::Type> ind_type = {
        ov::element::i64,
        ov::element::i32
};

const std::vector<std::vector<ov::Shape>> input_shapes_static =
    {{{5, 6}},
    {{10, 35}},
    {{5, 4, 16}}};

const std::vector<std::vector<size_t>> indices = {{0, 1, 2, 2, 3}, {4, 4, 3, 1, 2}};
const std::vector<std::vector<size_t>> segment_ids = {{0, 1, 2, 3, 4}, {0, 0, 2, 2, 4}};
const std::vector<size_t> num_segments = {5, 7};
const std::vector<size_t> default_index = {0, 4};
const std::vector<bool> with_weights = {false, true};
const std::vector<bool> with_default_index = {false, true};

const auto embSegmentsSumArgSet = ::testing::Combine(
        ::testing::ValuesIn(indices),
        ::testing::ValuesIn(segment_ids),
        ::testing::ValuesIn(num_segments),
        ::testing::ValuesIn(default_index),
        ::testing::ValuesIn(with_weights),
        ::testing::ValuesIn(with_default_index)
);

INSTANTIATE_TEST_SUITE_P(smoke, EmbeddingSegmentsSumLayerTest,
                        ::testing::Combine(
                                embSegmentsSumArgSet,
                                ::testing::ValuesIn(ov::test::static_shapes_to_test_representation(input_shapes_static)),
                                ::testing::ValuesIn(model_type),
                                ::testing::ValuesIn(ind_type),
                                ::testing::Values(ov::test::utils::DEVICE_CPU)),
                        EmbeddingSegmentsSumLayerTest::getTestCaseName);
}  // namespace
