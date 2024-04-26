// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_ND_update.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ScatterNDUpdateLayerTest;

// map<inputShape map<indicesShape, indicesValue>>
// updateShape is gotten from inputShape and indicesShape
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> sliceSelectInShape{
    {{4, 3, 2, 3, 2}, {{{2, 2, 1}, {3, 2, 0, 1}}}},
    {{10, 9, 9, 11}, {{{4, 1}, {1, 3, 5, 7}}, {{1, 2}, {4, 6}}, {{2, 3}, {0, 1, 1, 2, 2, 2}}, {{1, 4}, {5, 5, 4, 9}}}},
    {{10, 9, 12, 10, 11}, {{{2, 2, 1}, {5, 6, 2, 8}}, {{2, 3}, {0, 4, 6, 5, 7, 1}}}},
    {{15},                        {{{2, 1}, {1, 3}}}},
    {{15, 14},                    {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}}},
    {{15, 14, 13},                {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}}},
    {{15, 14, 13, 12},            {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}, {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
    {{2, 2, 2}, {2, 3, 1, 8, 7, 5, 6, 5}}}},
    {{15, 14, 13, 12, 16},        {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}, {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
    {{2, 5}, {2, 3, 1, 8, 6, 9, 7, 5, 6, 5}}}},
    {{15, 14, 13, 12, 16, 10},    {{{2, 1}, {1, 3}}, {{2, 2}, {2, 3, 10, 11}}, {{2, 3}, {2, 3, 1, 8, 10, 11}}, {{2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}},
    {{1, 2, 4}, {2, 3, 1, 8, 7, 5, 6, 5}}, {{2, 5}, {2, 3, 1, 8, 6,  9, 7, 5, 6, 5}}, {{2, 6}, {2, 3, 1, 8, 6, 5,  9, 7, 5, 6, 5, 7}}}}
};

std::vector<ov::test::scatterNDUpdateSpecParams> combineShapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::scatterNDUpdateSpecParams> resVec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            auto indices_shape = item.first;
            size_t indices_rank = indices_shape.size();
            std::vector<size_t> update_shape;
            for (size_t i = 0; i < indices_rank - 1; i++) {
                update_shape.push_back(indices_shape[i]);
            }
            auto src_shape = input_shape.first;
            for (size_t j = indices_shape[indices_rank - 1]; j < src_shape.size(); j++) {
                update_shape.push_back(src_shape[j]);
            }
            std::vector<ov::Shape> in_shapes{src_shape, update_shape};
            resVec.push_back(
                ov::test::scatterNDUpdateSpecParams{
                                ov::test::static_shapes_to_test_representation(in_shapes),
                                ov::Shape{indices_shape},
                                item.second});
        }
    }
    return resVec;
}

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> idxPrecisions = {
        ov::element::i32,
        ov::element::i64,
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterNDUpdate,
    ScatterNDUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(combineShapes(sliceSelectInShape)),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterNDUpdateLayerTest::getTestCaseName);
}  // namespace
