// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "single_op_tests/scatter_elements_update.hpp"
#include "common_test_utils/test_constants.hpp"

namespace {
using ov::test::ScatterElementsUpdateLayerTest;
using ov::test::ScatterElementsUpdate12LayerTest;

// map<inputShape, map<indicesShape, axis>>
std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>> axesShapeInShape {
    {{10, 12, 15}, {{{1, 2, 4}, {0, 1, 2}}, {{2, 2, 2}, {-1, -2, -3}}}},
    {{15, 9, 8, 12}, {{{1, 2, 2, 2}, {0, 1, 2, 3}}, {{1, 2, 1, 4}, {-1, -2, -3, -4}}}},
    {{9, 9, 8, 8, 11, 10}, {{{1, 2, 1, 2, 1, 2}, {5, -3}}}},
};

// index value should not be random data
const std::vector<std::vector<size_t>> idxValue = {
        {1, 0, 4, 6, 2, 3, 7, 5}
};

const std::vector<ov::element::Type> inputPrecisions = {
        ov::element::f32,
        ov::element::f16,
        ov::element::i32,
};

const std::vector<ov::element::Type> idxPrecisions = {
        ov::element::i32,
        ov::element::i64,
};

std::vector<ov::test::axisShapeInShape> combine_shapes(
    const std::map<std::vector<size_t>, std::map<std::vector<size_t>, std::vector<int>>>& input_shapes) {
    std::vector<ov::test::axisShapeInShape> res_vec;
    for (auto& input_shape : input_shapes) {
        for (auto& item : input_shape.second) {
            for (auto& elt : item.second) {
                res_vec.push_back(ov::test::axisShapeInShape{
                    ov::test::static_shapes_to_test_representation({input_shape.first, item.first}),
                    elt});
            }
        }
    }
    return res_vec;
}

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterEltsUpdate,
    ScatterElementsUpdateLayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxValue),
                       ::testing::ValuesIn(inputPrecisions),
                       ::testing::ValuesIn(idxPrecisions),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdateLayerTest::getTestCaseName);


const std::vector<ov::op::v12::ScatterElementsUpdate::Reduction> reduceModes{
    // Reduction::NONE is omitted intentionally, because v12 with Reduction::NONE is converted to v3,
    // and v3 is already tested by smoke_ScatterEltsUpdate testsuite. It doesn't make sense to test the same code twice.
    // Don't forget to add Reduction::NONE when/if ConvertScatterElementsUpdate12ToScatterElementsUpdate3
    // transformation will be disabled (in common transforamtions pipeline or for GPU only).
    ov::op::v12::ScatterElementsUpdate::Reduction::SUM,
    ov::op::v12::ScatterElementsUpdate::Reduction::PROD,
    ov::op::v12::ScatterElementsUpdate::Reduction::MIN,
    ov::op::v12::ScatterElementsUpdate::Reduction::MAX,
    ov::op::v12::ScatterElementsUpdate::Reduction::MEAN
};

const std::vector<std::vector<int64_t>> idxWithNegativeValues = {
    {1, 0, 4, 6, 2, 3, 7, 5},
    {-1, 0, -4, -6, -2, -3, -7, -5},
};

INSTANTIATE_TEST_SUITE_P(
    smoke_ScatterEltsUpdate12,
    ScatterElementsUpdate12LayerTest,
    ::testing::Combine(::testing::ValuesIn(combine_shapes(axesShapeInShape)),
                       ::testing::ValuesIn(idxWithNegativeValues),
                       ::testing::ValuesIn(reduceModes),
                       ::testing::ValuesIn({true, false}),
                       ::testing::Values(inputPrecisions[0]),
                       ::testing::Values(idxPrecisions[0]),
                       ::testing::Values(ov::test::utils::DEVICE_GPU)),
    ScatterElementsUpdate12LayerTest::getTestCaseName);
}  // namespace
