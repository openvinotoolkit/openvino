// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/elementwise_branch_selection_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
const std::vector<ov::element::Type> netPrecisions = {
        ov::element::f32
};

const std::vector<std::string> elementwiseTypes = {
    "add",
    "multiply"
};

const std::vector<LayerTestsDefinitions::ElementwiseBranchSelectionTestValues> params = {
    {
        {
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ov::element::i8, {3, 3, 1, 1} },
                { {ov::element::f32}, {}, {std::vector<float>(3, 1.f), ov::element::f32, {3, 1, 1, 1}} }
            },
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        },
        {
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ov::element::i8, {3, 3, 1, 1} },
                { {ov::element::f32}, {}, {std::vector<float>(3, 1.f), ov::element::f32, {3, 1, 1, 1}} }
            },
            {}
        },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {
            {"fakeQuantizeBefore1", "convolution1"},
            {"fakeQuantizeBefore2", "convolution2"},
            {"maxPool", "result"}
        },
        {
            {"convolution1", "u8"},
            {"convolution2", "u8"},
            {"eltwise", "u8"}
        }
    },
    {
        {
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ov::element::i8, {3, 3, 1, 1} },
                { {ov::element::f32}, {}, {std::vector<float>(3, 1.f), ov::element::f32, {3, 1, 1, 1}} }
            },
            {}
        },
        {
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ov::element::i8, {3, 3, 1, 1} },
                { {ov::element::f32}, {}, {std::vector<float>(3, 1.f), ov::element::f32, {3, 1, 1, 1}} }
            },
            { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        },
        { 256ul, ov::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {
            {"fakeQuantizeBefore1", "convolution1"},
            {"fakeQuantizeBefore2", "convolution2"},
            {"maxPool", "result"}
        },
        {
            {"convolution1", "u8"},
            {"convolution2", "u8"},
            {"eltwise", "u8"}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, ElementwiseBranchSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ov::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(ov::test::utils::DEVICE_CPU),
        ::testing::ValuesIn(params),
        ::testing::ValuesIn(elementwiseTypes)),
    ElementwiseBranchSelectionTransformation::getTestCaseName);
}  // namespace
