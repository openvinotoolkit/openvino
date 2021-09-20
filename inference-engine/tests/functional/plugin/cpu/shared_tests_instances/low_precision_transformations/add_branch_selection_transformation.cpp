// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/add_branch_selection_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<ngraph::element::Type> netPrecisions = {
    ngraph::element::f32,
};

const std::vector<LayerTestsDefinitions::AddBranchSelectionTestValues> params = {
    {
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            {}
        },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {
            {"Constant", "convolution1"},
            {"Constant", "convolution2"},
            {"fakeQuantizeBefore1", "convolution1"},
            {"fakeQuantizeBefore2", "convolution2"},
            {"maxPool", "result"}
        }
    },
    {
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            {}
        },
        {
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
            {
                {},
                { std::vector<float>(9, 1.f), ngraph::element::i8, {3, 3, 1, 1} },
                { {ngraph::element::f32}, {}, {std::vector<float>(3, 1.f), ngraph::element::f32, {3, 1, 1, 1}} }
            },
            { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        },
        { 256ul, ngraph::Shape { 1, 1, 1, 1 }, { 0.f }, { 2.55f }, { 0.f }, { 2.55f } },
        {
            {"Constant", "convolution1"},
            {"Constant", "convolution2"},
            {"fakeQuantizeBefore1", "convolution1"},
            {"fakeQuantizeBefore2", "convolution2"},
            {"maxPool", "result"}
        }
    }
};

INSTANTIATE_TEST_SUITE_P(smoke_LPT, AddBranchSelectionTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(ngraph::PartialShape({ 1, 3, 16, 16 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(params)),
    AddBranchSelectionTransformation::getTestCaseName);
}  // namespace
