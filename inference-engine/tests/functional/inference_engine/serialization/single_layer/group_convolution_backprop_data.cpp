// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/group_convolution_backprop_data.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(GroupConvBackpropDataLayerTest, Serialize) {
    Serialize();
}

const std::vector<InferenceEngine::Precision> precisions = {
    InferenceEngine::Precision::FP64, InferenceEngine::Precision::FP32,
    InferenceEngine::Precision::FP16, InferenceEngine::Precision::BF16,
    InferenceEngine::Precision::I8,   InferenceEngine::Precision::I16,
    InferenceEngine::Precision::I32,  InferenceEngine::Precision::I64,
    InferenceEngine::Precision::U8,   InferenceEngine::Precision::U16,
    InferenceEngine::Precision::U32,  InferenceEngine::Precision::U64,
};
const std::vector<std::vector<size_t>> kernels = {{3, 3}};
const std::vector<std::vector<size_t>> strides = {{1, 1}};
const std::vector<std::vector<ptrdiff_t>> padBegins = {{0, 0}};
const std::vector<std::vector<ptrdiff_t>> padEnds = {{0, 0}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<size_t> numGroups = {2, 8};
const std::vector<ngraph::op::PadType> pad_types = {
    ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID,
    ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER};
const auto inputShapes = std::vector<size_t>({1, 16, 30, 30});

const auto groupConvBackpropData2DParams = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::ValuesIn(numGroups), ::testing::ValuesIn(pad_types));

INSTANTIATE_TEST_SUITE_P(
    smoke_GroupConvBackpropData2D_Serialization, GroupConvBackpropDataLayerTest,
    ::testing::Combine(
        groupConvBackpropData2DParams,
        ::testing::ValuesIn(precisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(inputShapes),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    GroupConvBackpropDataLayerTest::getTestCaseName);

}   // namespace
