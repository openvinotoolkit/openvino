// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "shared_test_classes/single_layer/convolution_backprop.hpp"

using namespace LayerTestsDefinitions;

namespace {

TEST_P(ConvolutionBackpropLayerTest, Serialize) {
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
const std::vector<std::vector<ptrdiff_t>> outPadding = {{}, {1, 1}};
const std::vector<size_t> numOutChannels = {8, 16};
const std::vector<ngraph::op::PadType> pad_types = {
    ngraph::op::PadType::EXPLICIT, ngraph::op::PadType::VALID,
    ngraph::op::PadType::SAME_LOWER, ngraph::op::PadType::SAME_UPPER};
const auto inputShapes = std::vector<size_t>({1, 16, 20, 20});
const std::vector<std::vector<size_t >> emptyOutputShape = {{}};

const auto convolutionBackpropData2DParams = ::testing::Combine(
    ::testing::ValuesIn(kernels), ::testing::ValuesIn(strides),
    ::testing::ValuesIn(padBegins), ::testing::ValuesIn(padEnds),
    ::testing::ValuesIn(dilations), ::testing::ValuesIn(numOutChannels),
    ::testing::ValuesIn(pad_types), ::testing::ValuesIn(outPadding));

INSTANTIATE_TEST_SUITE_P(
    smoke_convolutionBackpropData2D_Serialization, ConvolutionBackpropLayerTest,
    ::testing::Combine(
        convolutionBackpropData2DParams,
        ::testing::ValuesIn(precisions),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(InferenceEngine::Layout::ANY),
        ::testing::Values(inputShapes),
        ::testing::ValuesIn(emptyOutputShape),
        ::testing::Values(CommonTestUtils::DEVICE_CPU)),
    ConvolutionBackpropLayerTest::getTestCaseName);

}   // namespace
