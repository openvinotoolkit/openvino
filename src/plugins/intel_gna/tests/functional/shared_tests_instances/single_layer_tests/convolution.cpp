// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "../skip_tests_check.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/opsets/opset11.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"
#include "shared_test_classes/base/layer_test_utils.hpp"

using namespace ov;
using namespace ov::op;
using namespace ov::opset11;

namespace {

// ! [test_convolution:definition]
typedef std::tuple<InferenceEngine::SizeVector,  // Kernel size
                   InferenceEngine::SizeVector,  // Strides
                   std::vector<ptrdiff_t>,       // Pad begin
                   std::vector<ptrdiff_t>,       // Pad end
                   InferenceEngine::SizeVector,  // Dilation
                   size_t,                       // Num out channels
                   PadType                       // Padding type
                   >
    convSpecificParams;

typedef std::tuple<convSpecificParams,
                   InferenceEngine::Precision,         // Net precision
                   InferenceEngine::Precision,         // Input precision
                   InferenceEngine::Precision,         // Output precision
                   InferenceEngine::Layout,            // Input layout
                   InferenceEngine::Layout,            // Output layout
                   InferenceEngine::SizeVector,        // Input shapes
                   LayerTestsUtils::TargetDevice,      // Device name
                   std::map<std::string, std::string>  // GNA Config params
                   >
    ConvLayerTestFixtureParamsSet;

class ConvolutionLayerTestFixture : public testing::WithParamInterface<ConvLayerTestFixtureParamsSet>,
                                    virtual public LayerTestsUtils::LayerTestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvLayerTestFixtureParamsSet>& obj);

protected:
    void SetUp() override;
};

std::string ConvolutionLayerTestFixture::getTestCaseName(
    const testing::TestParamInfo<ConvLayerTestFixtureParamsSet>& obj) {
    convSpecificParams convParams;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::Precision inPrc, outPrc;
    InferenceEngine::Layout inLayout, outLayout;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::map<std::string, std::string> config;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShapes, targetDevice, config) =
        obj.param;
    PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;

    std::ostringstream result;
    result << "IS=" << ov::test::utils::vec2str(inputShapes) << "_";
    result << "K" << ov::test::utils::vec2str(kernel) << "_";
    result << "S" << ov::test::utils::vec2str(stride) << "_";
    result << "PB" << ov::test::utils::vec2str(padBegin) << "_";
    result << "PE" << ov::test::utils::vec2str(padEnd) << "_";
    result << "D=" << ov::test::utils::vec2str(dilation) << "_";
    result << "O=" << convOutChannels << "_";
    result << "AP=" << padType << "_";
    result << "netPRC=" << netPrecision.name() << "_";
    result << "inPRC=" << inPrc.name() << "_";
    result << "outPRC=" << outPrc.name() << "_";
    result << "inL=" << inLayout << "_";
    result << "outL=" << outLayout << "_";
    result << "trgDev=" << targetDevice;
    for (auto const& config_entry : config) {
        result << "_config_entry=" << config_entry.first << "_" << config_entry.second;
    }
    return result.str();
}

void ConvolutionLayerTestFixture::SetUp() {
    convSpecificParams convParams;
    std::vector<size_t> inputShape;
    auto netPrecision = InferenceEngine::Precision::UNSPECIFIED;
    std::tie(convParams, netPrecision, inPrc, outPrc, inLayout, outLayout, inputShape, targetDevice, configuration) =
        this->GetParam();
    PadType padType;
    InferenceEngine::SizeVector kernel, stride, dilation;
    std::vector<ptrdiff_t> padBegin, padEnd;
    size_t convOutChannels;
    std::tie(kernel, stride, padBegin, padEnd, dilation, convOutChannels, padType) = convParams;
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);
    ov::ParameterVector params{std::make_shared<ov::op::v0::Parameter>(ngPrc, ov::Shape(inputShape))};
    std::vector<float> filter_weights;

    auto filter_size = std::accumulate(std::begin(kernel), std::end(kernel), 1, std::multiplies<size_t>());
    filter_weights =
        ov::test::utils::generate_float_numbers(convOutChannels * inputShape[1] * filter_size, -0.1f, 0.1f);

    auto conv = std::dynamic_pointer_cast<Convolution>(ngraph::builder::makeConvolution(params[0],
                                                                                        ngPrc,
                                                                                        kernel,
                                                                                        stride,
                                                                                        padBegin,
                                                                                        padEnd,
                                                                                        dilation,
                                                                                        padType,
                                                                                        convOutChannels,
                                                                                        false,
                                                                                        filter_weights));
    ResultVector results{std::make_shared<Result>(conv)};
    function = std::make_shared<Model>(results, params, "convolution");
}

TEST_P(ConvolutionLayerTestFixture, CompareWithRefs) {
    Run();
}
const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernelsH1 = {{1, 3}, {1, 5}};
const std::vector<std::vector<size_t>> stridesH1 = {{1, 1}, {1, 3}};
const std::vector<std::vector<size_t>> stridesH1Big = {{1, 9}, {1, 16}};
const std::vector<std::vector<ptrdiff_t>> padBeginsH1 = {{1, 0}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> padEndsH1 = {{1, 0}, {1, 3}};
const std::vector<std::vector<size_t>> dilationsH1 = {{1, 1}, {1, 3}};
const std::vector<std::vector<size_t>> inputShapesH1 = {{1, 1, 1, 32}, {1, 32, 1, 160}, {1, 8, 1, 64}};
const std::vector<std::vector<size_t>> kernelsW1 = {{3, 1}, {5, 1}};
const std::vector<std::vector<size_t>> stridesW1 = {{1, 1}, {3, 1}};
const std::vector<std::vector<ptrdiff_t>> padBeginsW1 = {{0, 1}, {3, 1}};
const std::vector<std::vector<ptrdiff_t>> padEndsW1 = {{0, 1}, {3, 1}};
const std::vector<std::vector<size_t>> dilationsW1 = {{1, 1}, {3, 1}};
const std::vector<std::vector<size_t>> inputShapesW1 = {{1, 1, 32, 1}, {1, 32, 160, 1}, {1, 8, 64, 1}};
const std::vector<size_t> numOutCannels = {4, 8, 12};

const std::vector<std::vector<size_t>> kernels2D = {
    {5, 1},
    {4, 1},
    {1, 3},
    {1, 2},
    {2, 2},
    {7, 1},
    {3, 3},
};

const std::vector<std::vector<size_t>> kernels2D_big = {
    {7, 2},
    {2, 7},
    {3, 7},
    {6, 6},
    {7, 7},
};

const std::vector<std::vector<size_t>> kernels2D_3x3 = {
    {3, 3},
};
const std::vector<std::vector<size_t>> kernels2D_5x6 = {
    {5, 6},
};

const std::vector<std::vector<size_t>> strides2D = {
    {1, 1},
};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {
    {0, 0},
};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {
    {0, 0},
};
const std::vector<std::vector<size_t>> dilations2D = {
    {1, 1},
};
const std::vector<size_t> numOutCannels2D = {8, 16, 32};

const std::vector<size_t> num_out_channels_for_mapped_2d = {4, 8, 12};

const std::vector<size_t> input2DNCHW = {1, 8, 20, 16};

const std::vector<size_t> input2DNCHW_3x3 = {1, 16, 3, 3};
const std::vector<size_t> input2DNCHW_5x6 = {1, 16, 5, 6};

const std::vector<std::vector<size_t>> inputShapesMapTo1d = {{1, 1, 56, 5}, {1, 32, 56, 5}, {1, 2, 64, 5}};

std::vector<std::map<std::string, std::string>> configs = {
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_0"}},
    {{"GNA_DEVICE_MODE", "GNA_SW_EXACT"}, {"GNA_EXEC_TARGET", "GNA_TARGET_3_5"}}};

const auto conv2DParams_Kernels2D = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                       ::testing::ValuesIn(strides2D),
                                                       ::testing::ValuesIn(padBegins2D),
                                                       ::testing::ValuesIn(padEnds2D),
                                                       ::testing::ValuesIn(dilations2D),
                                                       ::testing::ValuesIn(numOutCannels2D),
                                                       ::testing::Values(PadType::EXPLICIT));
const auto conv2DParams_Kernels2D_big = ::testing::Combine(::testing::ValuesIn(kernels2D_big),
                                                           ::testing::ValuesIn(strides2D),
                                                           ::testing::ValuesIn(padBegins2D),
                                                           ::testing::ValuesIn(padEnds2D),
                                                           ::testing::ValuesIn(dilations2D),
                                                           ::testing::ValuesIn(numOutCannels2D),
                                                           ::testing::Values(PadType::EXPLICIT));

const auto conv2DParams_Kernels2D_3x3 = ::testing::Combine(::testing::ValuesIn(kernels2D_3x3),
                                                           ::testing::ValuesIn(strides2D),
                                                           ::testing::ValuesIn(padBegins2D),
                                                           ::testing::ValuesIn(padEnds2D),
                                                           ::testing::ValuesIn(dilations2D),
                                                           ::testing::ValuesIn(num_out_channels_for_mapped_2d),
                                                           ::testing::Values(PadType::EXPLICIT));

const auto conv2DParams_Kernels2D_5x6 = ::testing::Combine(::testing::ValuesIn(kernels2D_5x6),
                                                           ::testing::ValuesIn(strides2D),
                                                           ::testing::ValuesIn(padBegins2D),
                                                           ::testing::ValuesIn(padEnds2D),
                                                           ::testing::ValuesIn(dilations2D),
                                                           ::testing::ValuesIn(num_out_channels_for_mapped_2d),
                                                           ::testing::Values(PadType::EXPLICIT));

const auto conv2DParams_ExplicitPadding_Height1 = ::testing::Combine(::testing::ValuesIn(kernelsH1),
                                                                     ::testing::ValuesIn(stridesH1),
                                                                     ::testing::ValuesIn(padBeginsH1),
                                                                     ::testing::ValuesIn(padEndsH1),
                                                                     ::testing::ValuesIn(dilationsH1),
                                                                     ::testing::ValuesIn(numOutCannels),
                                                                     ::testing::Values(PadType::EXPLICIT));
const auto conv2DParams_ExplicitPadding_Width1 = ::testing::Combine(::testing::ValuesIn(kernelsW1),
                                                                    ::testing::ValuesIn(stridesW1),
                                                                    ::testing::ValuesIn(padBeginsW1),
                                                                    ::testing::ValuesIn(padEndsW1),
                                                                    ::testing::ValuesIn(dilationsW1),
                                                                    ::testing::ValuesIn(numOutCannels),
                                                                    ::testing::Values(PadType::EXPLICIT));
const auto conv2DParams_AutoPadValid_Height1 = ::testing::Combine(::testing::ValuesIn(kernelsH1),
                                                                  ::testing::ValuesIn(stridesH1),
                                                                  ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                  ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                  ::testing::ValuesIn(dilationsH1),
                                                                  ::testing::ValuesIn(numOutCannels),
                                                                  ::testing::Values(PadType::VALID));
const auto conv2DParams_AutoPadValid_Height1_BigStride =
    ::testing::Combine(::testing::ValuesIn(kernelsH1),
                       ::testing::ValuesIn(stridesH1Big),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                       ::testing::ValuesIn(dilationsH1),
                       ::testing::ValuesIn(numOutCannels),
                       ::testing::Values(PadType::VALID));
const auto conv2DParams_AutoPadValid_Width1 = ::testing::Combine(::testing::ValuesIn(kernelsW1),
                                                                 ::testing::ValuesIn(stridesW1),
                                                                 ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                 ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                 ::testing::ValuesIn(dilationsW1),
                                                                 ::testing::ValuesIn(numOutCannels),
                                                                 ::testing::Values(PadType::VALID));
const auto conv2DParams_AutoPadValid_MapTo1d = ::testing::Combine(::testing::Values(std::vector<size_t>{3, 5}),
                                                                  ::testing::ValuesIn(stridesW1),
                                                                  ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                  ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                  ::testing::Values(std::vector<size_t>{1, 1}),
                                                                  ::testing::ValuesIn(numOutCannels),
                                                                  ::testing::Values(PadType::VALID));

// TODO: padding isn't currently supported in GNA
INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Convolution2D_ExplicitPadding_Height1,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_ExplicitPadding_Height1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesH1),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(DISABLED_smoke_Convolution2D_ExplicitPadding_Width1,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_ExplicitPadding_Width1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesW1),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid_Height1,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_AutoPadValid_Height1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesH1),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid_Height1_BigStride,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_AutoPadValid_Height1_BigStride,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesH1),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid_Width1,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_AutoPadValid_Width1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesW1),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_AutoPadValid_MapTo1d,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_AutoPadValid_MapTo1d,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(inputShapesMapTo1d),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_Kernels2D,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_Kernels2D,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input2DNCHW),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_Kernels2D_big,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_Kernels2D_big,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input2DNCHW),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_Map2D_Not_Transpose_h_w_3_3,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_Kernels2D_3x3,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input2DNCHW_3x3),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_Convolution2D_Map2D_Not_Transpose_h_w_5_6,
                         ConvolutionLayerTestFixture,
                         ::testing::Combine(conv2DParams_Kernels2D_5x6,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input2DNCHW_5x6),
                                            ::testing::Values(ov::test::utils::DEVICE_GNA),
                                            ::testing::ValuesIn(configs)),
                         ConvolutionLayerTestFixture::getTestCaseName);
}  // namespace
