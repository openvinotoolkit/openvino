// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "../skip_tests_check.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "openvino/runtime/intel_gna/properties.hpp"
#include "shared_test_classes/single_layer/convolution.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels2D = {{1, 3}, {7, 1}, {3, 3}, {7, 2}, {2, 7}};

const std::vector<std::vector<size_t>> kernels2DInvalid = {
    {9, 3},
    {1, 9},
    {1, 8},
    {8, 1},
    {8, 8},
};

const std::vector<std::vector<size_t>> kernels2DInvalidFor56InC = {
    {1, 6},
    {2, 6},
    {7, 7},
    {1, 7},
    {4, 7},
};

const std::vector<std::vector<size_t>> kernels2DInvalidFor120InC = {
    {1, 4},
    {8, 3},
    {7, 5},
    {1, 6},
    {4, 7},
};

const std::vector<std::vector<size_t>> strides2D = {
    {1, 1},
};
const std::vector<std::vector<size_t>> strides2DInvalid = {{8, 8}, {1, 8}};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = {
    {0, 0},
};
// Padding must be less than kernel size
const std::vector<std::vector<ptrdiff_t>> padTooBigForKernels2D = {
    {3, 3},
};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = {
    {0, 0},
};
const std::vector<std::vector<ptrdiff_t>> padBegins2DInvalid = {{1, 0}, {1, 1}, {0, 1}};
const std::vector<std::vector<ptrdiff_t>> padEnds2DInvalid = {{1, 0}, {1, 1}, {0, 1}};
const std::vector<std::vector<size_t>> dilations2D = {
    {1, 1},
};
const std::vector<std::vector<size_t>> dilations2DInvalid = {
    {2, 2},
};
const std::vector<size_t> numOutChannels2D = {32};
const std::vector<size_t> numOutChannels2DInvalid = {1, 7, 9, 1032};

const std::vector<std::vector<size_t>> input2DNCHWFine = {{1, 8, 20, 16}};

const std::vector<std::vector<size_t>> input2DNCHWWithInC56 = {{1, 56, 20, 16}};
const std::vector<std::vector<size_t>> input2DNCHWWithInC120 = {{1, 120, 20, 16}};

const std::vector<std::vector<size_t>> input2DNCHWInvalidInputC = {{1, 7, 20, 16}, {1, 9, 20, 16}, {1, 400, 20, 16}};
const std::vector<std::vector<size_t>> input2DNCHWInvalidInputH = {{1, 8, 15, 16}, {1, 8, 400, 16}};
const std::vector<std::vector<size_t>> input2DNCHWInvalidInputW = {{1, 8, 20, 14}, {1, 8, 20, 400}};

const auto conv2DParametersFine = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                     ::testing::ValuesIn(strides2D),
                                                     ::testing::ValuesIn(padBegins2D),
                                                     ::testing::ValuesIn(padEnds2D),
                                                     ::testing::ValuesIn(dilations2D),
                                                     ::testing::ValuesIn(numOutChannels2D),
                                                     ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParametersInvalidKernel = ::testing::Combine(::testing::ValuesIn(kernels2DInvalid),
                                                              ::testing::ValuesIn(strides2D),
                                                              ::testing::ValuesIn(padBegins2D),
                                                              ::testing::ValuesIn(padEnds2D),
                                                              ::testing::ValuesIn(dilations2D),
                                                              ::testing::ValuesIn(numOutChannels2D),
                                                              ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv2DParametersInvalidKernelFor56InC = ::testing::Combine(::testing::ValuesIn(kernels2DInvalidFor56InC),
                                                                      ::testing::ValuesIn(strides2D),
                                                                      ::testing::ValuesIn(padBegins2D),
                                                                      ::testing::ValuesIn(padEnds2D),
                                                                      ::testing::ValuesIn(dilations2D),
                                                                      ::testing::ValuesIn(numOutChannels2D),
                                                                      ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv2DParametersInvalidKernelFor120InC =
    ::testing::Combine(::testing::ValuesIn(kernels2DInvalidFor120InC),
                       ::testing::ValuesIn(strides2D),
                       ::testing::ValuesIn(padBegins2D),
                       ::testing::ValuesIn(padEnds2D),
                       ::testing::ValuesIn(dilations2D),
                       ::testing::ValuesIn(numOutChannels2D),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv2DParametersInvalidFilterNumber = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                                    ::testing::ValuesIn(strides2D),
                                                                    ::testing::ValuesIn(padBegins2D),
                                                                    ::testing::ValuesIn(padEnds2D),
                                                                    ::testing::ValuesIn(dilations2D),
                                                                    ::testing::ValuesIn(numOutChannels2DInvalid),
                                                                    ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParametersInvalidPadding = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                               ::testing::ValuesIn(strides2D),
                                                               ::testing::ValuesIn(padBegins2DInvalid),
                                                               ::testing::ValuesIn(padEnds2DInvalid),
                                                               ::testing::ValuesIn(dilations2D),
                                                               ::testing::ValuesIn(numOutChannels2D),
                                                               ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParametersInvalidStride = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                              ::testing::ValuesIn(strides2DInvalid),
                                                              ::testing::ValuesIn(padBegins2D),
                                                              ::testing::ValuesIn(padEnds2D),
                                                              ::testing::ValuesIn(dilations2D),
                                                              ::testing::ValuesIn(numOutChannels2D),
                                                              ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParametersInvalidDilation = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                                ::testing::ValuesIn(strides2D),
                                                                ::testing::ValuesIn(padBegins2D),
                                                                ::testing::ValuesIn(padEnds2D),
                                                                ::testing::ValuesIn(dilations2DInvalid),
                                                                ::testing::ValuesIn(numOutChannels2D),
                                                                ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv2DParametersInvalidPaddingSize = ::testing::Combine(::testing::ValuesIn(kernels2D),
                                                                   ::testing::ValuesIn(strides2D),
                                                                   ::testing::ValuesIn(padTooBigForKernels2D),
                                                                   ::testing::ValuesIn(padTooBigForKernels2D),
                                                                   ::testing::ValuesIn(dilations2D),
                                                                   ::testing::ValuesIn(numOutChannels2D),
                                                                   ::testing::Values(ngraph::op::PadType::EXPLICIT));

class GnaConv2DNegativeTest : public ConvolutionLayerTest {
protected:
    virtual std::string expectedSubstring() = 0;
    virtual std::string getTarget() = 0;
    void Run() override {
        try {
            ConvolutionLayerTest::LoadNetwork();
            FAIL() << "GNA's unsupported configuration of Convolution2D was not detected in "
                      "ConvolutionLayerTest::LoadNetwork()";
        } catch (std::runtime_error& e) {
            const std::string errorMsg = e.what();
            const auto expected = expectedSubstring();
            ASSERT_STR_CONTAINS(errorMsg, expected);
            EXPECT_TRUE(errorMsg.find(expected) != std::string::npos)
                << "Wrong error message, actula error message: " << errorMsg << ", expected: " << expected;
        }
    }
    void SetUp() override {
        ConvolutionLayerTest::SetUp();
        const auto target = getTarget();
        configuration[ov::intel_gna::execution_target.name()] = target;
        configuration[ov::intel_gna::compile_target.name()] = target;
    }
};

#define GNA_NEG_INSTANTIATE(whats_wrong, suffix_params, suffix_input, error_message, gna_hw_gen)            \
    struct GnaConv2DNegativeTest##whats_wrong : GnaConv2DNegativeTest {                                     \
        std::string expectedSubstring() override {                                                          \
            return error_message;                                                                           \
        }                                                                                                   \
        std::string getTarget() override {                                                                  \
            std::stringstream s;                                                                            \
            s << gna_hw_gen;                                                                                \
            return s.str();                                                                                 \
        }                                                                                                   \
    };                                                                                                      \
    TEST_P(GnaConv2DNegativeTest##whats_wrong, ThrowAsNotSupported) {                                       \
        Run();                                                                                              \
    }                                                                                                       \
    INSTANTIATE_TEST_SUITE_P(smoke_GnaConv2DNegativeTestInvalid##whats_wrong,                               \
                             GnaConv2DNegativeTest##whats_wrong,                                            \
                             ::testing::Combine(conv2DParameters##suffix_params,                            \
                                                ::testing::ValuesIn(netPrecisions),                         \
                                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), \
                                                ::testing::Values(InferenceEngine::Precision::UNSPECIFIED), \
                                                ::testing::Values(InferenceEngine::Layout::ANY),            \
                                                ::testing::Values(InferenceEngine::Layout::ANY),            \
                                                ::testing::ValuesIn(input2DNCHW##suffix_input),             \
                                                ::testing::Values(ov::test::utils::DEVICE_GNA)),            \
                             GnaConv2DNegativeTest##whats_wrong::getTestCaseName);

constexpr auto GNA_3_0 = ov::intel_gna::HWGeneration::GNA_3_0;
constexpr auto GNA_3_5 = ov::intel_gna::HWGeneration::GNA_3_5;

GNA_NEG_INSTANTIATE(FilterNumber, InvalidFilterNumber, Fine, "Unsupported number of kernels", GNA_3_0)
GNA_NEG_INSTANTIATE(Kernel, InvalidKernel, Fine, "Unsupported kernel shape", GNA_3_0)
GNA_NEG_INSTANTIATE(BigKernelFor56InC, InvalidKernelFor56InC, WithInC56, "Unsupported kernel shape", GNA_3_0)
GNA_NEG_INSTANTIATE(BigKernelFor120InC, InvalidKernelFor120InC, WithInC120, "Unsupported kernel shape", GNA_3_0)
GNA_NEG_INSTANTIATE(InputH, Fine, InvalidInputH, "Unsupported input height", GNA_3_0)
GNA_NEG_INSTANTIATE(InputW, Fine, InvalidInputW, "Unsupported input width", GNA_3_0)
GNA_NEG_INSTANTIATE(InputC, Fine, InvalidInputC, "Unsupported number of input channels", GNA_3_0)
GNA_NEG_INSTANTIATE(Padding, InvalidPadding, Fine, "Unsupported convolution input padding", GNA_3_0)
GNA_NEG_INSTANTIATE(Stride, InvalidStride, Fine, "Unsupported convolution stride shape", GNA_3_0)
GNA_NEG_INSTANTIATE(Dilation, InvalidDilation, Fine, "Unsupported dilation", GNA_3_0)
GNA_NEG_INSTANTIATE(Dilation35, InvalidDilation, Fine, "Unsupported dilation", GNA_3_5)
GNA_NEG_INSTANTIATE(PaddingSize, InvalidPaddingSize, Fine, "Unsupported convolution input padding", GNA_3_0)
GNA_NEG_INSTANTIATE(PaddingSize35, InvalidPaddingSize, Fine, "Unsupported convolution input padding", GNA_3_5)
}  // namespace
