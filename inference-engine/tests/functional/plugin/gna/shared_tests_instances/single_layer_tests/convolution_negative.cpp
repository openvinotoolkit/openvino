// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include "shared_test_classes/single_layer/convolution.hpp"
#include "common_test_utils/test_assertions.hpp"
#include "common_test_utils/test_constants.hpp"
#include "../skip_tests_check.hpp"

using namespace LayerTestsDefinitions;

namespace {

const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t >> kernels2D = {
    {1, 3},
    {7, 1},
    {3, 3},
};

const std::vector<std::vector<size_t >> kernels2DInvalid = {
    {1, 4},
    {2, 3},
    {3, 2},
    {8, 1},
    {4, 4},
};

const std::vector<std::vector<size_t >> strides2D = {
                                                          {1, 1},
};
const std::vector<std::vector<size_t >> strides2DInvalid = {
                                                          {4, 4}, {1, 4}
};
const std::vector<std::vector<ptrdiff_t>> padBegins2D = { {0, 0},
};
const std::vector<std::vector<ptrdiff_t>> padEnds2D = { {0, 0},
};
const std::vector<std::vector<ptrdiff_t>> padBegins2DInvalid = { {1, 0}, {1, 1}, {0, 1}
};
const std::vector<std::vector<ptrdiff_t>> padEnds2DInvalid = { {1, 0}, {1, 1}, {0, 1}
};
const std::vector<std::vector<size_t >> dilations2D = { {1, 1},
};
const std::vector<std::vector<size_t >> dilations2DInvalid = { {2, 2},
};
const std::vector<size_t> numOutChannels2D = { 32 };
const std::vector<size_t> numOutChannels2DInvalid = { 1, 7, 9, 400 };

const std::vector<std::vector<size_t>> input2DNCHWFine = { { 1, 8, 20, 16 } };

const std::vector<std::vector<size_t>> input2DNCHWInvalidInputC = {
    { 1, 7, 20, 16 },
    { 1, 9, 20, 16 },
    { 1, 400, 20, 16 } };
const std::vector<std::vector<size_t>> input2DNCHWInvalidInputH = { { 1, 8, 15, 16 }, { 1, 8, 400, 16 } };
const std::vector<std::vector<size_t>> input2DNCHWInvalidInputW = { { 1, 8, 20, 14 }, { 1, 8, 20, 400 } };

const auto conv2DParametersFine = ::testing::Combine(
    ::testing::ValuesIn(kernels2D),
    ::testing::ValuesIn(strides2D),
    ::testing::ValuesIn(padBegins2D),
    ::testing::ValuesIn(padEnds2D),
    ::testing::ValuesIn(dilations2D),
    ::testing::ValuesIn(numOutChannels2D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParametersInvalidKernel = ::testing::Combine(
    ::testing::ValuesIn(kernels2DInvalid),
    ::testing::ValuesIn(strides2D),
    ::testing::ValuesIn(padBegins2D),
    ::testing::ValuesIn(padEnds2D),
    ::testing::ValuesIn(dilations2D),
    ::testing::ValuesIn(numOutChannels2D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParametersInvalidFilterNumber = ::testing::Combine(
    ::testing::ValuesIn(kernels2D),
    ::testing::ValuesIn(strides2D),
    ::testing::ValuesIn(padBegins2D),
    ::testing::ValuesIn(padEnds2D),
    ::testing::ValuesIn(dilations2D),
    ::testing::ValuesIn(numOutChannels2DInvalid),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParametersInvalidPadding = ::testing::Combine(
    ::testing::ValuesIn(kernels2D),
    ::testing::ValuesIn(strides2D),
    ::testing::ValuesIn(padBegins2DInvalid),
    ::testing::ValuesIn(padEnds2DInvalid),
    ::testing::ValuesIn(dilations2D),
    ::testing::ValuesIn(numOutChannels2D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParametersInvalidStride = ::testing::Combine(
    ::testing::ValuesIn(kernels2D),
    ::testing::ValuesIn(strides2DInvalid),
    ::testing::ValuesIn(padBegins2D),
    ::testing::ValuesIn(padEnds2D),
    ::testing::ValuesIn(dilations2D),
    ::testing::ValuesIn(numOutChannels2D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);
const auto conv2DParametersInvalidDilation = ::testing::Combine(
    ::testing::ValuesIn(kernels2D),
    ::testing::ValuesIn(strides2D),
    ::testing::ValuesIn(padBegins2D),
    ::testing::ValuesIn(padEnds2D),
    ::testing::ValuesIn(dilations2DInvalid),
    ::testing::ValuesIn(numOutChannels2D),
    ::testing::Values(ngraph::op::PadType::EXPLICIT)
);

class GnaConv2DNegativeTest : public ConvolutionLayerTest, protected GnaLayerTestCheck {
protected:
    virtual std::string expectedSubstring() = 0;
    void Run() override {
        GnaLayerTestCheck::SkipTestCheck();

        if (!GnaLayerTestCheck::skipTest) {
            try {
                ConvolutionLayerTest::LoadNetwork();
                FAIL() << "GNA's unsupported configuration of Convolution2D was not detected in ConvolutionLayerTest::LoadNetwork()";
            }
            catch (std::runtime_error& e) {
                const std::string errorMsg = e.what();
                const auto expected = expectedSubstring();
                ASSERT_STR_CONTAINS(errorMsg, expected);
                EXPECT_TRUE(errorMsg.find(expected) != std::string::npos) << "Wrong error message, actula error message: " << errorMsg <<
                    ", expected: " << expected;
            }
        }
    }
    void SetUp() override {
        ConvolutionLayerTest::SetUp();
    }
};

#define GNA_NEG_INSTANTIATE(whats_wrong, suffix_params, suffix_input, error_message)                            \
struct GnaConv2DNegativeTest##whats_wrong : GnaConv2DNegativeTest {                                             \
    std::string expectedSubstring() override {                                                                  \
        return error_message;                                                                                   \
    }                                                                                                           \
};                                                                                                              \
TEST_P(GnaConv2DNegativeTest##whats_wrong, ThrowAsNotSupported) {                                               \
    Run();                                                                                                      \
}                                                                                                               \
INSTANTIATE_TEST_CASE_P(smoke_GnaConv2DNegativeTestInvalid##whats_wrong, GnaConv2DNegativeTest##whats_wrong,    \
::testing::Combine(                                                                                             \
    conv2DParameters##suffix_params,                                                                            \
    ::testing::ValuesIn(netPrecisions),                                                                         \
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),                                                 \
    ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),                                                 \
    ::testing::Values(InferenceEngine::Layout::ANY),                                                            \
    ::testing::Values(InferenceEngine::Layout::ANY),                                                            \
    ::testing::ValuesIn(input2DNCHW##suffix_input),                                                             \
    ::testing::Values(CommonTestUtils::DEVICE_GNA)),                                                            \
    GnaConv2DNegativeTest##whats_wrong::getTestCaseName);

GNA_NEG_INSTANTIATE(FilterNumber, InvalidFilterNumber, Fine, "Unsupported number of kernels")
GNA_NEG_INSTANTIATE(Kernel, InvalidKernel, Fine, "Unsupported kernel shape")
GNA_NEG_INSTANTIATE(InputH, Fine, InvalidInputH, "Unsupported input height")
GNA_NEG_INSTANTIATE(InputW, Fine, InvalidInputW, "Unsupported input width")
GNA_NEG_INSTANTIATE(InputC, Fine, InvalidInputC, "Unsupported number of input channels")
GNA_NEG_INSTANTIATE(Padding, InvalidPadding, Fine, "Convolution's input padding is not supported")
GNA_NEG_INSTANTIATE(Stride, InvalidStride, Fine, "Unsupported convolution stride shape")
GNA_NEG_INSTANTIATE(Dilation, InvalidDilation, Fine, "dilation is not supported on GNA")

}  // namespace
