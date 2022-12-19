// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "../skip_tests_check.hpp"
#include "common_test_utils/test_constants.hpp"
#include "single_layer_tests/group_convolution.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine;

namespace {

class GnaDwscLayerTest : public GroupConvolutionLayerTest, public GnaLayerTestCheck {
protected:
    void Run() override {
        GroupConvolutionLayerTest::Run();
    }

    void SetUp() override {
        GnaLayerTestCheck::SetUp(CommonTestUtils::DEVICE_GNA);
        if (GnaLayerTestCheck::gnaLibVersionLessThan("3.6")) {
            GTEST_SKIP() << GnaLayerTestCheck::getLastCmpResultMsg() << std::endl;
        }
        configuration["GNA_DEVICE_MODE"] = "GNA_SW_EXACT";
        configuration["GNA_COMPILE_TARGET"] = "GNA_TARGET_3_6";
        GroupConvolutionLayerTest::SetUp();
    }
};

TEST_P(GnaDwscLayerTest, CompareWithRefs) {
    Run();
}

const std::vector<InferenceEngine::Precision> netPrecisions = {InferenceEngine::Precision::FP32,
                                                               InferenceEngine::Precision::FP16};

/* ============= 2D Convolution ============= */
const std::vector<std::vector<size_t>> kernels_h1 = {{1, 3}, {1, 5}};
const std::vector<std::vector<size_t>> strides_h1 = {{1, 1}, {1, 3}};
const std::vector<std::vector<ptrdiff_t>> pad_begins_h1 = {{0, 2}};
const std::vector<std::vector<ptrdiff_t>> pad_ends_h1 = {{0, 2}};
const std::vector<std::vector<size_t>> dilations = {{1, 1}};
const std::vector<std::vector<size_t>> input_shapes_h1 = {{1, 8, 1, 32}, {1, 8, 1, 160}, {1, 8, 1, 64}};
const std::vector<size_t> num_out_channels_h1 = {8};
const std::vector<std::vector<size_t>> kernels_w1 = {{3, 1}, {5, 1}};
const std::vector<std::vector<size_t>> strides_w1 = {{1, 1}, {3, 1}};
const std::vector<std::vector<ptrdiff_t>> pad_begins_w1 = {{2, 0}};
const std::vector<std::vector<ptrdiff_t>> pad_ends_w1 = {{2, 0}};
const std::vector<std::vector<size_t>> input_shapes_w1 = {{1, 32, 32, 1}, {1, 32, 160, 1}, {1, 32, 64, 1}};
const std::vector<size_t> num_out_channels_w1 = {32};

const auto conv_2d_params_auto_pad_valid_height_1 =
    ::testing::Combine(::testing::ValuesIn(kernels_h1),
                       ::testing::ValuesIn(strides_h1),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                       ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(num_out_channels_h1),
                       ::testing::ValuesIn(num_out_channels_h1),
                       ::testing::Values(ngraph::op::PadType::VALID));
const auto conv_2d_params_auto_pad_valid_width_1 = ::testing::Combine(::testing::ValuesIn(kernels_w1),
                                                                      ::testing::ValuesIn(strides_w1),
                                                                      ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                      ::testing::Values(std::vector<ptrdiff_t>({0, 0})),
                                                                      ::testing::ValuesIn(dilations),
                                                                      ::testing::ValuesIn(num_out_channels_w1),
                                                                      ::testing::ValuesIn(num_out_channels_w1),
                                                                      ::testing::Values(ngraph::op::PadType::VALID));

const auto conv_2d_params_explicit_padding_height_1 =
    ::testing::Combine(::testing::ValuesIn(kernels_h1),
                       ::testing::ValuesIn(strides_h1),
                       ::testing::ValuesIn(pad_begins_h1),
                       ::testing::ValuesIn(pad_ends_h1),
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(num_out_channels_h1),
                       ::testing::ValuesIn(num_out_channels_h1),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv_2d_params_explicit_padding_width_1 =
    ::testing::Combine(::testing::ValuesIn(kernels_w1),
                       ::testing::ValuesIn(strides_w1),
                       ::testing::ValuesIn(pad_begins_w1),
                       ::testing::ValuesIn(pad_ends_w1),
                       ::testing::ValuesIn(dilations),
                       ::testing::ValuesIn(num_out_channels_w1),
                       ::testing::ValuesIn(num_out_channels_w1),
                       ::testing::Values(ngraph::op::PadType::EXPLICIT));

const std::vector<std::vector<size_t>> kernels_2d = {
    {5, 1},
    {4, 1},
    {1, 3},
    {1, 2},
    {2, 2},
    {7, 1},
    {3, 3},
};

const std::vector<std::vector<size_t>> kernels_2d_big = {
    {7, 2},
    {2, 7},
    {3, 7},
    {6, 6},
    {7, 7},
};

const std::vector<std::vector<size_t>> kernels_2d_3x3 = {
    {3, 3},
};
const std::vector<std::vector<size_t>> kernels_2d_5x6 = {
    {5, 6},
};

const std::vector<std::vector<size_t>> strides_2d = {
    {1, 1},
};
const std::vector<std::vector<ptrdiff_t>> pad_begins_2d = {
    {0, 0},
};
const std::vector<std::vector<ptrdiff_t>> pad_ends_2d = {
    {0, 0},
};

const std::vector<size_t> input_2d_nchw = {1, 8, 20, 16};

const std::vector<size_t> input_2d_nchw_3x3 = {1, 16, 3, 3};
const std::vector<size_t> input_2d_nchw_5x6 = {1, 16, 5, 6};

const std::vector<size_t> num_out_channels_2d = {8};
const std::vector<size_t> num_out_channels_3x3 = {16};

const auto conv_2d_params_kernels_2d = ::testing::Combine(::testing::ValuesIn(kernels_2d),
                                                          ::testing::ValuesIn(strides_2d),
                                                          ::testing::ValuesIn(pad_begins_2d),
                                                          ::testing::ValuesIn(pad_ends_2d),
                                                          ::testing::ValuesIn(dilations),
                                                          ::testing::ValuesIn(num_out_channels_2d),
                                                          ::testing::ValuesIn(num_out_channels_2d),
                                                          ::testing::Values(ngraph::op::PadType::EXPLICIT));
const auto conv_2d_params_kernels_2d_big = ::testing::Combine(::testing::ValuesIn(kernels_2d_big),
                                                              ::testing::ValuesIn(strides_2d),
                                                              ::testing::ValuesIn(pad_begins_2d),
                                                              ::testing::ValuesIn(pad_ends_2d),
                                                              ::testing::ValuesIn(dilations),
                                                              ::testing::ValuesIn(num_out_channels_2d),
                                                              ::testing::ValuesIn(num_out_channels_2d),
                                                              ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv_2d_params_kernels_2d_3x3 = ::testing::Combine(::testing::ValuesIn(kernels_2d_3x3),
                                                              ::testing::ValuesIn(strides_2d),
                                                              ::testing::ValuesIn(pad_begins_2d),
                                                              ::testing::ValuesIn(pad_ends_2d),
                                                              ::testing::ValuesIn(dilations),
                                                              ::testing::ValuesIn(num_out_channels_3x3),
                                                              ::testing::ValuesIn(num_out_channels_3x3),
                                                              ::testing::Values(ngraph::op::PadType::EXPLICIT));

const auto conv_2d_params_kernels_2d_5x6 = ::testing::Combine(::testing::ValuesIn(kernels_2d_5x6),
                                                              ::testing::ValuesIn(strides_2d),
                                                              ::testing::ValuesIn(pad_begins_2d),
                                                              ::testing::ValuesIn(pad_ends_2d),
                                                              ::testing::ValuesIn(dilations),
                                                              ::testing::ValuesIn(num_out_channels_3x3),
                                                              ::testing::ValuesIn(num_out_channels_3x3),
                                                              ::testing::Values(ngraph::op::PadType::EXPLICIT));

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_auto_pad_valid_height_1,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_auto_pad_valid_height_1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(input_shapes_h1),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_auto_pad_valid_width_1,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_auto_pad_valid_width_1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(input_shapes_w1),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_explicit_padding_height_1,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_explicit_padding_height_1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(input_shapes_h1),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_explicit_padding_width_1,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_explicit_padding_width_1,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::ValuesIn(input_shapes_w1),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_kernels_2d,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_kernels_2d,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input_2d_nchw),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_kernels_2d_big,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_kernels_2d_big,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input_2d_nchw),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_map_2d_not_transpose_h_w_3_3,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_kernels_2d_3x3,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input_2d_nchw_3x3),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(smoke_conv_2d_map_2d_not_transpose_h_w_5_6,
                         GnaDwscLayerTest,
                         ::testing::Combine(conv_2d_params_kernels_2d_5x6,
                                            ::testing::ValuesIn(netPrecisions),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Precision::UNSPECIFIED),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(InferenceEngine::Layout::ANY),
                                            ::testing::Values(input_2d_nchw_5x6),
                                            ::testing::Values(CommonTestUtils::DEVICE_GNA)),
                         GnaDwscLayerTest::getTestCaseName);
}  // namespace
