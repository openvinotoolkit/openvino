// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/conv_eltwise_fusion.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace SubgraphTestsDefinitions;

namespace {
    const std::vector<ngraph::element::Type> types{ngraph::element::f32, ngraph::element::f16};

#define MUL(X) std::tuple<ngraph::NodeTypeInfo, int64_t>(ngraph::opset4::Multiply::type_info, X)
#define ADD(X) std::tuple<ngraph::NodeTypeInfo, int64_t>(ngraph::opset4::Add::type_info, X)
#define IN std::vector<std::tuple<ngraph::NodeTypeInfo, int64_t>>

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_1D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 8, 64}),
                                    ::testing::Values(ngraph::Shape{64, 8, 1}),
                                    ::testing::Values(ngraph::Shape{64, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_1D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 5}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 2}),
                                    ::testing::Values(ngraph::Shape{20, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData_1D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::ConvolutionBackpropData::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 64}),
                                    ::testing::Values(ngraph::Shape{12, 20, 1}),
                                    ::testing::Values(ngraph::Shape{20, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData_1D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolutionBackpropData::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 64}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 1}),
                                    ::testing::Values(ngraph::Shape{1, 20, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    const std::vector<ngraph::Shape> const_shapes_2d{
        {},
        {1, 1},
        {1, 1, 1},
        {20, 1, 1},
        {1, 1, 1, 1}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{20, 3, 1, 1}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 2}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::ConvolutionBackpropData::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{3, 20, 3, 3}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData_2D, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolutionBackpropData::type_info),
                                    ::testing::ValuesIn(IN({MUL(4), ADD(5)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 1, 1}),
                                    ::testing::ValuesIn(const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    const std::vector<ngraph::Shape> neg_const_shapes_2d{
        {1, 1, 1, 1, 1}, /* Broadcast output */
        {3}, {3, 1}, {3, 1, 1, 1}
    };

    INSTANTIATE_TEST_SUITE_P(smoke_Convolution_2D_Negative, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 3, 3, 3}),
                                    ::testing::Values(ngraph::Shape{3, 3, 1, 1}),
                                    ::testing::ValuesIn(neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolution_2D_Negative, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 3, 3}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 1}),
                                    ::testing::ValuesIn(neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_ConvolutionBackpropData_2D_Negative, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::ConvolutionBackpropData::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 3, 3}),
                                    ::testing::Values(ngraph::Shape{12, 3, 1, 1}),
                                    ::testing::ValuesIn(neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);

    INSTANTIATE_TEST_SUITE_P(smoke_GroupConvolutionBackpropData_2D_Negative, ConvEltwiseFusion,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolutionBackpropData::type_info),
                                    ::testing::ValuesIn(IN({MUL(6), ADD(6)})),
                                    ::testing::Values(ngraph::Shape{1, 12, 3, 3}),
                                    ::testing::Values(ngraph::Shape{4, 3, 5, 1, 1}),
                                    ::testing::ValuesIn(neg_const_shapes_2d),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvEltwiseFusion::getTestCaseName);
}  // namespace