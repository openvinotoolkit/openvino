// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "subgraph_tests/conv_mul_fusion.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;

namespace {
    const std::vector<ngraph::element::Type> types{ngraph::element::f32, ngraph::element::f16};

    INSTANTIATE_TEST_CASE_P(Convolution_1D, ConvMultiply,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 8, 64}),
                                    ::testing::Values(ngraph::Shape{64, 8, 1}),
                                    ::testing::Values(ngraph::Shape{64, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvMultiply::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(Convolution_2D, ConvMultiply,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::Convolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 3, 64, 64}),
                                    ::testing::Values(ngraph::Shape{64, 3, 1, 1}),
                                    ::testing::Values(ngraph::Shape{64, 1, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvMultiply::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(GroupConvolution_1D, ConvMultiply,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 5}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 2}),
                                    ::testing::Values(ngraph::Shape{20, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvMultiply::getTestCaseName);

    INSTANTIATE_TEST_CASE_P(GroupConvolution_2D, ConvMultiply,
                            ::testing::Combine(
                                    ::testing::Values(ngraph::opset4::GroupConvolution::type_info),
                                    ::testing::Values(ngraph::Shape{1, 12, 64, 64}),
                                    ::testing::Values(ngraph::Shape{4, 5, 3, 1, 2}),
                                    ::testing::Values(ngraph::Shape{1, 20, 1, 1}),
                                    ::testing::ValuesIn(types),
                                    ::testing::Values(CommonTestUtils::DEVICE_CPU)),
                            ConvMultiply::getTestCaseName);
}  // namespace