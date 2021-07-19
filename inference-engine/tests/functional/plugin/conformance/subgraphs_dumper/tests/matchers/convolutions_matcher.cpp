// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gtest/gtest.h"
#include "matchers/convolutions.hpp"
#include "ngraph/ops.hpp"
#include "functional_test_utils/include/functional_test_utils/layer_test_utils/op_info.hpp"

using namespace ngraph::op;
using namespace ngraph;
using ngraph::element::Type_t;

class ConvolutionMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        matcher = SubgraphsDumper::ConvolutionsMatcher();
        op_info = LayerTestsUtils::OPInfo();
    }

    SubgraphsDumper::ConvolutionsMatcher matcher;
    LayerTestsUtils::OPInfo op_info;
};


// Check that two convolutions with different input shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, ConvsSameKernelSize) {
    const auto param = std::make_shared<Parameter>(Type_t::f32, Shape({1, 3, 10, 10}));
    const auto weights = std::make_shared<Constant>(Type_t::f32, Shape({10, 3, 3, 3}), 1);
    const auto op1 = std::make_shared<v1::Convolution>(param, weights, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));

    const auto param2 = std::make_shared<Parameter>(Type_t::f32, Shape({1, 5, 20, 20}));
    const auto weights2 = std::make_shared<Constant>(Type_t::f32, Shape({10, 5, 3, 3}), 1);
    const auto op2 = std::make_shared<v1::Convolution>(param2, weights2, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));
    ASSERT_TRUE(matcher.match(op1, op2, op_info));
}

// Check that two convolutions with different input shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, ConvsDifferentKernelSize) {
    const auto param = std::make_shared<Parameter>(Type_t::f32, Shape({1, 3, 10, 10}));
    const auto weights = std::make_shared<Constant>(Type_t::f32, Shape({10, 3, 3, 5}), 1);
    const auto op1 = std::make_shared<v1::Convolution>(param, weights, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));

    const auto param2 = std::make_shared<Parameter>(Type_t::f32, Shape({1, 5, 20, 20}));
    const auto weights2 = std::make_shared<Constant>(Type_t::f32, Shape({10, 5, 3, 3}), 1);
    const auto op2 = std::make_shared<v1::Convolution>(param2, weights2, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));
    ASSERT_FALSE(matcher.match(op1, op2, op_info));
}

// Check that two group convolutions with different input shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, GroupConvsSameKernelSize) {
    const auto param = std::make_shared<Parameter>(Type_t::f32, Shape({1, 4, 10, 10}));
    const auto weights = std::make_shared<Constant>(Type_t::f32, Shape({2, 10, 2, 3, 3}), 1);
    const auto op1 = std::make_shared<v1::GroupConvolution>(param, weights, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));

    const auto param2 = std::make_shared<Parameter>(Type_t::f32, Shape({1, 6, 20, 20}));
    const auto weights2 = std::make_shared<Constant>(Type_t::f32, Shape({2, 10, 3, 3, 3}), 1);
    const auto op2 = std::make_shared<v1::GroupConvolution>(param2, weights2, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));
    ASSERT_TRUE(matcher.match(op1, op2, op_info));
}

// Check that two group convolutions with different input shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, GroupConvsDifferentKernelSize) {
    const auto param = std::make_shared<Parameter>(Type_t::f32, Shape({1, 4, 10, 10}));
    const auto weights = std::make_shared<Constant>(Type_t::f32, Shape({2, 10, 2, 3, 5}), 1);
    const auto op1 = std::make_shared<v1::GroupConvolution>(param, weights, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));

    const auto param2 = std::make_shared<Parameter>(Type_t::f32, Shape({1, 6, 20, 20}));
    const auto weights2 = std::make_shared<Constant>(Type_t::f32, Shape({2, 10, 3, 3, 3}), 1);
    const auto op2 = std::make_shared<v1::GroupConvolution>(param2, weights2, Strides(0, 0), CoordinateDiff(0, 0),
                                                       CoordinateDiff(0, 0), Strides(0, 0));
    ASSERT_FALSE(matcher.match(op1, op2, op_info));
}