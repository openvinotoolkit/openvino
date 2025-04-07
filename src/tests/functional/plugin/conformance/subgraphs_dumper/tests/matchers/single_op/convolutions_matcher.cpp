// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "matchers/single_op/convolutions.hpp"
#include "openvino/op/ops.hpp"
#include "base_test.hpp"

namespace {

using namespace ov::tools::subgraph_dumper;

class ConvolutionMatcherTest : public SubgraphsDumperBaseTest {
protected:
    void SetUp() override {
        SubgraphsDumperBaseTest::SetUp();
        matcher = ConvolutionsMatcher();
    }

    ConvolutionsMatcher matcher;
};


// Check that two convolutions with different input ov::Shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, ConvsSameKernelSize) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 3, 10, 10}));
    const auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({10, 3, 3, 3}), 1);
    const auto op1 = std::make_shared<ov::op::v1::Convolution>(param, weights, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));

    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 5, 20, 20}));
    const auto weights2 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({10, 5, 3, 3}), 1);
    const auto op2 = std::make_shared<ov::op::v1::Convolution>(param2, weights2, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));
    ASSERT_TRUE(matcher.match(op1, op2));
}

// Check that two convolutions with different input ov::Shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, ConvsDifferentKernelSize) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 3, 10, 10}));
    const auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({10, 3, 3, 5}), 1);
    const auto op1 = std::make_shared<ov::op::v1::Convolution>(param, weights, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));

    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 5, 20, 20}));
    const auto weights2 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({10, 5, 3, 3}), 1);
    const auto op2 = std::make_shared<ov::op::v1::Convolution>(param2, weights2, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));
    ASSERT_FALSE(matcher.match(op1, op2));
}

// Check that two group convolutions with different input ov::Shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, GroupConvsSameKernelSize) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 4, 10, 10}));
    const auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 10, 2, 3, 3}), 1);
    const auto op1 = std::make_shared<ov::op::v1::GroupConvolution>(param, weights, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));

    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 6, 20, 20}));
    const auto weights2 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 10, 3, 3, 3}), 1);
    const auto op2 = std::make_shared<ov::op::v1::GroupConvolution>(param2, weights2, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));
    ASSERT_TRUE(matcher.match(op1, op2));
}

// Check that two group convolutions with different input ov::Shapes but same kernel size are match each other
TEST_F(ConvolutionMatcherTest, GroupConvsDifferentKernelSize) {
    const auto param = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 4, 10, 10}));
    const auto weights = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 10, 2, 3, 5}), 1);
    const auto op1 = std::make_shared<ov::op::v1::GroupConvolution>(param, weights, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));

    const auto param2 = std::make_shared<ov::op::v0::Parameter>(ov::element::Type_t::f32, ov::Shape({1, 6, 20, 20}));
    const auto weights2 = std::make_shared<ov::op::v0::Constant>(ov::element::Type_t::f32, ov::Shape({2, 10, 3, 3, 3}), 1);
    const auto op2 = std::make_shared<ov::op::v1::GroupConvolution>(param2, weights2, ov::Strides(0, 0), ov::CoordinateDiff(0, 0),
                                                       ov::CoordinateDiff(0, 0), ov::Strides(0, 0));
    ASSERT_FALSE(matcher.match(op1, op2));
}

}  // namespace
