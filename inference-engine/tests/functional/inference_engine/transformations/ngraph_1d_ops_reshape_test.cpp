// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include "common_test_utils/test_common.hpp"
#include <string>
#include <sstream>
#include <fstream>
#include <memory>
#include <map>

#include <ngraph/function.hpp>
#include <ngraph/op/constant.hpp>
#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <transformations/convert_opset1_to_legacy/reshape_1d_ops.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, ConvReshapeTest1) {
    auto input = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3, 64}, {1});
    auto w = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6, 3, 3/*OIW*/}, {1});

    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        ngraph::Strides strides{1}, dilations{1};
        ngraph::CoordinateDiff pads_begin{1}, pads_end{2};
        ngraph::Shape output_shape{1, 6, 62};
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input, w, strides, pads_begin, pads_end, dilations, output_shape, 1);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::Reshape1DOps().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    std::vector<size_t> ref_shape{1, 6, 1, 62};
    ngraph::Strides ref_strides{1, 1};
    ngraph::CoordinateDiff ref_pads_begin{0, 1}, ref_pads_end{0, 2};
    for (auto op : f->get_ops()) {
        if (auto conv = ngraph::as_type_ptr<ngraph::op::ConvolutionIE>(op)) {
            ASSERT_EQ(conv->get_shape(), ref_shape);
            ASSERT_EQ(conv->get_strides(), ref_strides);
            ASSERT_EQ(conv->get_dilations(), ref_strides);
            ASSERT_EQ(conv->get_pads_begin(), ref_pads_begin);
            ASSERT_EQ(conv->get_pads_end(), ref_pads_end);
        }
    }
}

TEST(TransformationTests, ConvBiasReshapeTest1) {
    auto input = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{1, 3, 64}, {1});
    auto w = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6, 3, 3/*OIW*/}, {1});
    auto b = ngraph::op::Constant::create(ngraph::element::f32, ngraph::Shape{6}, {1});

    std::shared_ptr<ngraph::Function> f(nullptr);
    {
        ngraph::Strides strides{1}, dilations{1};
        ngraph::CoordinateDiff pads_begin{1}, pads_end{2};
        ngraph::Shape output_shape{1, 6, 62};
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input, w, b, strides, pads_begin, pads_end, dilations, output_shape, 1);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{conv}, ngraph::ParameterVector{});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::Reshape1DOps().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    std::vector<size_t> ref_shape{1, 6, 1, 62};
    ngraph::Strides ref_strides{1, 1};
    ngraph::CoordinateDiff ref_pads_begin{0, 1}, ref_pads_end{0, 2};
    for (auto op : f->get_ops()) {
        if (auto conv = ngraph::as_type_ptr<ngraph::op::ConvolutionIE>(op)) {
            ASSERT_EQ(conv->get_shape(), ref_shape);
            ASSERT_EQ(conv->get_strides(), ref_strides);
            ASSERT_EQ(conv->get_dilations(), ref_strides);
            ASSERT_EQ(conv->get_pads_begin(), ref_pads_begin);
            ASSERT_EQ(conv->get_pads_end(), ref_pads_end);
        }
    }
}

TEST(TransformationTests, MaxPoolReshapeTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64});

        ngraph::Strides strides{1};
        ngraph::Shape pads_begin{0}, pads_end{0}, kernel{3};
        auto pool = std::make_shared<ngraph::opset1::MaxPool>(input, strides, pads_begin, pads_end, kernel, ngraph::op::RoundingType::FLOOR);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{pool}, ngraph::ParameterVector{input});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::Reshape1DOps().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64});

        auto reshape_begin = std::make_shared<opset1::Reshape>(input, opset1::Constant::create(element::i64, Shape{4}, {1, 3, 1, 64}), true);

        ngraph::Strides strides{1, 1};
        ngraph::Shape pads_begin{0, 0}, pads_end{0, 0}, kernel{1, 3};
        auto pool = std::make_shared<ngraph::opset1::MaxPool>(reshape_begin, strides, pads_begin, pads_end, kernel, ngraph::op::RoundingType::FLOOR);

        auto reshape_end = std::make_shared<opset1::Reshape>(pool, opset1::Constant::create(element::i64, Shape{3}, {1, 3, 62}), true);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_end}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, AvgPoolReshapeTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64});

        ngraph::Strides strides{1};
        ngraph::Shape pads_begin{0}, pads_end{0}, kernel{3};
        auto pool = std::make_shared<ngraph::opset1::AvgPool>(input, strides, pads_begin, pads_end, kernel, false, ngraph::op::RoundingType::FLOOR);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{pool}, ngraph::ParameterVector{input});
        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::Reshape1DOps().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 64});

        auto reshape_begin = std::make_shared<opset1::Reshape>(input, opset1::Constant::create(element::i64, Shape{4}, {1, 3, 1, 64}), true);

        ngraph::Strides strides{1, 1};
        ngraph::Shape pads_begin{0, 0}, pads_end{0, 0}, kernel{1, 3};
        auto pool = std::make_shared<ngraph::opset1::AvgPool>(reshape_begin, strides, pads_begin, pads_end, kernel, false, ngraph::op::RoundingType::FLOOR);

        auto reshape_end = std::make_shared<opset1::Reshape>(pool, opset1::Constant::create(element::i64, Shape{3}, {1, 3, 62}), true);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{reshape_end}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}