// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/convert_opset3_to_opset2/convert_broadcast3.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "ngraph_test_utils.hpp"

using namespace testing;

// Broadcast-3 is converted directly to Broadcast-1 for modes NUMPY, NONE and PDPD
TEST(TransformationTests, ConvertBroadcast3WithNumpyModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, ngraph::op::BroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertBroadcast3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input1, target_shape, ngraph::op::AutoBroadcastType::NUMPY);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}

TEST(TransformationTests, ConvertBroadcast3WithPDPDModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, ngraph::op::BroadcastType::PDPD);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertBroadcast3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto target_shape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input1, target_shape, ngraph::op::AutoBroadcastType::PDPD);
        broadcast->set_friendly_name("broadcast");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}

TEST(TransformationTests, ConvertBroadcast3WithExplicitModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 5, 2});
        auto brodcast_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{0, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, brodcast_axis, ngraph::op::BroadcastType::EXPLICIT);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertBroadcast3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 5, 2});
        auto brodcast_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{0, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 2});
        auto broadcast = std::make_shared<ngraph::opset1::Broadcast>(input1, target_shape, brodcast_axis, ngraph::op::AutoBroadcastType::EXPLICIT);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto broadcast_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = broadcast_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(broadcast_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}

// Broadcast-3 with mode BIDIRECTIONAL is converted to Multiply with constant with 1s of the corresponding type
TEST(TransformationTests, ConvertBroadcast3WithBidirectionalModeToBroadcast1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 2});
        auto target_shape = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 1});
        auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(input1, target_shape, ngraph::op::BroadcastType::BIDIRECTIONAL);
        broadcast->set_friendly_name("broadcast");

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input1});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertBroadcast3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 1, 2});
        auto target_shape = std::make_shared<ngraph::opset1::Constant>(ngraph::element::i64, ngraph::Shape{3}, std::vector<int64_t>{3, 5, 1});
        auto constant_one = std::make_shared<ngraph::opset1::Constant>(input->get_output_element_type(0), ngraph::Shape({1}), std::vector<int>{1});
        auto broadcast_ones = std::make_shared<ngraph::opset1::Broadcast>(constant_one, target_shape, ngraph::op::AutoBroadcastType::NUMPY);
        auto multiply = std::make_shared<ngraph::opset1::Multiply>(input, broadcast_ones);
        multiply->set_friendly_name("broadcast");

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{multiply}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto result_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    auto crop_node = result_node->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(result_node->get_friendly_name() == "broadcast") << "Transformation ConvertBroadcast3 should keep output names.\n";
}
