// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/convert_opset3_to_opset2/convert_topk3.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

// check that the first output from the TopK-3 with I32 output indices is equal to the TopK-1 first output
TEST(TransformationTests, ConvertTopK3I32Output0) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertTopK3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto topk_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(topk_node->get_friendly_name() == "topk") << "Transformation ConvertTopK3 should keep output names.\n";
}

// check that the second output from the TopK-3 with I32 output indices is equal to the TopK-1 second output
TEST(TransformationTests, ConvertTopK3I32Output1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(1)}, ngraph::ParameterVector{input});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertTopK3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(1)}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto topk_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(topk_node->get_friendly_name() == "topk") << "Transformation ConvertTopK3 should keep output names.\n";
}

// check that the first output from the TopK-3 with I64 output indices is equal to the TopK-1 first output
TEST(TransformationTests, ConvertTopK3I64Output0) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i64);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertTopK3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto topk_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(topk_node->get_friendly_name() == "topk") << "Transformation ConvertTopK3 should keep output names.\n";
}

// check that the second output from the TopK-3 with I64 output indices is equal to the TopK-1 second output converted to I64
TEST(TransformationTests, DISABLED_ConvertTopK3I64Output1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i64);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(1)}, ngraph::ParameterVector{input});

        ngraph::pass::InitNodeInfo().run_on_function(f);
        ngraph::pass::ConvertTopK3().run_on_function(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        auto convert = std::make_shared<ngraph::opset2::Convert>(topk->output(1), topk->get_index_element_type());
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{convert}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;

    auto result_node_of_converted_f = f->get_output_op(0);
    auto convert_node = result_node_of_converted_f->input(0).get_source_output().get_node_shared_ptr();
    ASSERT_TRUE(convert_node->get_friendly_name() == "topk.1") << "Transformation ConvertTopK3 should keep output names.\n";
}
