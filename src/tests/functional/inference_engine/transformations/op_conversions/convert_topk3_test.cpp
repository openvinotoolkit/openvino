// Copyright (C) 2018-2022 Intel Corporation
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
#include <transformations/op_conversions/convert_topk3.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

// check that the first output from the TopK-3 with I32 output indices is equal to the TopK-1 first output
TEST_F(TransformationTestsF, ConvertTopK3I32Output0) {
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        topk->set_friendly_name("topk");

        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }
}

// check that the second output from the TopK-3 with I32 output indices is equal to the TopK-1 second output
TEST_F(TransformationTestsF, ConvertTopK3I32Output1) {
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i32);

        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(1)}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);

        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(1)}, ngraph::ParameterVector{input});
    }
}

// check that the first output from the TopK-3 with I64 output indices is equal to the TopK-1 first output
TEST_F(TransformationTestsF, ConvertTopK3I64Output0) {
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i64);

        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);

        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }
}

// check that the second output from the TopK-3 with I64 output indices is equal to the TopK-1 second output converted to I64
TEST_F(TransformationTestsF, ConvertTopK3I64Output1) {
    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset3::TopK>(input, k, 1, "min", "value", ngraph::element::i64);

        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(1)}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopK3>();
    }

    {
        auto input = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset3::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset2::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        auto convert = std::make_shared<ngraph::opset2::Convert>(topk->output(1), ngraph::element::i64);

        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{convert}, ngraph::ParameterVector{input});
    }
}
