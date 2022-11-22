// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <legacy/ngraph_ops/topk_ie.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertTopKToTopKIEStatic) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIEDynamic1) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIEDynamic2) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, DYN, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, DYN, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIEDynamic3) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 20, DYN});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 20, DYN});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIENegative) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        function = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input, k});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ngraph::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        function_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input, k});
    }
}