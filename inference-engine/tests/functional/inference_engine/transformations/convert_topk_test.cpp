// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/topk_ie.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, ConvertTopKToTopKIEStatic) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ngraph::pass::InjectionPass>([](std::shared_ptr<ngraph::Function> f) {
            check_rt_info(f);
        });
        manager.register_pass<ngraph::pass::ConstantFolding>();
        ASSERT_NO_THROW(manager.run_passes(f));
        ASSERT_TRUE(f->get_output_partial_shape(0).is_static()) << "Shape " << f->get_output_partial_shape(0) << " should be static";
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTopKToTopKIEDynamic1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{DYN, 20, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTopKToTopKIEDynamic2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, DYN, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, DYN, 3});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTopKToTopKIEDynamic3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 20, DYN});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {10});
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 20, DYN});
        auto k = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {10});
        //auto unsqueezed_k = std::make_shared<ngraph::opset1::Unsqueeze>(k, ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input, k, 1, ngraph::op::TopKMode::MIN,
                ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertTopKToTopKIENegative) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        f = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input, k});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        ngraph::pass::ConstantFolding().run_on_function(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{15, 20, 3});
        auto k = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto topk = std::make_shared<ngraph::opset1::TopK>(input, k, 1, "min", "value", ngraph::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        f_ref = std::make_shared<ngraph::Function>(ngraph::OutputVector{topk->output(0)}, ngraph::ParameterVector{input, k});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}