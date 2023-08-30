// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <legacy/ngraph_ops/topk_ie.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_topk_to_topk_ie.hpp>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertTopKToTopKIEStatic) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15, 20, 3});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {10});
        auto topk = std::make_shared<ov::opset1::TopK>(input, k, 1, "min", "value", ov::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});

        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ov::pass::InjectionPass>([](std::shared_ptr<ov::Model> f) {
            check_rt_info(f);
        });
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15, 20, 3});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {10});
        // auto unsqueezed_k = std::make_shared<ov::opset1::Unsqueeze>(k,
        // ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input,
                                                         k,
                                                         1,
                                                         ngraph::op::TopKMode::MIN,
                                                         ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIEDynamic1) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{DYN, 20, 3});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {10});
        auto topk = std::make_shared<ov::opset1::TopK>(input, k, 1, "min", "value", ov::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{DYN, 20, 3});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {10});
        // auto unsqueezed_k = std::make_shared<ov::opset1::Unsqueeze>(k,
        // ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input,
                                                         k,
                                                         1,
                                                         ngraph::op::TopKMode::MIN,
                                                         ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIEDynamic2) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, DYN, 3});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {10});
        auto topk = std::make_shared<ov::opset1::TopK>(input, k, 1, "min", "value", ov::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, DYN, 3});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {10});
        // auto unsqueezed_k = std::make_shared<ov::opset1::Unsqueeze>(k,
        // ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input,
                                                         k,
                                                         1,
                                                         ngraph::op::TopKMode::MIN,
                                                         ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIEDynamic3) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 20, DYN});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {10});
        auto topk = std::make_shared<ov::opset1::TopK>(input, k, 1, "min", "value", ov::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape{1, 20, DYN});
        auto k = ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {10});
        // auto unsqueezed_k = std::make_shared<ov::opset1::Unsqueeze>(k,
        // ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {0}));
        auto topk = std::make_shared<ngraph::op::TopKIE>(input,
                                                         k,
                                                         1,
                                                         ngraph::op::TopKMode::MIN,
                                                         ngraph::op::TopKSortType::SORT_VALUES);
        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertTopKToTopKIENegative) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15, 20, 3});
        auto k = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape::dynamic());
        auto topk = std::make_shared<ov::opset1::TopK>(input, k, 1, "min", "value", ov::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        model = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input, k});
        manager.register_pass<ngraph::pass::ConvertTopKToTopKIEMatcher>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{15, 20, 3});
        auto k = std::make_shared<ov::opset1::Parameter>(ov::element::i32, ov::PartialShape::dynamic());
        auto topk = std::make_shared<ov::opset1::TopK>(input, k, 1, "min", "value", ov::element::i32);
        // due to the 'compare_functions' limitation we will check only one output
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{topk->output(0)}, ov::ParameterVector{input, k});
    }
}
