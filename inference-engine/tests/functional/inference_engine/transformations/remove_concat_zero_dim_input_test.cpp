// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/function.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/common_optimizations/nop_elimination.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, RemoveConcatZeroDimInputStaticShape) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    int64_t axis = 1;
    {
        auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 3});
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NopElimination>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input3}, axis);
        f_ref = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveConcatZeroDimInputSubgraph) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    int64_t axis = 1;
    {
        auto in_abs = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 3});
        auto abs = std::make_shared<ov::opset8::Abs>(in_abs);
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, abs, input3}, axis);

        f = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3, in_abs});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NopElimination>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input3}, axis);
        f_ref = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveConcatZeroDimInputSubgraph2) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, ov::Dimension::dynamic(), 3});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto abs = std::make_shared<ov::opset8::Abs>(input1);
    int64_t axis = 1;
    {
        auto in_mul = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 3});
        auto mul = std::make_shared<ov::opset8::Multiply>(in_mul, abs);
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{mul, input3}, axis);

        f = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3, in_mul});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NopElimination>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input3}, axis);
        f_ref = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveConcatZeroDimInputPartiallyKnowShape) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    int64_t axis = 0;
    {
        auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32,
            ov::PartialShape{0, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NopElimination>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        f->validate_nodes_and_infer_types();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input3}, axis);
        f_ref = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveConcatZeroDimInputDynamicRank) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    int64_t axis = 0;
    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NopElimination>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);
        f_ref = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});
    }
    // the pass should be not applied
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveConcatZeroDimTwoInputs) {
    std::shared_ptr<ov::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32,
        ov::PartialShape{1, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    int64_t axis = 1;
    {
        auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32,
            ov::PartialShape{1, 0, ov::Dimension::dynamic()});
        auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32,
            ov::PartialShape{1, ov::Dimension::dynamic(), 0});
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        ov::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::NopElimination>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
        f->validate_nodes_and_infer_types();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1}, axis);
        f_ref = std::make_shared<ov::Function>(ov::NodeVector{concat}, ov::ParameterVector{input1});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
