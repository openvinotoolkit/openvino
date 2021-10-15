// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/remove_concat_zero_dim_input.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST(TransformationTests, RemoveCancatZeroDimInputStaticShape) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 2, 3});
    auto input3 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 2, 3});
    int64_t axis = 1;
    {
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 0, 3});
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input2, input3});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::RemoveConcatZeroDimInput>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input3}, axis);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveCancatZeroDimInputSubgraph) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 2, 3});
    auto input3 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 2, 3});
    int64_t axis = 1;
    {
        auto in_abs = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 0, 3});
        auto abs = std::make_shared<ngraph::opset8::Abs>(in_abs);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, abs, input3}, axis);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input3, in_abs});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::RemoveConcatZeroDimInput>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input3}, axis);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveCancatZeroDimInputSubgraph2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, ngraph::Dimension::dynamic(), 3});
    auto input3 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 2, 3});
    auto abs = std::make_shared<ngraph::opset8::Abs>(input1);
    int64_t axis = 1;
    {
        auto in_mul = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape{1, 0, 3});
        auto mul = std::make_shared<ngraph::opset8::Multiply>(in_mul, abs);
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{mul, input3}, axis);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input3, in_mul});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::RemoveConcatZeroDimInput>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input3}, axis);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveCancatZeroDimInputPartiallyKnowShape) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto input3 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    int64_t axis = 0;
    {
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32,
            ngraph::PartialShape{0, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()});
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input2, input3});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::RemoveConcatZeroDimInput>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input3}, axis);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input3});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveCancatZeroDimInputDynamicRank) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    auto input3 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
    int64_t axis = 0;
    {
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input2, input3});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::RemoveConcatZeroDimInput>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2, input3}, axis);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input2, input3});
    }
    // the pass should be not applied
    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}

TEST(TransformationTests, RemoveCancatZeroDimTwoInputs) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32,
        ngraph::PartialShape{1, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic()});
    int64_t axis = 1;
    {
        auto input2 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32,
            ngraph::PartialShape{1, 0, ngraph::Dimension::dynamic()});
        auto input3 = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32,
            ngraph::PartialShape{1, ngraph::Dimension::dynamic(), 0});
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1, input2, input3}, axis);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1, input2, input3});

        ngraph::pass::Manager manager;
        manager.register_pass<ngraph::pass::InitNodeInfo>();
        manager.register_pass<ngraph::pass::RemoveConcatZeroDimInput>();
        manager.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto concat = std::make_shared<ngraph::opset8::Concat>(ngraph::OutputVector{input1}, axis);
        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{concat}, ngraph::ParameterVector{input1});
    }

    const auto fc = FunctionsComparator::with_default().enable(FunctionsComparator::ATTRIBUTES);
    const auto res = fc.compare(f, f_ref);
    ASSERT_TRUE(res.valid) << res.message;
}
