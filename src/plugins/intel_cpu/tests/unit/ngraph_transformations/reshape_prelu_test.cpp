// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_transformations/reshape_prelu.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST(TransformationTests, ReshapePReluTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 3, 1, 1 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input_pshape = ngraph::PartialShape{ ngraph::Dimension::dynamic(), 3, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic() };
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_pshape);
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input_pshape = ngraph::PartialShape{ ngraph::Dimension::dynamic(), 3, ngraph::Dimension::dynamic(), ngraph::Dimension::dynamic() };
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, input_pshape);
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 3, 1, 1 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 3, 1, 1 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        f_ref = f;

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 3 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        f_ref = f;

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest6) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 4, 4 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 4 }, { -2.f, -1.f, -2.f, -1.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        f_ref = f;

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest7) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto relaxed_prelu = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::PRelu>>(
            ngraph::element::TypeVector{ ngraph::element::f32, ngraph::element::f32 },
            ngraph::element::TypeVector{ ngraph::element::f32 },
            ngraph::op::TemporaryReplaceOutputType(input, ngraph::element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(slope, ngraph::element::f32).get());

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ relaxed_prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 3, 1, 1 }, { -2.f, -1.f, -2.f });
        auto relaxed_prelu = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::PRelu>>(
            ngraph::element::TypeVector{ ngraph::element::f32, ngraph::element::f32 },
            ngraph::element::TypeVector{ ngraph::element::f32 },
            ngraph::op::TemporaryReplaceOutputType(input, ngraph::element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(slope, ngraph::element::f32).get());

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ relaxed_prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest8) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 4, 3 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 4, 3 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest9) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto slope = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input, slope });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto slope = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));

        auto shape_of = std::make_shared<ngraph::opset1::ShapeOf>(slope);
        auto reshape_const = ngraph::opset1::Constant::create(ngraph::element::i64, { 4 }, { 1, -1, 1, 1 });
        auto reshape = std::make_shared<ngraph::opset1::Reshape>(slope, reshape_const, true);
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, reshape);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input, slope });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
