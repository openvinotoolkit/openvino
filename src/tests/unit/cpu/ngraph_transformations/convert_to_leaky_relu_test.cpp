// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph_transformations/convert_to_leaky_relu.hpp>
#include <ngraph_transformations/op/leaky_relu.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/pass/manager.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST(TransformationTests, ConvertToLeakyReluTest1) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ngraph::element::f32);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest2) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(4));
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ngraph::element::f32);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest3) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ngraph::element::f32);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest4) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, { -2.f });
        auto relaxed_prelu = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::PRelu>>(
            ngraph::element::TypeVector{ ngraph::element::f32, ngraph::element::f32 },
            ngraph::element::TypeVector{ ngraph::element::f32 },
            ngraph::op::TemporaryReplaceOutputType(input, ngraph::element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(slope, ngraph::element::f32).get());

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ relaxed_prelu }, ngraph::ParameterVector{ input });
        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::u8, ngraph::Shape{ 1, 3, 16, 16 });
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ngraph::element::f32);

        f_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest5) {
    std::shared_ptr<ngraph::Function> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{ 1, 3, 16, 16 });
        auto slope = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ngraph::opset1::PRelu>(input, slope);

        f = std::make_shared<ngraph::Function>(ngraph::NodeVector{ prelu }, ngraph::ParameterVector{ input });
        f_ref = f;

        ngraph::pass::Manager m;
        m.register_pass<ngraph::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
