// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reshape_prelu.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov;
using namespace testing;
using namespace ov::pass;

TEST(TransformationTests, ReshapePReluTest1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto slope = opset1::Constant::create(element::f32, Shape{3}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto slope = opset1::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest2) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input_pshape = PartialShape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_pshape);
        auto slope = opset1::Constant::create(element::f32, Shape{3}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input_pshape = PartialShape{Dimension::dynamic(), 3, Dimension::dynamic(), Dimension::dynamic()};
        auto input = std::make_shared<opset1::Parameter>(element::f32, input_pshape);
        auto slope = opset1::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest3) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic(4));
        auto slope = opset1::Constant::create(element::f32, Shape{3}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic(4));
        auto slope = opset1::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest4) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 16, 16});
        auto slope = opset1::Constant::create(element::f32, Shape{}, {-2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
        f_ref = f;

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest5) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{3});
        auto slope = opset1::Constant::create(element::f32, Shape{3}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
        f_ref = f;

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest6) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{1, 3, 4, 4});
        auto slope = opset1::Constant::create(element::f32, Shape{4}, {-2.f, -1.f, -2.f, -1.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
        f_ref = f;

        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest7) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::u8, Shape{1, 3, 16, 16});
        auto slope = opset1::Constant::create(element::f32, Shape{3}, {-2.f, -1.f, -2.f});
        auto relaxed_prelu = std::make_shared<ov::op::TypeRelaxed<opset1::PRelu>>(
            element::TypeVector{element::f32, element::f32},
            element::TypeVector{element::f32},
            ov::op::TemporaryReplaceOutputType(input, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(slope, element::f32).get());

        f = std::make_shared<ov::Model>(NodeVector{relaxed_prelu}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::u8, Shape{1, 3, 16, 16});
        auto slope = opset1::Constant::create(element::f32, Shape{1, 3, 1, 1}, {-2.f, -1.f, -2.f});
        auto relaxed_prelu = std::make_shared<ov::op::TypeRelaxed<opset1::PRelu>>(
            element::TypeVector{element::f32, element::f32},
            element::TypeVector{element::f32},
            ov::op::TemporaryReplaceOutputType(input, element::f32).get(),
            ov::op::TemporaryReplaceOutputType(slope, element::f32).get());

        f_ref = std::make_shared<ov::Model>(NodeVector{relaxed_prelu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest8) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{4, 3});
        auto slope = opset1::Constant::create(element::f32, Shape{3}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, Shape{4, 3});
        auto slope = opset1::Constant::create(element::f32, Shape{1, 3}, {-2.f, -1.f, -2.f});
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f_ref = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ReshapePReluTest9) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic(4));
        auto slope = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic(1));
        auto prelu = std::make_shared<opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input, slope});
        pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ReshapePRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic(4));
        auto slope = std::make_shared<opset1::Parameter>(element::f32, PartialShape::dynamic(1));

        auto shape_of = std::make_shared<opset1::ShapeOf>(slope);
        auto reshape_const = opset1::Constant::create(element::i64, {4}, {1, -1, 1, 1});
        auto reshape = std::make_shared<opset1::Reshape>(slope, reshape_const, true);
        auto prelu = std::make_shared<opset1::PRelu>(input, reshape);

        f_ref = std::make_shared<ov::Model>(NodeVector{prelu}, ParameterVector{input, slope});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
