// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <transformations/cpu_opset/common/pass/convert_to_leaky_relu.hpp>
#include <transformations/cpu_opset/common/op/leaky_relu.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

TEST(TransformationTests, ConvertToLeakyReluTest1) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, { -2.f });
        auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest2) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, { -2.f });
        auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest3) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, { -2.f });
        auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest4) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::Shape{ 1, 3, 16, 16 });
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, { -2.f });
        auto relaxed_prelu = std::make_shared<ov::op::TypeRelaxed<ov::opset1::PRelu>>(
            ov::element::TypeVector{ ov::element::f32, ov::element::f32 },
            ov::element::TypeVector{ ov::element::f32 },
            ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(slope, ov::element::f32).get());

        f = std::make_shared<ov::Model>(ov::NodeVector{ relaxed_prelu }, ov::ParameterVector{ input });
        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::Shape{ 1, 3, 16, 16 });
        auto prelu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);

        f_ref = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ConvertToLeakyReluTest5) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{ 1, 3, 16, 16 });
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{ 3 }, { -2.f, -1.f, -2.f });
        auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);

        f = std::make_shared<ov::Model>(ov::NodeVector{ prelu }, ov::ParameterVector{ input });
        f_ref = f;

        ov::pass::Manager m;
        m.register_pass<ov::pass::InitNodeInfo>();
        m.register_pass<ConvertToLeakyRelu>();
        m.run_passes(f);
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
