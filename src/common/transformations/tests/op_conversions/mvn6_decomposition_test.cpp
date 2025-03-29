// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/mvn6_decomposition.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, MVN6Decomposition_No_Variance) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{2}, {2, 3});
        auto mvn = std::make_shared<opset6::MVN>(data, axes_const, false, 1e-5f, op::MVNEpsMode::INSIDE_SQRT);

        model = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});

        manager.register_pass<ov::pass::MVN6Decomposition>();
    }

    {
        auto input0 = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{2}, {2, 3});
        auto mean = std::make_shared<opset6::ReduceMean>(input0, axes_const, true);
        auto mean_normalization = std::make_shared<opset6::Subtract>(input0, mean);

        model_ref = std::make_shared<ov::Model>(NodeVector{mean_normalization}, ParameterVector{input0});
    }
}

TEST_F(TransformationTestsF, MVN6Decomposition_Inside_Sqrt) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{2}, {2, 3});
        auto mvn = std::make_shared<opset6::MVN>(data, axes_const, true, 1e-5f, op::MVNEpsMode::INSIDE_SQRT);

        model = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});

        manager.register_pass<ov::pass::MVN6Decomposition>();
    }

    {
        auto input0 = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{2}, {2, 3});
        auto mean = std::make_shared<opset6::ReduceMean>(input0, axes_const, true);
        auto mean_normalization = std::make_shared<opset6::Subtract>(input0, mean);

        auto sqr_const = opset6::Constant::create(element::f32, Shape{1}, {2});
        auto sqr = std::make_shared<opset6::Power>(mean_normalization, sqr_const);
        auto mean2 = std::make_shared<opset6::ReduceMean>(sqr, axes_const, true);

        auto eps_node = opset6::Constant::create(element::f32, Shape{1}, {1e-5});

        auto eps_add = std::make_shared<opset6::Add>(mean2, eps_node);
        auto sqrt = std::make_shared<opset6::Sqrt>(eps_add);
        auto div = std::make_shared<opset6::Divide>(mean_normalization, sqrt);

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input0});
    }
}

TEST_F(TransformationTestsF, MVN6Decomposition_Outside_Sqrt) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{2}, {2, 3});
        auto mvn = std::make_shared<opset6::MVN>(data, axes_const, true, 1e-5f, op::MVNEpsMode::OUTSIDE_SQRT);

        model = std::make_shared<ov::Model>(NodeVector{mvn}, ParameterVector{data});

        manager.register_pass<ov::pass::MVN6Decomposition>();
    }

    {
        auto input0 = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3, 4});
        auto axes_const = opset6::Constant::create(element::i64, Shape{2}, {2, 3});
        auto mean = std::make_shared<opset6::ReduceMean>(input0, axes_const, true);
        auto mean_normalization = std::make_shared<opset6::Subtract>(input0, mean);

        auto sqr_const = opset6::Constant::create(element::f32, Shape{1}, {2});
        auto sqr = std::make_shared<opset6::Power>(mean_normalization, sqr_const);
        auto mean2 = std::make_shared<opset6::ReduceMean>(sqr, axes_const, true);

        auto eps_node = opset6::Constant::create(element::f32, Shape{1}, {1e-5});

        auto sqrt = std::make_shared<opset6::Sqrt>(mean2);
        auto eps_add = std::make_shared<opset6::Add>(sqrt, eps_node);
        auto div = std::make_shared<opset6::Divide>(mean_normalization, eps_add);

        model_ref = std::make_shared<ov::Model>(NodeVector{div}, ParameterVector{input0});
    }
}
