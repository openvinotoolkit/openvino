// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/clamp_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset5.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, ClampFusion) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{1}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{1}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_const);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        model = std::make_shared<Model>(NodeVector{min}, ParameterVector{data});

        manager.register_pass<ov::pass::ClampFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto clamp = std::make_shared<opset5::Clamp>(data, 0.1, 5);
        model_ref = std::make_shared<Model>(NodeVector{clamp}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ClampFusionScalars) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_const);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        model = std::make_shared<Model>(NodeVector{min}, ParameterVector{data});

        manager.register_pass<ov::pass::ClampFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto clamp = std::make_shared<opset5::Clamp>(data, 0.1, 5);
        model_ref = std::make_shared<Model>(NodeVector{clamp}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ClampFusionNonConstMin) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_val = std::make_shared<opset5::Parameter>(element::f32, Shape{});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_val);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        model = std::make_shared<Model>(NodeVector{min}, ParameterVector{data, min_val});

        manager.register_pass<ov::pass::ClampFusion>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_val = std::make_shared<opset5::Parameter>(element::f32, Shape{});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_val);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        model_ref = std::make_shared<Model>(NodeVector{min}, ParameterVector{data, min_val});
    }
}

TEST_F(TransformationTestsF, ClampFusionMinMax) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{1}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{1}, {5});
        auto min = std::make_shared<opset5::Minimum>(data, max_const);
        auto max = std::make_shared<opset5::Maximum>(min, min_const);

        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data});

        manager.register_pass<ov::pass::ClampFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto clamp = std::make_shared<opset5::Clamp>(data, 0.1, 5);
        model_ref = std::make_shared<Model>(NodeVector{clamp}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ClampFusionMinMaxScalars) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto min = std::make_shared<opset5::Minimum>(data, max_const);
        auto max = std::make_shared<opset5::Maximum>(min, min_const);
        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data});

        manager.register_pass<ov::pass::ClampFusion>();
    }

    {
        auto data = std::make_shared<ov::op::v0::Parameter>(element::f32, Shape{2, 2});
        auto clamp = std::make_shared<opset5::Clamp>(data, 0.1, 5);
        model_ref = std::make_shared<Model>(NodeVector{clamp}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ClampFusionMinMaxNonConstMax) {
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto max_val = std::make_shared<opset5::Parameter>(element::f32, Shape{});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto min = std::make_shared<opset5::Minimum>(data, max_const);
        auto max = std::make_shared<opset5::Maximum>(min, max_val);
        model = std::make_shared<Model>(NodeVector{max}, ParameterVector{data, max_val});

        manager.register_pass<ov::pass::ClampFusion>();
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_val = std::make_shared<opset5::Parameter>(element::f32, Shape{});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto min = std::make_shared<opset5::Minimum>(data, max_const);
        auto max = std::make_shared<opset5::Maximum>(min, min_val);
        model_ref = std::make_shared<Model>(NodeVector{max}, ParameterVector{data, min_val});
    }
}
