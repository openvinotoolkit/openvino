// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/common_optimizations/disable_shapeof_constant_folding.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, DisableShapeOfConstantFolding) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto abs = std::make_shared<opset6::Abs>(shape_of);
        auto reshape = std::make_shared<opset6::Reshape>(data, abs, false);
        model = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto abs = std::make_shared<opset6::Abs>(shape_of);
        auto reshape = std::make_shared<opset6::Reshape>(data, abs, false);
        model_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeOfShapeOfConstantFolding) {
    std::shared_ptr<Model> f, f_ref;
    {
        auto data = std::make_shared<opset6::Parameter>(element::i64, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto reshape = std::make_shared<opset6::Reshape>(data, shape_of, false);
        auto rank = std::make_shared<opset6::ShapeOf>(shape_of);
        auto mul = std::make_shared<opset6::Multiply>(reshape, rank);
        model = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::i64, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto reshape = std::make_shared<opset6::Reshape>(data, shape_of, false);
        auto mul = std::make_shared<opset6::Multiply>(reshape, opset6::Constant::create(element::i64, Shape{1}, {4}));
        model_ref = std::make_shared<Model>(NodeVector{mul}, ParameterVector{data});
    }
}
