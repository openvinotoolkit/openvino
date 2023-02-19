// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/pass/manager.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset10.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <transformations/common_optimizations/disable_shapeof_constant_folding.hpp>
#include <transformations/common_optimizations/enable_shapeof_constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov;

TEST_F(TransformationTestsF, EnableShapeOfV0ConstantFolding) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset1::ShapeOf>(data);
        auto abs = std::make_shared<opset10::Abs>(shape_of);
        auto reshape = std::make_shared<opset10::Reshape>(data, abs, false);
        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
        manager.register_pass<ov::pass::EnableShapeOfConstantFolding>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto reshape =
            std::make_shared<opset10::Reshape>(data,
                                               opset10::Constant::create(element::i64, Shape{4}, {1, 4, 10, 10}),
                                               false);
        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, EnableShapeOfV3ConstantFolding) {
    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset10::ShapeOf>(data);
        auto abs = std::make_shared<opset10::Abs>(shape_of);
        auto reshape = std::make_shared<opset10::Reshape>(data, abs, false);
        function = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<ov::pass::DisableShapeOfConstantFolding>();
        manager.register_pass<ov::pass::EnableShapeOfConstantFolding>();
        manager.register_pass<ov::pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset10::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto reshape =
            std::make_shared<opset10::Reshape>(data,
                                               opset10::Constant::create(element::i64, Shape{4}, {1, 4, 10, 10}),
                                               false);
        function_ref = std::make_shared<Model>(NodeVector{reshape}, ParameterVector{data});
    }
}
