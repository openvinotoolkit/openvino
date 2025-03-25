// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/smart_reshape/shape_of_const_folding.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace ov;

TEST_F(TransformationTestsF, ShapeOfConstFolding) {
    {
        auto input = std::make_shared<opset10::Constant>(element::f32, Shape{10, 20}, 1);
        auto shape_of = std::make_shared<opset10::ShapeOf>(input);
        model = std::make_shared<Model>(shape_of, ParameterVector{});
    }

    manager.register_pass<ov::pass::ShapeOfConstFolding>();

    {
        auto input = std::make_shared<opset10::Constant>(element::i64, Shape{2}, std::vector<int64_t>{10, 20});
        model_ref = std::make_shared<Model>(input, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, v0ShapeOfConstFolding) {
    {
        auto input = std::make_shared<opset10::Constant>(element::f32, Shape{10, 20}, 1);
        auto shape_of = std::make_shared<op::v0::ShapeOf>(input);
        model = std::make_shared<Model>(shape_of, ParameterVector{});
    }

    manager.register_pass<ov::pass::ShapeOfConstFolding>();

    {
        auto input = std::make_shared<opset10::Constant>(element::i64, Shape{2}, std::vector<int64_t>{10, 20});
        model_ref = std::make_shared<Model>(input, ParameterVector{});
    }
}

TEST_F(TransformationTestsF, ShapeOfConstFoldingNegative) {
    {
        auto input = std::make_shared<opset10::Parameter>(element::f32, Shape{10, 20});
        auto shape_of = std::make_shared<opset10::ShapeOf>(input);
        model = std::make_shared<Model>(shape_of, ParameterVector{input});
    }

    manager.register_pass<ov::pass::ShapeOfConstFolding>();
}
