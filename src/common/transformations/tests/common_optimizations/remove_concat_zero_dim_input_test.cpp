// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/remove_concat_zero_dim_input.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/init_node_info.hpp"
#include "transformations/utils/utils.hpp"
using namespace ov;
using namespace testing;

TEST_F(TransformationTestsF, RemoveConcatZeroDimInputStaticShape) {
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    int64_t axis = 1;
    {
        auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 3});
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input3}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3});
    }
}

TEST_F(TransformationTestsF, DisableRemoveConcatZeroDimInputStaticShape) {
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 3});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    int64_t axis = 1;
    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);
        ov::pass::disable_remove_concat_zerodim_input(concat);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});
    }
}

TEST_F(TransformationTestsF, DisableRemoveConcatZeroDimInputPartiallyKnowShape) {
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
    auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, 0, -1});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{-1, -1, -1});
    int64_t axis = 1;
    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);
        ov::pass::disable_remove_concat_zerodim_input(concat);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});
    }
}

TEST_F(TransformationTestsF, RemoveConcatZeroDimInputSubgraph) {
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    int64_t axis = 1;
    {
        auto in_abs = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 3});
        auto abs = std::make_shared<ov::opset8::Abs>(in_abs);
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, abs, input3}, axis);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3, in_abs});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input3}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3});
    }
}

TEST_F(TransformationTestsF, RemoveConcatZeroDimInputSubgraph2) {
    auto input1 =
        std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, ov::Dimension::dynamic(), 3});
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 2, 3});
    auto abs = std::make_shared<ov::opset8::Abs>(input1);
    int64_t axis = 1;
    {
        auto in_mul = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 3});
        auto mul = std::make_shared<ov::opset8::Multiply>(in_mul, abs);
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{mul, input3}, axis);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3, in_mul});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input3}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input3});
    }
}

TEST_F(TransformationTestsF, RemoveConcatZeroDimInputPartiallyKnowShape) {
    std::shared_ptr<ov::Model> f(nullptr), f_ref(nullptr);
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    int64_t axis = 0;
    {
        auto input2 = std::make_shared<ov::opset8::Parameter>(
            ov::element::f32,
            ov::PartialShape{0, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});
        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input3}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input3});
    }
}

TEST_F(TransformationTestsF, RemoveConcatZeroDimInputDynamicRank) {
    auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    auto input3 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
    int64_t axis = 0;
    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});
    }
}

TEST_F(TransformationTestsF, RemoveConcatZeroDimTwoInputs) {
    auto input1 = std::make_shared<ov::opset8::Parameter>(
        ov::element::f32,
        ov::PartialShape{1, ov::Dimension::dynamic(), ov::Dimension::dynamic()});
    int64_t axis = 1;
    {
        auto input2 =
            std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, ov::Dimension::dynamic()});
        auto input3 =
            std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, ov::Dimension::dynamic(), 0});
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2, input3}, axis);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2, input3});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1}, axis);
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1});
    }
}

TEST_F(TransformationTestsF, RemoveConcatZeroDimAllInputsEmpty) {
    {
        auto input1 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 1});
        auto input2 = std::make_shared<ov::opset8::Parameter>(ov::element::f32, ov::PartialShape{1, 0, 1});

        int64_t axis = -1;
        auto concat = std::make_shared<ov::opset8::Concat>(ov::OutputVector{input1, input2}, axis);

        model = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{input1, input2});

        manager.register_pass<ov::pass::RemoveConcatZeroDimInput>();
    }

    {
        auto concat =
            std::make_shared<ov::opset8::Constant>(ov::element::f32, ov::Shape{1, 0, 2}, std::vector<float>{});
        model_ref = std::make_shared<ov::Model>(ov::NodeVector{concat}, ov::ParameterVector{});
    }
}
