// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/glu_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/glu.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov::pass;

TEST_F(TransformationTestsF, GLUFusionTest1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, 1, 6});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(1));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<GLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, 1, 6});
        auto swiglu = std::make_shared<ov::op::internal::GLU>(input,
                                                              axis,
                                                              split_lenghts,
                                                              ov::op::internal::GLU::GluType::Swish,
                                                              0,
                                                              ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, GLUFusionTest2) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 6});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(1));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<GLUFusion>();
    }
}

TEST_F(TransformationTestsF, GLUFusionTest3) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 6});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(1));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<GLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 6});
        auto swiglu = std::make_shared<ov::op::internal::GLU>(input,
                                                              axis,
                                                              split_lenghts,
                                                              ov::op::internal::GLU::GluType::Swish,
                                                              0,
                                                              ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, GLUFusionTest3ReverseOrder) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 6});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(variadic_split->output(1), swish);

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<GLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 6});
        auto swiglu = std::make_shared<ov::op::internal::GLU>(input,
                                                              axis,
                                                              split_lenghts,
                                                              ov::op::internal::GLU::GluType::Swish,
                                                              0,
                                                              ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, GLUFusionTest4) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{-1, -1, 6});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(0));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<GLUFusion>();
    }
}

TEST_F(TransformationTestsF, GeGLUFusionTest1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, 1, 6});
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto gelu = std::make_shared<ov::op::v7::Gelu>(variadic_split->output(1));
        auto mul = std::make_shared<ov::op::v1::Multiply>(variadic_split->output(0), gelu);

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<GLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{2, 1, 6});
        auto swiglu = std::make_shared<ov::op::internal::GLU>(input,
                                                              axis,
                                                              split_lenghts,
                                                              ov::op::internal::GLU::GluType::Gelu,
                                                              1,
                                                              ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}
