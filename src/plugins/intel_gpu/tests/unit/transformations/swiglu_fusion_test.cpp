// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include <transformations/utils/utils.hpp>

#include <plugin/transformations/swiglu_fusion.hpp>

#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/gelu.hpp"
#include "openvino/op/variadic_split.hpp"
#include "intel_gpu/op/swiglu.hpp"

using namespace testing;
using namespace ov::intel_gpu;

TEST_F(TransformationTestsF, SwiGLUFusionTest1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 1, 6 });
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(1));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<SwiGLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 1, 6 });
        auto swiglu = std::make_shared<op::SwiGLU>(input, axis, split_lenghts, op::SwiGLU::GluType::Swish, 0, ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwiGLUFusionTest2) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 6 });
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {0});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, 3});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(1));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<SwiGLUFusion>();
    }
}

TEST_F(TransformationTestsF, SwiGLUFusionTest3) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 6 });
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(1));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<SwiGLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 6 });
        auto swiglu = std::make_shared<op::SwiGLU>(input, axis, split_lenghts, op::SwiGLU::GluType::Swish, 0, ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwiGLUFusionTest3ReverseOrder) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 6 });
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(variadic_split->output(1), swish);

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<SwiGLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 6 });
        auto swiglu = std::make_shared<op::SwiGLU>(input, axis, split_lenghts, op::SwiGLU::GluType::Swish, 0, ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, SwiGLUFusionTest4) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ -1, -1, 6 });
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto swish = std::make_shared<ov::op::v4::Swish>(variadic_split->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, variadic_split->output(0));

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<SwiGLUFusion>();
    }
}

TEST_F(TransformationTestsF, GeGLUFusionTest1) {
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 1, 6 });
        auto axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split_lengths_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {3, -1});
        auto variadic_split = std::make_shared<ov::op::v1::VariadicSplit>(input, axis_const, split_lengths_const);
        auto gelu = std::make_shared<ov::op::v7::Gelu>(variadic_split->output(1));
        auto mul = std::make_shared<ov::op::v1::Multiply>(variadic_split->output(0), gelu);

        model = std::make_shared<ov::Model>(ov::NodeVector{mul}, ov::ParameterVector{input});
        manager.register_pass<SwiGLUFusion>();
    }
    {
        int64_t axis = -1;
        int64_t split_lenghts = 3;
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f16, ov::PartialShape{ 2, 1, 6 });
        auto swiglu = std::make_shared<op::SwiGLU>(input, axis, split_lenghts, op::SwiGLU::GluType::Gelu, 1, ov::element::f16);

        model_ref = std::make_shared<ov::Model>(ov::NodeVector{swiglu}, ov::ParameterVector{input});
    }
}
