// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/reshape_sequence_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <queue>
#include <string>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/visibility.hpp"
#include "openvino/opsets/opset6.hpp"

using namespace testing;
using namespace ov;

namespace {
Output<Node> reshape(Output<Node> input, std::vector<int64_t> values, bool special_zero = true) {
    return std::make_shared<opset6::Reshape>(input,
                                             opset6::Constant::create(element::i64, Shape{values.size()}, values),
                                             special_zero);
}
}  // namespace

TEST_F(TransformationTestsF, ReshapeSequenceFusion1) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {3, 2});
        auto b = reshape(a, {2, 3});
        auto c = reshape(b, {6});
        model = std::make_shared<Model>(OutputVector{c}, ParameterVector{data});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto c = reshape(data, {6});
        model_ref = std::make_shared<Model>(OutputVector{c}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusion2) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {3, 2});
        auto b = reshape(a, {6});
        model = std::make_shared<Model>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto c = reshape(data, {6});
        model_ref = std::make_shared<Model>(OutputVector{c}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusion3_special_zero_true) {
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{6});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{6});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusion3_special_zero_false) {
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{6});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{6});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153165
TEST_F(TransformationTestsF, DISABLED_ReshapeSequenceFusion4_i32_special_zero_true) {
#else
TEST_F(TransformationTestsF, ReshapeSequenceFusion4_i32_special_zero_true) {
#endif
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153165
TEST_F(TransformationTestsF, DISABLED_ReshapeSequenceFusion4_i32_special_zero_false) {
#else
TEST_F(TransformationTestsF, ReshapeSequenceFusion4_i32_special_zero_false) {
#endif
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153165
TEST_F(TransformationTestsF, DISABLED_ReshapeSequenceFusion4_i64_special_zero_true) {
#else
TEST_F(TransformationTestsF, ReshapeSequenceFusion4_i64_special_zero_true) {
#endif
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i64, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i64, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153165
TEST_F(TransformationTestsF, DISABLED_ReshapeSequenceFusion4_i64_special_zero_false) {
#else
TEST_F(TransformationTestsF, ReshapeSequenceFusion4_i64_special_zero_false) {
#endif
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i64, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i64, PartialShape{-1, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153165
TEST_F(TransformationTestsF, DISABLED_ReshapeSequenceFusion5_special_zero_true) {
#else
TEST_F(TransformationTestsF, ReshapeSequenceFusion5_special_zero_true) {
#endif
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, 3, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, 3, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

#if (defined OPENVINO_ARCH_ARM && defined(__linux__))
// Ticket: 153165
TEST_F(TransformationTestsF, DISABLED_ReshapeSequenceFusion5_special_zero_false) {
#else
TEST_F(TransformationTestsF, ReshapeSequenceFusion5_special_zero_false) {
#endif
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, 3, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto target_shape_param = std::make_shared<opset6::Parameter>(element::i32, PartialShape{-1, 3, -1});
        auto reshape_b_pattern = std::make_shared<opset6::ShapeOf>(target_shape_param);
        auto reshape_b = std::make_shared<opset6::Reshape>(data, reshape_b_pattern, special_zero);
        model_ref = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, target_shape_param});
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg1) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {-1, 2});
        auto b = reshape(a, {6});
        model = std::make_shared<Model>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg2) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {-1, 3});
        auto b = reshape(a, {6});
        model = std::make_shared<Model>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg3) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {2, 3});
        auto b = reshape(a, {6});
        model = std::make_shared<Model>(OutputVector{a, b}, ParameterVector{data});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg4) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto a = reshape(data, {2, 3});
        auto b = reshape(a, {0, 3});
        model = std::make_shared<Model>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg5_special_zero_true) {
    {
        const bool special_zero = true;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2});
        auto reshape_b_pattern = std::make_shared<opset6::Parameter>(element::i32, PartialShape{3});
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, reshape_b_pattern});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionNeg5_special_zero_false) {
    {
        const bool special_zero = false;
        auto data = std::make_shared<opset6::Parameter>(element::f32, PartialShape{1, 2, 3});
        auto reshape_a = reshape(data, {3, 2}, special_zero);
        auto reshape_b_pattern = std::make_shared<opset6::Parameter>(element::i32, PartialShape{3});
        auto reshape_b = std::make_shared<opset6::Reshape>(reshape_a, reshape_b_pattern, special_zero);
        model = std::make_shared<Model>(OutputVector{reshape_b}, ParameterVector{data, reshape_b_pattern});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }
}

TEST_F(TransformationTestsF, ReshapeSequenceFusionEliminate) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto relu = std::make_shared<opset6::Relu>(data);
        auto a = reshape(relu, {2, 3});
        auto b = reshape(a, {1, 2, 3});
        model = std::make_shared<Model>(OutputVector{b}, ParameterVector{data});

        manager.register_pass<ov::pass::ReshapeSequenceFusion>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 2, 3});
        auto relu = std::make_shared<opset6::Relu>(data);
        model_ref = std::make_shared<Model>(OutputVector{relu}, ParameterVector{data});
    }
}
