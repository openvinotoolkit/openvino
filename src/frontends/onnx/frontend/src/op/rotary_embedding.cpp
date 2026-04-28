// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "utils/common.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_23 {

namespace {

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<v3::ShapeOf>& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

}  // namespace

ov::OutputVector rotary_embedding(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 3);

    const auto inputs = node.get_ov_inputs();
    const auto& input = inputs[0];
    const auto& cos_cache = inputs[1];
    const auto& sin_cache = inputs[2];

    const bool has_position_ids = common::is_input_valid(node, 3);

    const int64_t interleaved = node.get_attribute_value<int64_t>("interleaved", 0);
    const int64_t num_heads_attr = node.get_attribute_value<int64_t>("num_heads", 0);
    const int64_t rotary_embedding_dim = node.get_attribute_value<int64_t>("rotary_embedding_dim", 0);

    const auto minus_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    const auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    ov::Output<ov::Node> cos_values, sin_values;
    if (has_position_ids) {
        const auto& position_ids = inputs[3];
        cos_values = std::make_shared<v8::Gather>(cos_cache, position_ids, zero);
        sin_values = std::make_shared<v8::Gather>(sin_cache, position_ids, zero);
    } else {
        cos_values = cos_cache;
        sin_values = sin_cache;
    }

    const auto cos_cache_shape = cos_cache.get_partial_shape();
    const auto cos_cache_rank = cos_cache_shape.rank();
    CHECK_VALID_NODE(node,
                     cos_cache_rank.is_static() && cos_cache_rank.get_length() >= 1,
                     "cos_cache must have static rank with at least one dimension, got: ",
                     cos_cache_shape);
    const auto last_dim = cos_cache_shape[cos_cache_rank.get_length() - 1];
    CHECK_VALID_NODE(node,
                     last_dim.is_static(),
                     "cos_cache last dimension must be static to derive rotation dimension, got: ",
                     cos_cache_shape);
    const auto half_rotary_dim = static_cast<int64_t>(last_dim.get_length());
    const auto full_rotary_dim = half_rotary_dim * 2;

    const auto input_rank = input.get_partial_shape().rank();
    const bool input_is_3d = input_rank.is_static() && input_rank.get_length() == 3;
    const bool input_is_4d = input_rank.is_static() && input_rank.get_length() == 4;

    int64_t head_size_val = 0;
    if (input_is_4d) {
        const auto head_dim = input.get_partial_shape()[3];
        CHECK_VALID_NODE(node,
                         head_dim.is_static(),
                         "For 4D input, head_size dimension must be static, got: ",
                         input.get_partial_shape());
        head_size_val = static_cast<int64_t>(head_dim.get_length());
    } else if (input_is_3d) {
        CHECK_VALID_NODE(node, num_heads_attr > 0, "num_heads attribute must be provided for 3D input");
        const auto hidden_dim = input.get_partial_shape()[2];
        CHECK_VALID_NODE(node,
                         hidden_dim.is_static(),
                         "For 3D input, hidden_size dimension must be static, got: ",
                         input.get_partial_shape());
        head_size_val = static_cast<int64_t>(hidden_dim.get_length()) / num_heads_attr;
    }

    const int64_t actual_rotary_dim = (rotary_embedding_dim > 0) ? rotary_embedding_dim : head_size_val;
    const int64_t rot_half_dim = actual_rotary_dim / 2;

    const auto input_shape = std::make_shared<v3::ShapeOf>(input);
    const auto perm_3d_to_4d = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    ov::Output<ov::Node> input_4d = input;
    if (input_is_3d) {
        const auto headsize = v0::Constant::create(ov::element::i64, ov::Shape{1}, {head_size_val});
        const auto input_shape_prev_2 = get_dimensions(input_shape, {0, 1});
        auto new_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{input_shape_prev_2, minus_one, headsize}, 0);
        auto input_reshaped = std::make_shared<v1::Reshape>(input, new_input_shape, false);
        input_4d = std::make_shared<v1::Transpose>(input_reshaped, perm_3d_to_4d);
    }

    ov::Output<ov::Node> x_rotate, x_passthrough;
    bool has_passthrough = (actual_rotary_dim < head_size_val && head_size_val > 0);

    if (has_passthrough) {
        const auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        const auto split_lengths = v0::Constant::create(ov::element::i64,
                                                        ov::Shape{2},
                                                        {actual_rotary_dim, head_size_val - actual_rotary_dim});
        auto split = std::make_shared<v1::VariadicSplit>(input_4d, split_axis, split_lengths);
        x_rotate = split->output(0);
        x_passthrough = split->output(1);
    } else {
        x_rotate = input_4d;
    }

    ov::Output<ov::Node> cos_4d, sin_4d;
    const auto cos_out_rank = cos_values.get_partial_shape().rank();
    if (cos_out_rank.is_static() && cos_out_rank.get_length() == 2) {
        auto axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        cos_4d = std::make_shared<v0::Unsqueeze>(cos_values, axes);
        sin_4d = std::make_shared<v0::Unsqueeze>(sin_values, axes);
    } else {
        auto axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        cos_4d = std::make_shared<v0::Unsqueeze>(cos_values, axes);
        sin_4d = std::make_shared<v0::Unsqueeze>(sin_values, axes);
    }

    ov::Output<ov::Node> rope_input = x_rotate;
    std::shared_ptr<v3::ShapeOf> x_rotate_shape;
    std::shared_ptr<ov::Node> dim_bns;
    auto half_rotary = v0::Constant::create(ov::element::i64, ov::Shape{1}, {rot_half_dim});
    auto perm_5d = v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});

    if (interleaved) {
        x_rotate_shape = std::make_shared<v3::ShapeOf>(x_rotate);
        dim_bns = get_dimensions(x_rotate_shape, {0, 1, 2});

        auto deinterleave_5d = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_rotary, two}, 0);
        auto reshaped_5d = std::make_shared<v1::Reshape>(x_rotate, deinterleave_5d, false);
        auto transposed_5d = std::make_shared<v1::Transpose>(reshaped_5d, perm_5d);
        rope_input = std::make_shared<v1::Reshape>(transposed_5d, x_rotate_shape, false);
    }

    if (has_passthrough) {
        const auto slice_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        const auto slice_lengths =
            v0::Constant::create(ov::element::i64, ov::Shape{2}, {rot_half_dim, half_rotary_dim - rot_half_dim});
        cos_4d = std::make_shared<v1::VariadicSplit>(cos_4d, slice_axis, slice_lengths)->output(0);
        sin_4d = std::make_shared<v1::VariadicSplit>(sin_4d, slice_axis, slice_lengths)->output(0);
    }

    const auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    const auto split_lengths = v0::Constant::create(ov::element::i64, ov::Shape{2}, {rot_half_dim, rot_half_dim});
    auto in_split = std::make_shared<v1::VariadicSplit>(rope_input, split_axis, split_lengths)->outputs();

    auto x1_cos = std::make_shared<v1::Multiply>(in_split[0], cos_4d);
    auto x2_sin = std::make_shared<v1::Multiply>(in_split[1], sin_4d);
    const auto neg_one = v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto neg_x2_sin = std::make_shared<v1::Multiply>(x2_sin, neg_one);
    auto res_0 = std::make_shared<v1::Add>(x1_cos, neg_x2_sin);

    auto x2_cos = std::make_shared<v1::Multiply>(in_split[1], cos_4d);
    auto x1_sin = std::make_shared<v1::Multiply>(in_split[0], sin_4d);
    auto res_1 = std::make_shared<v1::Add>(x2_cos, x1_sin);

    ov::Output<ov::Node> rotated = std::make_shared<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);

    if (interleaved) {
        auto reinterleave_5d = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, two, half_rotary}, 0);
        auto result_5d = std::make_shared<v1::Reshape>(rotated, reinterleave_5d, false);
        auto result_transposed = std::make_shared<v1::Transpose>(result_5d, perm_5d);
        rotated = std::make_shared<v1::Reshape>(result_transposed, x_rotate_shape, false);
    }

    ov::Output<ov::Node> output;
    if (has_passthrough) {
        output = std::make_shared<v0::Concat>(ov::OutputVector{rotated, x_passthrough}, -1);
    } else {
        output = rotated;
    }

    if (input_is_3d) {
        output = std::make_shared<v1::Transpose>(output, perm_3d_to_4d);
        output = std::make_shared<v1::Reshape>(output, input_shape, false);
    }

    return {output};
}

ONNX_OP("RotaryEmbedding", OPSET_SINCE(1), ai_onnx::opset_23::rotary_embedding);

}  // namespace opset_23
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
