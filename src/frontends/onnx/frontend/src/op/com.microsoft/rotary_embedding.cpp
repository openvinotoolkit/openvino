// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<v3::ShapeOf>& shape, const std::vector<int>& dims) {
    static const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    const auto dims_const = v0::Constant::create(ov::element::i32, ov::Shape{dims.size()}, dims);
    return std::make_shared<v8::Gather>(shape, dims_const, zero);
}

ov::OutputVector rotary_embedding(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 4);
    // Original documentation:
    // https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RotaryEmbedding
    const auto inputs = node.get_ov_inputs();
    const auto& input = inputs[0];         // [bs,seqlen,hidden] or [bs,num_heads,seqlen,head_size]
    const auto& position_ids = inputs[1];  // [seqlen] or [bs, seqlen]
    const auto& cos_cache = inputs[2];     // [max_seqlen, head_size/2]
    const auto& sin_cache = inputs[3];     // [max_seqlen, head_size/2]

    const auto interleaved = node.get_attribute_value<int64_t>("interleaved");  // required
    const auto minus_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    const auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    const auto cos = std::make_shared<v8::Gather>(cos_cache,
                                                  position_ids,
                                                  zero);  // [seqlen, head_size/2] or [bs, seqlen, head_size/2]
    const auto sin = std::make_shared<v8::Gather>(sin_cache,
                                                  position_ids,
                                                  zero);  // [seqlen, head_size/2] or [bs, seqlen, head_size/2]

    const auto cos_cache_shape = cos_cache.get_partial_shape();
    const auto cos_cache_rank = cos_cache_shape.rank();
    CHECK_VALID_NODE(node,
                     cos_cache_rank.is_static() && cos_cache_rank.get_length() >= 1,
                     "cos_cache must have static rank with at least one dimension, got: ",
                     cos_cache_shape);
    const auto last_dim = cos_cache_shape[cos_cache_rank.get_length() - 1];
    CHECK_VALID_NODE(node,
                     last_dim.is_static(),
                     "cos_cache last dimension must be static to derive head size, got: ",
                     cos_cache_shape);
    const auto half_head_size_val = static_cast<int64_t>(last_dim.get_length());
    const auto head_size_val = half_head_size_val * 2;

    const auto input_shape = std::make_shared<v3::ShapeOf>(input);
    const auto input_rank = input.get_partial_shape().rank();
    const bool input_is_3d = input_rank.is_static() && input_rank.get_length() == 3;
    const auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    ov::Output<ov::Node> input_4d = input;
    if (input_is_3d) {
        const auto headsize = v0::Constant::create(ov::element::i64, ov::Shape{1}, {head_size_val});
        const auto input_shape_prev_2 = get_dimensions(input_shape, {0, 1});
        auto new_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{input_shape_prev_2, minus_one, headsize}, 0);
        auto input_reshaped =
            std::make_shared<v1::Reshape>(input, new_input_shape, false);  // [bs,seqlen,num_heads,head_size]
        input_4d = std::make_shared<v1::Transpose>(input_reshaped, perm);  // [bs,num_heads,seqlen,head_size]
    }

    // Unsqueeze cos/sin to 4D [?, 1, ?, head_size/2] to match RoPE fusion pattern
    ov::Output<ov::Node> cos_4d, sin_4d;
    const auto cos_out_rank = cos->get_output_partial_shape(0).rank();
    if (cos_out_rank.is_static() && cos_out_rank.get_length() == 2) {
        // cos is [seqlen, head_size/2] → [1, 1, seqlen, head_size/2]
        auto axes = v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        cos_4d = std::make_shared<v0::Unsqueeze>(cos, axes);
        sin_4d = std::make_shared<v0::Unsqueeze>(sin, axes);
    } else {
        // cos is [bs, seqlen, head_size/2] → [bs, 1, seqlen, head_size/2]
        auto axes = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
        cos_4d = std::make_shared<v0::Unsqueeze>(cos, axes);
        sin_4d = std::make_shared<v0::Unsqueeze>(sin, axes);
    }

    // For interleaved mode, deinterleave first so the core RoPE formula is identical
    ov::Output<ov::Node> rope_input = input_4d;
    std::shared_ptr<v3::ShapeOf> input_4d_shape;
    std::shared_ptr<ov::Node> dim_bns;
    std::shared_ptr<v0::Constant> half_head_size;
    std::shared_ptr<v0::Constant> perm_5d;
    if (interleaved) {
        input_4d_shape = std::make_shared<v3::ShapeOf>(input_4d);
        dim_bns = get_dimensions(input_4d_shape, {0, 1, 2});
        half_head_size = v0::Constant::create(ov::element::i64, ov::Shape{1}, {half_head_size_val});
        perm_5d = v0::Constant::create(ov::element::i64, ov::Shape{5}, {0, 1, 2, 4, 3});

        // Deinterleave: [bs,num_heads,seqlen,head_size]
        //   → reshape [bs,num_heads,seqlen,head_size/2,2]
        //   → transpose [bs,num_heads,seqlen,2,head_size/2]
        //   → reshape [bs,num_heads,seqlen,head_size]  (now [first_half, second_half])
        auto deinterleave_5d = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_head_size, two}, 0);
        auto reshaped_5d = std::make_shared<v1::Reshape>(input_4d, deinterleave_5d, false);
        auto transposed_5d = std::make_shared<v1::Transpose>(reshaped_5d, perm_5d);
        rope_input = std::make_shared<v1::Reshape>(transposed_5d, input_4d_shape, false);
    }

    // Core RoPE formula (matches RoPEFusionGPTOSS pattern for both modes)
    // first_ = first_half * cos - second_half * sin
    // second_ = second_half * cos + first_half * sin
    const auto split_axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    const auto split_lengths =
        v0::Constant::create(ov::element::i64, ov::Shape{2}, {half_head_size_val, half_head_size_val});
    // Split along last axis using constant split_lengths to enable RoPE fusion pattern matching
    auto in_split = std::make_shared<v1::VariadicSplit>(rope_input, split_axis, split_lengths)->outputs();
    auto first_half_mul_cos = std::make_shared<v1::Multiply>(in_split[0], cos_4d);
    auto second_half_mul_sin = std::make_shared<v1::Multiply>(in_split[1], sin_4d);
    const auto neg_one = v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto neg_second_sin = std::make_shared<v1::Multiply>(second_half_mul_sin, neg_one);
    auto res_0 = std::make_shared<v1::Add>(first_half_mul_cos, neg_second_sin);
    auto second_half_mul_cos = std::make_shared<v1::Multiply>(in_split[1], cos_4d);
    auto first_half_mul_sin = std::make_shared<v1::Multiply>(in_split[0], sin_4d);
    auto res_1 = std::make_shared<v1::Add>(second_half_mul_cos, first_half_mul_sin);
    ov::Output<ov::Node> output =
        std::make_shared<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);  // [bs,num_heads,seqlen,head_size]

    // For interleaved mode, re-interleave the result
    if (interleaved) {
        // Re-interleave: [bs,num_heads,seqlen,head_size]
        //   → reshape [bs,num_heads,seqlen,2,head_size/2]
        //   → transpose [bs,num_heads,seqlen,head_size/2,2]
        //   → reshape [bs,num_heads,seqlen,head_size]
        auto reinterleave_5d = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, two, half_head_size}, 0);
        auto result_5d = std::make_shared<v1::Reshape>(output, reinterleave_5d, false);
        auto result_transposed = std::make_shared<v1::Transpose>(result_5d, perm_5d);
        output = std::make_shared<v1::Reshape>(result_transposed,
                                               input_4d_shape,
                                               false);  // [bs,num_heads,seqlen,head_size]
    }

    if (input_is_3d) {
        output = std::make_shared<v1::Transpose>(output, perm);  // [bs,seqlen,num_heads,headsize]
        output = std::make_shared<v1::Reshape>(output, input_shape,
                                               false);  // [bs,seqlen,hidden]
    }

    return {output};
}

ONNX_OP("RotaryEmbedding", OPSET_SINCE(1), com_microsoft::opset_1::rotary_embedding, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov