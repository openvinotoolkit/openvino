// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cmath"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/greater.hpp"
#include "openvino/op/greater_eq.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "utils/common.hpp"
#include "utils/reshape.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector make_split(const ov::Output<ov::Node>& value, int64_t num_splits, int64_t axis) {
    using namespace ov::op;
    const auto axis_node = v0::Constant::create(ov::element::i64, ov::Shape{}, {axis});
    const auto split = std::make_shared<v1::Split>(value, axis_node, num_splits);

    return split->outputs();
}

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
    const auto& input = inputs[0];         // [bs,seqlen,hidden] or [bs,num_heads,seqlen,headsize]
    const auto& position_ids = inputs[1];  // [seqlen] or [bs, seqlen]
    const auto& cos_cache = inputs[2];     // [max_seqlen, head_size/2]
    const auto& sin_cache = inputs[3];     // [max_seqlen, head_size/2]

    const auto interleaved = node.get_attribute_value<int64_t>("interleaved");  // required
    const auto minus_one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    const auto zero = v0::Constant::create(ov::element::i64, ov::Shape{1}, {0});
    const auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    const auto two = v0::Constant::create(ov::element::i64, ov::Shape{1}, {2});

    const auto cos = std::make_shared<v8::Gather>(cos_cache,
                                                  position_ids,
                                                  zero);  // [seqlen, head_size/2] or [bs, seqlen, head_size/2]
    const auto sin = std::make_shared<v8::Gather>(sin_cache,
                                                  position_ids,
                                                  zero);  // [seqlen, head_size/2] or [bs, seqlen, head_size/2]

    const auto input_shape = std::make_shared<v3::ShapeOf>(input);
    const bool input_is_3d = input.get_partial_shape().rank().get_length() == 3;
    const auto perm = v0::Constant::create(ov::element::i64, ov::Shape{4}, {0, 2, 1, 3});

    ov::Output<ov::Node> input_4d = input;
    if (input_is_3d) {
        const auto headsize = v0::Constant::create(ov::element::i64, ov::Shape{1}, {cos_cache.get_shape()[-1] * 2});
        const auto input_shape_prev_2 = get_dimensions(input_shape, {0, 1});
        auto new_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{input_shape_prev_2, minus_one, headsize}, 0);
        auto input_reshaped =
            std::make_shared<v1::Reshape>(input, new_input_shape, false);  // [bs,seqlen,num_heads,headsize]
        input_4d = std::make_shared<v1::Transpose>(input_reshaped, perm);  // [bs,num_heads,seqlen,headsize]
    }

    ov::Output<ov::Node> output;
    if (interleaved) {
        auto input_4d_shape = std::make_shared<v3::ShapeOf>(input_4d);
        auto dim_bns = get_dimensions(input_4d_shape, {0, 1, 2});
        auto half_head_size = v0::Constant::create(ov::element::i64, ov::Shape{1}, {cos_cache.get_shape()[-1]});
        auto split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_head_size, two}, 0);
        auto reshaped_input = std::make_shared<v1::Reshape>(input_4d, split_input_shape, false);

        auto in_split = make_split(reshaped_input, 2, -1);
        split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_head_size}, 0);
        auto in_split_0 = std::make_shared<v1::Reshape>(in_split[0], split_input_shape, false);
        auto in_split_1 = std::make_shared<v1::Reshape>(in_split[1], split_input_shape, false);

        auto res_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in_split_0, cos),
                                                    std::make_shared<v1::Multiply>(in_split_1, sin));
        auto res_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in_split_0, sin),
                                               std::make_shared<v1::Multiply>(in_split_1, cos));

        split_input_shape = std::make_shared<v0::Concat>(ov::NodeVector{dim_bns, half_head_size, one}, 0);
        auto res_0_5d = std::make_shared<v1::Reshape>(res_0, split_input_shape, false);
        auto res_1_5d = std::make_shared<v1::Reshape>(res_1, split_input_shape, false);

        auto concat_ret = std::make_shared<v0::Concat>(ov::NodeVector{res_0_5d, res_1_5d}, -1);
        output = std::make_shared<v1::Reshape>(concat_ret, input_4d_shape,
                                               false);  // [bs,num_heads,seqlen,headsize]
    } else {
        auto in_split = make_split(input_4d, 2, -1);  // [bs,num_heads,seqlen,headsize/2]
        auto res_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in_split[0], cos),
                                                    std::make_shared<v1::Multiply>(in_split[1], sin));
        auto res_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in_split[0], sin),
                                               std::make_shared<v1::Multiply>(in_split[1], cos));
        output = std::make_shared<v0::Concat>(ov::NodeVector{res_0, res_1}, -1);  // [bs,num_heads,seqlen,headsize]
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