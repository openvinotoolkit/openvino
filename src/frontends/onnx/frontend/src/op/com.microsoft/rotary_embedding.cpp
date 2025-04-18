// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/subtract.hpp"
#include "utils/common.hpp"


using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {

ov::OutputVector rotary_embedding(const ov::frontend::onnx::Node& node) {
    common::default_op_checks(node, 4);

    const auto inputs = node.get_ov_inputs();
    auto input = inputs[0];
    auto position_ids = inputs[1];
     auto cos_cache = inputs[2];
    auto sin_cache = inputs[3];

    const bool interleaved = static_cast<bool>(node.get_attribute_value<int64_t>("interleaved", 0));
    const bool is_packed_batching = static_cast<bool>(node.get_attribute_value<int64_t>("is_packed_batching", 0));
    const int64_t num_heads = node.get_attribute_value<int64_t>("num_heads", 0);
    const int64_t rotary_embedding_dim = node.get_attribute_value<int64_t>("rotary_embedding_dim", 0);
    const float scale = node.get_attribute_value<float>("scale", 1.0f);

    auto input_shape = std::make_shared<v3::ShapeOf>(input)->output(0);
    const auto input_rank = input.get_partial_shape().rank().get_length();

    auto gather_dim = [&](int64_t axis) -> ov::Output<ov::Node> {
        return std::make_shared<v8::Gather>(input_shape,
                                            v0::Constant::create(element::i64, Shape{1}, {axis}),
                                            v0::Constant::create(element::i64, Shape{}, {0}))
            ->output(0);
    };

    ov::Output<ov::Node> batch = gather_dim(0);
    ov::Output<ov::Node> seq_len = gather_dim((input_rank == 3) ? 1 : 2);
    ov::Output<ov::Node> hidden_size = gather_dim((input_rank == 3) ? 2 : 3);
    
ov::Output<ov::Node> rotary_dim;
    if (rotary_embedding_dim > 0) {
        rotary_dim = v0::Constant::create(ov::element::i64, ov::Shape{1}, {rotary_embedding_dim})->output(0);
    } else {

        auto last_dim_idx = v0::Constant::create(ov::element::i64, ov::Shape{1}, {input_rank - 1})->output(0);
        auto axis = v0::Constant::create(ov::element::i64, ov::Shape{}, {0})->output(0);

        rotary_dim = std::make_shared<v8::Gather>(input_shape, last_dim_idx, axis)->output(0);
    }

    auto start = v0::Constant::create(element::i64, Shape{1}, {0});
    auto step = v0::Constant::create(element::i64, Shape{1}, {1});
    auto axis = v0::Constant::create(element::i64, Shape{1}, {0});

    auto cos = std::make_shared<v8::Slice>(cos_cache, start, seq_len, step, axis)->output(0);
    auto sin = std::make_shared<v8::Slice>(sin_cache, start, seq_len, step, axis)->output(0);

    if (interleaved) {
        auto two = v0::Constant::create(element::i64, Shape{1}, {2})->output(0);

        auto cos_shape = std::make_shared<v3::ShapeOf>(cos)->output(0);
        auto cos_last_dim = std::make_shared<v8::Gather>(cos_shape,
                                                         v0::Constant::create(element::i64, Shape{1}, {-1}),
                                                         v0::Constant::create(element::i64, Shape{}, {0}))
                                ->output(0);

        ov::Output<ov::Node> input_bns;
        if (input_rank > 1) {
            input_bns = std::make_shared<v8::Slice>(input_shape,
                                                    v0::Constant::create(element::i64, Shape{1}, {0}),
                                                    v0::Constant::create(element::i64, Shape{1}, {input_rank - 1}),
                                                    v0::Constant::create(element::i64, Shape{1}, {1}),
                                                    v0::Constant::create(element::i64, Shape{1}, {0}))
                            ->output(0);
        } else {
            input_bns = v0::Constant::create(element::i64, Shape{0}, {})->output(0);
        }
   
         auto reshape_shape =
            std::make_shared<v0::Concat>(OutputVector{input_bns, cos_last_dim, two}, 0)->output(0);
        auto reshaped_input = std::make_shared<v1::Reshape>(input, reshape_shape, false)->output(0);

        auto split = std::make_shared<v1::Split>(reshaped_input, v0::Constant::create(element::i64, Shape{}, {-1}), 2);
        auto in0 = std::make_shared<v1::Reshape>(
                       split->output(0),
                       std::make_shared<v0::Concat>(OutputVector{input_bns, cos_last_dim}, 0)->output(0),
                       false)
                       ->output(0);
        auto in1 = std::make_shared<v1::Reshape>(
                       split->output(1),
                       std::make_shared<v0::Concat>(OutputVector{input_bns, cos_last_dim}, 0)->output(0),
                       false)
                       ->output(0);

        auto rotated_0 = std::make_shared<v1::Subtract>(std::make_shared<v1::Multiply>(in0, cos)->output(0),
                                                        std::make_shared<v1::Multiply>(in1, sin)->output(0))
                             ->output(0);
        auto rotated_1 = std::make_shared<v1::Add>(std::make_shared<v1::Multiply>(in0, sin)->output(0),
                                                   std::make_shared<v1::Multiply>(in1, cos)->output(0))
                             ->output(0);

        auto concat = std::make_shared<v0::Concat>(OutputVector{rotated_0, rotated_1}, -1)->output(0);
        auto scaled =
            std::make_shared<v1::Multiply>(concat, v0::Constant::create(concat.get_element_type(), Shape{}, {scale}))
                ->output(0);

        return {std::make_shared<v1::Reshape>(scaled, input_shape, false)->output(0)};
    } 
else {
        if (rotary_embedding_dim == 0) {
            return {std::make_shared<op::v1::Multiply>(
                input,
                op::v0::Constant::create(input.get_element_type(), Shape{}, {scale}))};
        }

        // Get full input shape
        auto input_shape = std::make_shared<op::v3::ShapeOf>(input);
        auto rank = input.get_partial_shape().rank().get_length();
        const auto last_axis = rank - 1;

        // Prepare constants
        auto zero = op::v0::Constant::create(element::i64, Shape{1}, {0});
        auto one = op::v0::Constant::create(element::i64, Shape{1}, {1});
        auto last_axis_tensor = op::v0::Constant::create(element::i64, Shape{1}, {last_axis});
        auto axes_tensor = op::v0::Constant::create(element::i64, Shape{1}, {last_axis});

        // ---------------- Slice rotary part input[..., :rotary_dim]
        std::vector<int64_t> start_vals(rank, 0);
        auto start = op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(rank)}, start_vals);
        std::vector<int64_t> steps_vals(rank, 1);
        auto step = op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(rank)}, steps_vals);

        // Build stop shape: same as input shape but last dim = rotary_dim
        OutputVector stop_parts;
        for (int i = 0; i < rank - 1; ++i) {
            stop_parts.push_back(
                std::make_shared<op::v8::Gather>(input_shape,
                                                 op::v0::Constant::create(element::i64, Shape{1}, {i}),
                                                 op::v0::Constant::create(element::i64, Shape{}, {0})));
        }
        stop_parts.push_back(rotary_dim);  // override last dim
        auto stop = std::make_shared<op::v0::Concat>(stop_parts, 0);

        auto rotary_part = std::make_shared<op::v8::Slice>(input, start, stop, step);

        // ---------------- Split rotary part
        auto split = std::make_shared<op::v1::Split>(rotary_part,
                                                     op::v0::Constant::create(element::i64, Shape{}, {last_axis}),
                                                     2);
        auto x0 = split->output(0);
        auto x1 = split->output(1);

        // ---------------- Reshape cos/sin to match x0/x1
        auto cos_broadcasted = std::make_shared<op::v3::Broadcast>(cos, std::make_shared<op::v3::ShapeOf>(x0));
        auto sin_broadcasted = std::make_shared<op::v3::Broadcast>(sin, std::make_shared<op::v3::ShapeOf>(x0));

        // ---------------- Rotate
        auto rotated_0 = std::make_shared<op::v1::Subtract>(std::make_shared<op::v1::Multiply>(x0, cos_broadcasted),
                                                            std::make_shared<op::v1::Multiply>(x1, sin_broadcasted));
        auto rotated_1 = std::make_shared<op::v1::Add>(std::make_shared<op::v1::Multiply>(x0, sin_broadcasted),
                                                       std::make_shared<op::v1::Multiply>(x1, cos_broadcasted));

        auto rotated = std::make_shared<op::v0::Concat>(ov::OutputVector{rotated_0, rotated_1}, last_axis);

        // ---------------- Slice input[..., rotary_dim:]
        std::vector<int64_t> rest_start_vals(rank, 0);
        rest_start_vals[rank - 1] = rotary_embedding_dim;
        auto rest_start = op::v0::Constant::create(element::i64, Shape{static_cast<size_t>(rank)}, rest_start_vals);
        auto rest = std::make_shared<op::v8::Slice>(input, rest_start, input_shape, step);

        // ---------------- Final concat
        auto final = std::make_shared<op::v0::Concat>(OutputVector{rotated, rest}, last_axis);

        // ---------------- Apply scale
        auto scaled =
            std::make_shared<op::v1::Multiply>(final,
                                               op::v0::Constant::create(final->get_element_type(), Shape{}, {scale}));

        return {scaled->output(0)};
    }
}

ONNX_OP("RotaryEmbedding", OPSET_SINCE(1), com_microsoft::opset_1::rotary_embedding, MICROSOFT_DOMAIN);

}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
