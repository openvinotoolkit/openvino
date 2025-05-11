// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/frontend/exception.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/mvn.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace com_microsoft {
namespace opset_1 {
ov::OutputVector embed_layer_normalization(const ov::frontend::onnx::Node& node) {
    auto nodes = node.get_ov_inputs();
    auto num_nodes = nodes.size();

    FRONT_END_GENERAL_CHECK(num_nodes >= 7 && num_nodes <= 9,
                            "EmbedLayerNormalization takes 7 or 9 inputs. Provided " + std::to_string(num_nodes));
    FRONT_END_GENERAL_CHECK(nodes[0].get_element_type() == ov::element::i32, "input_ids must have int32 type");

    const auto& input_ids = nodes[0];
    const auto& segment_ids = nodes[1];
    const auto& word_embeddings = nodes[2];
    const auto& position_embeddings = nodes[3];
    const auto& segment_embeddings = nodes[4];
    const auto& gamma = nodes[5];
    const auto& beta = nodes[6];

    const auto zero = v0::Constant::create(ov::element::i32, ov::Shape{1}, {0});
    std::shared_ptr<ov::Node> input = std::make_shared<v8::Gather>(word_embeddings, input_ids, zero, 0);
    // add position embeddings
    if (num_nodes > 8 && !ov::op::util::is_null(nodes[8])) {
        // if we have position_ids
        const auto& position_ids = nodes[8];
        const auto gathered_position_embeddings =
            std::make_shared<v8::Gather>(position_embeddings, position_ids, zero, 0);
        input = std::make_shared<v1::Add>(input, gathered_position_embeddings);
    } else {
        // input_ids' shape is [batchsize, sequence_length]
        // input's shape is [batchsize, sequence_length, hidden_size]
        // position_embeddings's shape is [max_sequence_length, hidden_size]
        // therefore input and position_embeddings cannot be added together
        // so we need slice the position_embeddings to [sequence_length, hidden_size] first
        // then add it with input.
        const auto one = v0::Constant::create(ov::element::i32, ov::Shape{1}, {1});
        const auto input_ids_shape = std::make_shared<v3::ShapeOf>(input_ids, ov::element::i32);
        const auto seqlen = std::make_shared<v8::Gather>(input_ids_shape, one, zero, 0);
        const auto gathered_position_embeddings =
            std::make_shared<v8::Slice>(position_embeddings, zero, seqlen, one, zero);
        input = std::make_shared<v1::Add>(input, gathered_position_embeddings);
    }
    // add segment embeddings if available
    if (!ov::op::util::is_null(segment_ids)) {
        FRONT_END_GENERAL_CHECK(!ov::op::util::is_null(segment_embeddings),
                                "segment_ids provided, but segment_embedding input is missing");
        FRONT_END_GENERAL_CHECK(nodes[1].get_element_type() == ov::element::i32, "segment_ids must have int32 type");
        auto gathered_segment_embeddings = std::make_shared<v8::Gather>(segment_embeddings, segment_ids, zero, 0);
        input = std::make_shared<v1::Add>(input, gathered_segment_embeddings);
    }

    float eps = node.get_attribute_value<float>("epsilon");
    // reduce over hidden_size
    // hidden_size dimension is 2 here, because the shape after Gather(word_embedding, input_ids)
    // is (batch_size, seq_len, hidden_size)
    int hidden_size_dim = 2;
    const auto reduction_axes = v0::Constant::create(ov::element::i32, ov::Shape{1}, {hidden_size_dim});
    std::shared_ptr<ov::Node> result =
        std::make_shared<v6::MVN>(input, reduction_axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    // result = gamma * result + beta
    result = std::make_shared<v1::Multiply>(result, gamma);
    result = std::make_shared<v1::Add>(result, beta);

    // compute mask_index output
    std::shared_ptr<ov::Node> mask_index;
    if (num_nodes > 7 && !ov::op::util::is_null(nodes[7])) {
        FRONT_END_GENERAL_CHECK(nodes[7].get_element_type() == ov::element::i32, "mask must have int32 type");
        auto axis = v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        mask_index = std::make_shared<v1::ReduceSum>(nodes[7], axis, false);
    } else {
        auto batch_size = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(nodes[0]),
                                                       zero,   // indices
                                                       zero);  // axis
        mask_index = std::make_shared<v3::Broadcast>(zero, batch_size);
    }
    return {result, mask_index};
}
ONNX_OP("EmbedLayerNormalization", OPSET_SINCE(1), com_microsoft::opset_1::embed_layer_normalization, MICROSOFT_DOMAIN);
}  // namespace opset_1
}  // namespace com_microsoft
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
