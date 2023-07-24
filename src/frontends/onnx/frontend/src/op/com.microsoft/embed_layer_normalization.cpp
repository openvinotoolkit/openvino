// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/com.microsoft/embed_layer_normalization.hpp"

#include "default_opset.hpp"
#include "onnx_import/core/null_node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector embed_layer_normalization(const Node& node) {
    auto nodes = node.get_ng_inputs();
    auto num_nodes = nodes.size();

    NGRAPH_CHECK(num_nodes >= 7 && num_nodes <= 9,
                 "EmbedLayerNormalization takes 7 or 9 inputs. Provided " + std::to_string(num_nodes));
    NGRAPH_CHECK(nodes[0].get_element_type() == element::i32, "input_ids must have int32 type");

    const auto& input_ids = nodes[0];
    const auto& segment_ids = nodes[1];
    const auto& word_embeddings = nodes[2];
    const auto& position_embeddings = nodes[3];
    const auto& segment_embeddings = nodes[4];
    const auto& gamma = nodes[5];
    const auto& beta = nodes[6];

    const auto zero = default_opset::Constant::create(element::i32, Shape{1}, {0});
    std::shared_ptr<ngraph::Node> input = std::make_shared<default_opset::Gather>(word_embeddings, input_ids, zero, 0);
    // add position embeddings
    if (num_nodes > 8 && !ngraph::op::is_null(nodes[8])) {
        // if we have position_ids
        const auto& position_ids = nodes[8];
        const auto gathered_position_embeddings =
            std::make_shared<default_opset::Gather>(position_embeddings, position_ids, zero, 0);
        input = std::make_shared<default_opset::Add>(input, gathered_position_embeddings);
    } else {
        // input_ids' shape is [batchsize, sequence_length]
        // input's shape is [batchsize, sequence_length, hidden_size]
        // position_embeddings's shape is [max_sequence_length, hidden_size]
        // therefore input and position_embeddings cannot be added together
        // so we need slice the position_embeddings to [sequence_length, hidden_size] first
        // then add it with input.
        const auto one = default_opset::Constant::create(element::i32, Shape{1}, {1});
        const auto input_ids_shape = std::make_shared<default_opset::ShapeOf>(input_ids, element::i32);
        const auto seqlen = std::make_shared<default_opset::Gather>(input_ids_shape, one, zero, 0);
        const auto gathered_position_embeddings =
            std::make_shared<default_opset::Slice>(position_embeddings, zero, seqlen, one, zero);
        input = std::make_shared<default_opset::Add>(input, gathered_position_embeddings);
    }
    // add segment embeddings if available
    if (!ngraph::op::is_null(segment_ids)) {
        NGRAPH_CHECK(!ngraph::op::is_null(segment_embeddings),
                     "segment_ids provided, but segment_embedding input is missing");
        NGRAPH_CHECK(nodes[1].get_element_type() == element::i32, "segment_ids must have int32 type");
        auto gathered_segment_embeddings =
            std::make_shared<default_opset::Gather>(segment_embeddings, segment_ids, zero, 0);
        input = std::make_shared<default_opset::Add>(input, gathered_segment_embeddings);
    }

    float eps = node.get_attribute_value<float>("epsilon");
    // reduce over hidden_size
    // hidden_size dimension is 2 here, because the shape after Gather(word_embedding, input_ids)
    // is (batch_size, seq_len, hidden_size)
    int hidden_size_dim = 2;
    const auto reduction_axes = default_opset::Constant::create(element::i32, Shape{1}, {hidden_size_dim});
    std::shared_ptr<ngraph::Node> result =
        std::make_shared<default_opset::MVN>(input, reduction_axes, true, eps, ov::op::MVNEpsMode::INSIDE_SQRT);

    // result = gamma * result + beta
    result = std::make_shared<default_opset::Multiply>(result, gamma);
    result = std::make_shared<default_opset::Add>(result, beta);

    // compute mask_index output
    std::shared_ptr<ngraph::Node> mask_index;
    if (num_nodes > 7 && !ngraph::op::is_null(nodes[7])) {
        NGRAPH_CHECK(nodes[7].get_element_type() == element::i32, "mask must have int32 type");
        auto axis = default_opset::Constant::create(element::i32, Shape{}, {1});
        mask_index = std::make_shared<default_opset::ReduceSum>(nodes[7], axis, false);
    } else {
        auto batch_size = std::make_shared<default_opset::Gather>(std::make_shared<default_opset::ShapeOf>(nodes[0]),
                                                                  zero,   // indices
                                                                  zero);  // axis
        mask_index = std::make_shared<default_opset::Broadcast>(zero, batch_size);
    }
    return {result, mask_index};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
