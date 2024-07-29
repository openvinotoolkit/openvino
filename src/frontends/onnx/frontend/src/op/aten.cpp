// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/embeddingbag_offsets_sum.hpp"
#include "openvino/op/embeddingbag_packedsum.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/opsets/opset8.hpp"
using namespace ov::op;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {

ov::OutputVector aten(const ov::frontend::onnx::Node& node) {
    ov::OutputVector inputs{node.get_ov_inputs()};

    const auto operator_name = node.get_attribute_value<std::string>("operator", "");
    CHECK_VALID_NODE(node,
                     operator_name == "embedding_bag",
                     "Only `embedding_bag` is supported as ATen `operator` attribute. Got: ",
                     operator_name);

    const auto mode = node.get_attribute_value<int64_t>("mode");
    CHECK_VALID_NODE(node,
                     mode == 0,
                     "Unsupported mode, only `0` (sum) is supported as ATen embedding_bag `mode` attribute. Got: ",
                     mode);
    CHECK_VALID_NODE(node, inputs.size() >= 2, "Minimum 2 inputs are required. Got: ", inputs.size());

    const bool is_packed_two_inputs =
        inputs.size() == 2 || (inputs.size() == 3 && ov::op::util::is_null(inputs[2])) ||
        (inputs.size() == 4 && ov::op::util::is_null(inputs[2]) && ov::op::util::is_null(inputs[3]));
    const bool is_packed_three_inputs =
        inputs.size() == 4 && ov::op::util::is_null(inputs[2]) && !ov::op::util::is_null(inputs[3]);
    const bool is_offsets_three_inputs = inputs.size() == 3 && !ov::op::util::is_null(inputs[2]);

    ov::Output<ov::Node> embedding_bag;
    if (is_packed_two_inputs) {
        embedding_bag = std::make_shared<v3::EmbeddingBagPackedSum>(inputs[0], inputs[1]);
    } else if (is_packed_three_inputs) {
        embedding_bag = std::make_shared<v3::EmbeddingBagPackedSum>(inputs[0], inputs[1], inputs[3]);
    } else if (is_offsets_three_inputs) {
        embedding_bag = std::make_shared<v3::EmbeddingBagOffsetsSum>(inputs[0], inputs[1], inputs[2]);
    } else if (inputs.size() >= 4) {
        // Need to expand embedding table with zeros (default values for empty bags)
        const auto& emb_tbl_in = inputs[0];
        const auto& indices_in = inputs[1];
        const auto& offsets_in = inputs[2];
        const auto& per_sample_weights_in = inputs[3];

        const auto data_type = emb_tbl_in.get_element_type();
        const auto ind_type = indices_in.get_element_type();

        const auto zero_const = std::make_shared<v0::Constant>(ind_type, ov::Shape{}, 0);

        // Shape aligned node, filled with zeros
        const auto zero_of_data_type_const = std::make_shared<v0::Constant>(data_type, ov::Shape{1}, 0);
        const auto weights_shape_node = std::make_shared<v3::ShapeOf>(emb_tbl_in, ind_type);
        const auto weights_last_dim_idx = std::make_shared<v0::Constant>(ov::element::i32, ov::Shape{1}, -1);
        const auto weights_last_dim =
            std::make_shared<v8::Gather>(weights_shape_node, weights_last_dim_idx, zero_const);
        const auto zero_col_node = std::make_shared<v3::Broadcast>(zero_of_data_type_const, weights_last_dim);
        const auto default_embeddings_node = std::make_shared<v0::Unsqueeze>(zero_col_node, zero_const);

        // Expanded embedding table weights
        const auto weights_concat =
            std::make_shared<v0::Concat>(ov::OutputVector{emb_tbl_in, default_embeddings_node}, 0);
        // Index in embedding table to fill empty bags
        const auto weights_first_dim =
            std::make_shared<v0::Squeeze>(std::make_shared<v8::Gather>(weights_shape_node, zero_const, zero_const));

        embedding_bag = std::make_shared<v3::EmbeddingBagOffsetsSum>(weights_concat,
                                                                     indices_in,
                                                                     offsets_in,
                                                                     weights_first_dim,  // default index
                                                                     per_sample_weights_in);

    } else {
        OPENVINO_THROW("Unsupported inputs configuration for ATen `embedding_bag` operation.");
    }
    // Enable import onnx Node with duplicated outputs
    return ov::OutputVector(node.get_outputs_size(), embedding_bag);
}

ONNX_OP("ATen", OPSET_SINCE(1), ai_onnx::opset_1::aten);
}  // namespace opset_1
}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
