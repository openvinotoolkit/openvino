// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/aten.hpp"

#include "default_opset.hpp"
#include "exceptions.hpp"
#include "onnx_import/core/node.hpp"
#include "onnx_import/core/null_node.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {

OutputVector aten(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};

    std::string operator_name = node.get_attribute_value<std::string>("operator", "");
    CHECK_VALID_NODE(node,
                     operator_name == "embedding_bag",
                     "Only `embedding_bag` is supported as ATen operator attribute.");

    const int64_t mode = node.get_attribute_value<int64_t>("mode");
    CHECK_VALID_NODE(node, mode == 0, "Unsupported mode, only `sum` as ATen embedding_bag `mode` attribute.");
    CHECK_VALID_NODE(node, inputs.size() >= 2, "Minimum 2 inputs are required, Got: ", inputs.size());

    std::shared_ptr<ov::Node> embedding_bag;
    if (inputs.size() == 2) {  // embedding table and indices (packed) input provided
        embedding_bag = std::make_shared<default_opset::EmbeddingBagPackedSum>(inputs[0], inputs[1]);
    } else if (inputs.size() == 3) {
        if (ngraph::op::is_null(inputs.at(2))) {  // no offsets input
            embedding_bag = std::make_shared<default_opset::EmbeddingBagPackedSum>(inputs[0], inputs[1]);
        } else {
            embedding_bag = std::make_shared<default_opset::EmbeddingBagOffsetsSum>(inputs[0], inputs[1], inputs[2]);
        }
    } else if (inputs.size() >= 4 && !ngraph::op::is_null(inputs.at(3))) {  // per_sample_weights input provided
        if (ngraph::op::is_null(inputs.at(2))) {                            // no offsets input
            embedding_bag = std::make_shared<default_opset::EmbeddingBagPackedSum>(inputs[0], inputs[1], inputs[3]);
        } else {
            // Expand embedding table with zeros (default values for empty bags)
            const auto data_type = inputs[0].get_element_type();
            const auto ind_type = inputs[1].get_element_type();

            const auto weights_shape_node = std::make_shared<default_opset::ShapeOf>(inputs[0], ind_type);
            const auto weights_rank_node = std::make_shared<default_opset::ShapeOf>(weights_shape_node, ind_type);

            const auto zero_const = std::make_shared<default_opset::Constant>(ind_type, Shape{}, 0);
            const auto one_const =
                std::make_shared<default_opset::Constant>(ind_type, Shape{1}, 1);  // Change to last, Rank -1 ?

            // Shape aligned node, filled with zeros
            const auto zero_of_data_type_const = std::make_shared<default_opset::Constant>(data_type, Shape{1}, 0);
            const auto weights_last_dim =
                std::make_shared<default_opset::Gather>(weights_shape_node, one_const, zero_const);
            const auto zero_col_node =
                std::make_shared<default_opset::Broadcast>(zero_of_data_type_const, weights_last_dim);
            const auto default_embeddings_node = std::make_shared<default_opset::Unsqueeze>(zero_col_node, zero_const);

            // Expanded input at 0 (embedding table weights)
            const auto weights_concat =
                std::make_shared<default_opset::Concat>(OutputVector{inputs[0], default_embeddings_node}, 0);
            // Index in embedding table to fill empty bags
            const auto weights_first_dim = std::make_shared<default_opset::Squeeze>(
                std::make_shared<default_opset::Gather>(weights_shape_node, zero_const, zero_const));

            embedding_bag = std::make_shared<default_opset::EmbeddingBagOffsetsSum>(weights_concat,
                                                                                    inputs[1],
                                                                                    inputs[2],
                                                                                    weights_first_dim,
                                                                                    inputs[3]);
        }
    } else {
        OPENVINO_UNREACHABLE("Unsupported inputs configuration for ATen `embedding_bag` operation.");
    }
    return OutputVector(node.get_outputs_size(), embedding_bag->output(0));
}

}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
