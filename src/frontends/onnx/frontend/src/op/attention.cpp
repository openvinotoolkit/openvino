// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "core/null_node.hpp"
#include "core/operator_set.hpp"
#include "openvino/op/scaled_dot_product_attention.hpp"
#include "utils/common.hpp"

using namespace ov::op;
using ov::Shape;

namespace ov {
namespace frontend {
namespace onnx {
namespace ai_onnx {
namespace opset_1 {
ov::OutputVector attention(const ov::frontend::onnx::Node& node) {
    auto nodes = node.get_ov_inputs();

    const auto& query = nodes[0];
    const auto& key = nodes[1];
    const auto& value = nodes[2];

    std::shared_ptr<ov::Node> scale = nullptr;
    if (node.has_attribute("scale")) {
        float scale_value = node.get_attribute_value<float>("scale");
        scale = v0::Constant::create(query.get_element_type(), ov::Shape{}, {scale_value});
    }

    const bool is_causal = static_cast<bool>(node.get_attribute_value<int64_t>("causal", 0));
    const bool has_attention_mask = nodes.size() > 4 && !ov::op::util::is_null(nodes[4]);

    std::shared_ptr<ov::Node> sdpa_output;
    if (has_attention_mask && scale) {
        sdpa_output = std::make_shared<v13::ScaledDotProductAttention>(query, key, value, nodes[4], scale, is_causal);
    } else if (has_attention_mask) {
        sdpa_output = std::make_shared<v13::ScaledDotProductAttention>(query, key, value, nodes[4], is_causal);
    } else if (scale) {
        // Need to provide a dummy mask to use scale parameter - create all-zeros mask (no masking)
        auto dummy_mask = v0::Constant::create(query.get_element_type(), ov::Shape{}, {0.0});
        sdpa_output = std::make_shared<v13::ScaledDotProductAttention>(query, key, value, dummy_mask, scale, is_causal);
    } else {
        sdpa_output = std::make_shared<v13::ScaledDotProductAttention>(query, key, value, is_causal);
    }
    return {sdpa_output};
}

}  // namespace opset_1
// TODO: Align opset number after ONNX version upgrade
ONNX_OP("Attention", OPSET_SINCE(1), ai_onnx::opset_1::attention);

}  // namespace ai_onnx
}  // namespace onnx
}  // namespace frontend
}  // namespace ov
