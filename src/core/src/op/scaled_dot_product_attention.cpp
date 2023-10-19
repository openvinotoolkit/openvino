// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/scaled_dot_product_attention.hpp"

#include "itt.hpp"

using namespace std;
namespace ov {

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const OutputVector& inputs, bool causal)
    : op::Op(inputs),
      m_causal(causal) {
    constructor_validate_and_infer_types();
}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              const Output<Node>& attn_mask,
                                                              const Output<Node>& scale,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value, attn_mask, scale}, causal) {}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              const Output<Node>& attn_mask,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value, attn_mask}, causal) {}

op::v13::ScaledDotProductAttention::ScaledDotProductAttention(const Output<Node>& query,
                                                              const Output<Node>& key,
                                                              const Output<Node>& value,
                                                              bool causal)
    : ScaledDotProductAttention({query, key, value}, causal) {}

void op::v13::ScaledDotProductAttention::validate_and_infer_types() {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this, get_input_size() >= 3 && get_input_size() <= 5);
    // TODO: More checks and accurate deduction of dimensions in case when various
    // dynamic combinations appear.
    auto query = get_input_partial_shape(0);
    auto key = get_input_partial_shape(1);
    auto value = get_input_partial_shape(2);

    // using particular dimensions from query and value, to do that need to have them statically ranked
    if (query.rank().is_dynamic() || value.rank().is_dynamic()) {
        set_output_type(0, get_input_element_type(0), PartialShape::dynamic());
        return;
    }

    OPENVINO_ASSERT(query.rank().get_length() >= 3);
    OPENVINO_ASSERT(value.rank().get_length() >= 3);

    auto dimensions = std::vector<Dimension>(query.begin(), query.end() - 1);
    dimensions.push_back(*(value.end() - 1));
    set_output_type(0, get_input_element_type(0), PartialShape(dimensions));
}

std::shared_ptr<Node> op::v13::ScaledDotProductAttention::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_clone_with_new_inputs);
    return std::make_shared<ScaledDotProductAttention>(new_args, m_causal);
}

bool op::v13::ScaledDotProductAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v13_ScaledDotProductAttention_visit_attributes);
    visitor.on_attribute("causal", m_causal);
    return true;
}

}  // namespace ov
