// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngram.hpp"

#include "transformations/itt.hpp"

ov::intel_cpu::NgramNode::NgramNode(const ov::Output<Node>& embeddings,
                                    const ov::Output<Node>& batch_idces,
                                    const size_t k)
    : Op({embeddings, batch_idces}),
      m_k(k) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::NgramNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(NgramNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::NgramNode>(new_args.at(0), new_args.at(1), m_k);
}

bool ov::intel_cpu::NgramNode::visit_attributes(ov::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(NgramNode_visit_attributes);
    visitor.on_attribute("k", m_k);
    return true;
}

void ov::intel_cpu::NgramNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(NgramNode_validate_and_infer_types);
    OPENVINO_ASSERT(m_k > 0, "k attribute must be greater than zero");

    const auto& idces_et = get_input_element_type(1);
    const auto& idces_shape = get_input_partial_shape(1);
    OPENVINO_ASSERT(idces_shape.rank() == 2,
                    "'batch_idces' input must have 2D shape whereas current shape is",
                    idces_shape);
    OPENVINO_ASSERT(idces_et.is_integral_number(),
                    "'batch_idces' input must be integer whereas current element type is",
                    idces_et);

    const auto& embeddings_et = get_input_element_type(0);
    const auto& embeddings_shape = get_input_partial_shape(0);
    OPENVINO_ASSERT(embeddings_et.is_real(),
                    "'embeddings' input must be real whereas current element type is",
                    embeddings_et);
    OPENVINO_ASSERT(embeddings_shape.rank() == 2,
                    "'embeddings' input must have 2D shape whereas current shape is",
                    embeddings_shape);

    auto out_shape = embeddings_shape;
    out_shape[1] *= m_k;
    set_output_type(0, embeddings_et, out_shape);
}

size_t ov::intel_cpu::NgramNode::get_k() const {
    return m_k;
}
