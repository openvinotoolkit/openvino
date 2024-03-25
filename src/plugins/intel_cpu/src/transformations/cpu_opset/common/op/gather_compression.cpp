// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gather_compression.hpp"
#include "transformations/itt.hpp"

ov::intel_cpu::GatherCompressionNode::GatherCompressionNode(const ov::Output<Node>& data,
                                                            const ov::Output<Node>& zp_compressed,
                                                            const ov::Output<Node>& scale_compressed,
                                                            const ov::Output<Node>& indices,
                                                            const ov::element::Type output_type)
    : Op({data, zp_compressed, scale_compressed, indices}),
      m_output_type(output_type) {
    validate_and_infer_types();
}

std::shared_ptr<ov::Node> ov::intel_cpu::GatherCompressionNode::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(GatherCompressionNode_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    return std::make_shared<ov::intel_cpu::GatherCompressionNode>(new_args.at(0),
                                                                  new_args.at(1),
                                                                  new_args.at(2),
                                                                  new_args.at(3),
                                                                  m_output_type);
}

void ov::intel_cpu::GatherCompressionNode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(GatherCompressionNode_validate_and_infer_types);
    const auto input_size = get_input_size();
    NODE_VALIDATION_CHECK(this,
        input_size == 4,
        "Number of inputs is incorrect. Current value is: ",
        input_size,
        ", expected: 4.");

    // Weights/ZP/Scale shape
    const auto weights_pshape = get_input_partial_shape(0);
    NODE_VALIDATION_CHECK(this, weights_pshape.is_static(), "Weights pshape must be static");
    const auto weights_shape = weights_pshape.to_shape();

    const auto zp_pshape = get_input_partial_shape(1);
    NODE_VALIDATION_CHECK(this, zp_pshape.is_static(), "ZP pshape must be static");
    const auto zp_shape = zp_pshape.to_shape();

    const auto scale_pshape = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this, scale_pshape.is_static(), "Scale pshape must be static");
    const auto scale_shape = scale_pshape.to_shape();

    NODE_VALIDATION_CHECK(this, weights_pshape.size() == 2u || weights_pshape.size() == 3u, "Weights rank must be equal 2 or 3");
    NODE_VALIDATION_CHECK(this, zp_pshape.size() == 2u || zp_pshape.size() == 3u || zp_pshape.size() == 1u, "ZP rank must be equal 2 or 3");
    NODE_VALIDATION_CHECK(this, scale_pshape.size() == 2u || scale_pshape.size() == 3u, "Scale rank must be equal 2 or 3");
    NODE_VALIDATION_CHECK(this, scale_pshape[0] == weights_pshape[0], "Weights and scale dim 0 must be same");
    NODE_VALIDATION_CHECK(this, zp_pshape[zp_pshape.size() - 1] == 1u, "ZP last dim must be equal 1");
    NODE_VALIDATION_CHECK(this, scale_pshape[scale_pshape.size() - 1] == 1u, "Scale last dim must be equal 1");

    // Index
    const auto index_pshape = get_input_partial_shape(3);

    // Result shape
    ov::PartialShape output_pshape = index_pshape;
    output_pshape.push_back(weights_shape.size() == 2u ? weights_shape[1] : (weights_shape[1] * weights_shape[2]));

    auto output_type = m_output_type == ov::element::undefined ? ov::element::f32 : m_output_type;
    set_output_type(0, output_type, output_pshape);
}

bool ov::intel_cpu::GatherCompressionNode::visit_attributes(ov::AttributeVisitor &visitor) {
    INTERNAL_OP_SCOPE(GatherCompressionNode_visit_attributes);
    visitor.on_attribute("out-type", m_output_type);
    return true;
}
