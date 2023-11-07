// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "rope.hpp"

#include <algorithm>

#include "transformations/itt.hpp"

ov::intel_cpu::RoPENode::RoPENode(const OutputVector& args, const Config& cfg) : Op(args), m_config(cfg) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> ov::intel_cpu::RoPENode::clone_with_new_inputs(
    const ngraph::OutputVector& new_args) const {
    INTERNAL_OP_SCOPE(RoPENode_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ov::intel_cpu::RoPENode>(new_args, m_config);
}

void ov::intel_cpu::RoPENode::validate_and_infer_types() {
    INTERNAL_OP_SCOPE(RoPENode_validate_and_infer_types);
    auto input_pshape = get_input_partial_shape(0);
    auto input_slice_size = m_config.slice_stop - m_config.slice_start;
    if (input_slice_size > 0) {
        input_pshape[3] = input_slice_size;
    }
    if (m_config.input_trans0213) {
        // transpose 0213 ([B,L,H,S]=>[B,H,L,S]) happens before RoPE
        std::swap(input_pshape[2], input_pshape[1]);
    } else if (m_config.is_interleaved) {
        // transpose 0213 ([B,L,H,S]=>[B,H,L,S]) happens after RoPE
        std::swap(input_pshape[2], input_pshape[1]);
    }

    set_output_type(0, get_input_element_type(0), input_pshape);
}

bool ov::intel_cpu::RoPENode::visit_attributes(ngraph::AttributeVisitor& visitor) {
    INTERNAL_OP_SCOPE(RoPENode_visit_attributes);
    visitor.start_structure("config");
    visitor.on_attribute("slice_start", m_config.slice_start);
    visitor.on_attribute("slice_stop", m_config.slice_stop);
    visitor.on_attribute("input_trans0213", m_config.input_trans0213);
    visitor.on_attribute("is_interleaved", m_config.is_interleaved);
    visitor.on_attribute("ndims", m_config.ndims);
    visitor.on_attribute("gather_position_arg_id", m_config.gather_position_arg_id);
    visitor.finish_structure();
    return true;
}
