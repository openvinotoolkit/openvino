// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/reverse_sequence.hpp"

#include <algorithm>
#include <memory>

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "reverse_sequence_shape_inference.hpp"

namespace ov {
op::v0::ReverseSequence::ReverseSequence(const Output<Node>& arg,
                                         const Output<Node>& seq_indices,
                                         int64_t batch_axis,
                                         int64_t seq_axis)
    : Op({arg, seq_indices}),
      m_batch_axis(batch_axis),
      m_seq_axis(seq_axis) {
    constructor_validate_and_infer_types();
}

bool op::v0::ReverseSequence::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ReverseSequence_visit_attributes);
    visitor.on_attribute("batch_axis", m_batch_axis);
    visitor.on_attribute("seq_axis", m_seq_axis);
    return true;
}

void op::v0::ReverseSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ReverseSequence_validate_and_infer_types);
    const auto& seq_lengths_et = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          seq_lengths_et.is_real() || seq_lengths_et.is_integral_number(),
                          "Sequence lengths element type must be numeric type. Got: ",
                          seq_lengths_et);

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, get_input_element_type(0), output_shapes[0]);

    const auto data_rank = get_input_partial_shape(0).rank();

    if (data_rank.is_static()) {
        m_normalized_seq_axis = ov::util::try_normalize_axis(m_seq_axis, data_rank, *this);
    } else {
        NODE_VALIDATION_CHECK(this,
                              m_seq_axis >= 0,
                              "Rank must be static in order to normalize negative axis: ",
                              m_seq_axis);
    }
}

std::shared_ptr<Node> op::v0::ReverseSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ReverseSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<ReverseSequence>(new_args.at(0), new_args.at(1), m_batch_axis, m_seq_axis);
}

void op::v0::ReverseSequence::set_batch_axis(int64_t batch_axis) {
    m_batch_axis = batch_axis;
}

size_t op::v0::ReverseSequence::get_batch_axis() const {
    const auto& data_rank = get_input_partial_shape(0).rank();
    return ov::util::try_normalize_axis(m_batch_axis, data_rank, *this);
}

void op::v0::ReverseSequence::set_sequence_axis(int64_t sequence_axis) {
    m_seq_axis = sequence_axis;
    m_normalized_seq_axis = ov::util::try_normalize_axis(m_seq_axis, get_input_partial_shape(0).rank(), *this);
}
}  // namespace ov
