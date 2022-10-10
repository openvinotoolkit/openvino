// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/reverse_sequence.hpp"

#include <algorithm>
#include <memory>
#include <reverse_sequence_shape_inference.hpp>

#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/node.hpp"
#include "ngraph/validation_util.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v0::ReverseSequence);

op::ReverseSequence::ReverseSequence(const Output<Node>& arg,
                                     const Output<Node>& seq_indices,
                                     int64_t batch_axis,
                                     int64_t seq_axis)
    : Op({arg, seq_indices}),
      m_batch_axis(batch_axis),
      m_seq_axis(seq_axis),
      m_normalized_batch_axis{0},
      m_normalized_seq_axis{0} {
    constructor_validate_and_infer_types();
}

bool ngraph::op::v0::ReverseSequence::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_ReverseSequence_visit_attributes);
    visitor.on_attribute("batch_axis", m_batch_axis);
    visitor.on_attribute("seq_axis", m_seq_axis);
    return true;
}

void op::ReverseSequence::validate_and_infer_types() {
    OV_OP_SCOPE(v0_ReverseSequence_validate_and_infer_types);
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& seq_lengths_et = get_input_element_type(1);
    const auto& seq_lengths_pshape = get_input_partial_shape(1);
    normalize_axes(this);
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {data_pshape, seq_lengths_pshape};
    shape_infer(this, input_shapes, output_shapes);
    NODE_VALIDATION_CHECK(this,
                          seq_lengths_et.is_real() || seq_lengths_et.is_integral_number(),
                          "Sequence lengths element type must be numeric type. Got: ",
                          seq_lengths_et);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
}

shared_ptr<Node> op::ReverseSequence::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_ReverseSequence_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<ReverseSequence>(new_args.at(0), new_args.at(1), m_batch_axis, m_seq_axis);
}

void op::ReverseSequence::set_batch_axis(int64_t batch_axis) {
    m_batch_axis = batch_axis;
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& data_rank = data_pshape.rank();
    m_normalized_batch_axis = ov::normalize_axis(this, m_batch_axis, data_rank);
}

void op::ReverseSequence::set_sequence_axis(int64_t sequence_axis) {
    m_seq_axis = sequence_axis;
    const auto& data_pshape = get_input_partial_shape(0);
    const auto& data_rank = data_pshape.rank();
    m_normalized_seq_axis = ov::normalize_axis(this, m_seq_axis, data_rank);
}

void op::ReverseSequence::normalize_axes(ReverseSequence* op) {
    const auto& data_pshape = op->get_input_partial_shape(0);
    const auto& data_rank = data_pshape.rank();
    NODE_VALIDATION_CHECK(op,
                          data_rank.is_dynamic() || data_rank.get_length() >= 2,
                          "Data input rank should be equal or greater than 2. Got: ",
                          data_pshape);
    op->m_normalized_batch_axis = ov::normalize_axis(op, op->m_batch_axis, data_rank);
    op->m_normalized_seq_axis = ov::normalize_axis(op, op->m_seq_axis, data_rank);
}
