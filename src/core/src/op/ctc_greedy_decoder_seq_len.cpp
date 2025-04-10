// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_greedy_decoder_seq_len.hpp"

#include "ctc_greedy_decoder_seq_len_shape_inference.hpp"
#include "itt.hpp"

namespace ov {

op::v6::CTCGreedyDecoderSeqLen::CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                                       const Output<Node>& seq_len,
                                                       const bool merge_repeated,
                                                       const element::Type& classes_index_type,
                                                       const element::Type& sequence_length_type)
    : Op({input, seq_len}),
      m_merge_repeated(merge_repeated),
      m_classes_index_type(classes_index_type),
      m_sequence_length_type(sequence_length_type) {
    constructor_validate_and_infer_types();
}

op::v6::CTCGreedyDecoderSeqLen::CTCGreedyDecoderSeqLen(const Output<Node>& input,
                                                       const Output<Node>& seq_len,
                                                       const Output<Node>& blank_index,
                                                       const bool merge_repeated,
                                                       const element::Type& classes_index_type,
                                                       const element::Type& sequence_length_type)
    : Op({input, seq_len, blank_index}),
      m_merge_repeated(merge_repeated),
      m_classes_index_type(classes_index_type),
      m_sequence_length_type(sequence_length_type) {
    constructor_validate_and_infer_types();
}

void op::v6::CTCGreedyDecoderSeqLen::validate_and_infer_types() {
    OV_OP_SCOPE(v6_CTCGreedyDecoderSeqLen_validate_and_infer_types);
    const auto& logits_pshape = get_input_partial_shape(0);
    const auto& seq_len_pshape = get_input_partial_shape(1);
    std::vector<ov::PartialShape> input_shapes = {logits_pshape, seq_len_pshape};
    // check optional input type: blank index
    if (get_input_size() == 3) {
        const auto& blank_index_type = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              blank_index_type.is_integral_number(),
                              "The blank index type is expected to be an integer type. Got: ",
                              blank_index_type);
        input_shapes.push_back(get_input_partial_shape(2));
    }

    const auto output_shapes = shape_infer(this, input_shapes);

    set_output_type(0, m_classes_index_type, output_shapes[0]);
    set_output_type(1, m_sequence_length_type, output_shapes[1]);
}

bool op::v6::CTCGreedyDecoderSeqLen::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v6_CTCGreedyDecoderSeqLen_visit_attributes);
    visitor.on_attribute("merge_repeated", m_merge_repeated);
    visitor.on_attribute("classes_index_type", m_classes_index_type);
    visitor.on_attribute("sequence_length_type", m_sequence_length_type);
    return true;
}

std::shared_ptr<Node> op::v6::CTCGreedyDecoderSeqLen::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v6_CTCGreedyDecoderSeqLen_clone_with_new_inputs);
    check_new_args_count(this, new_args);

    size_t args_size = new_args.size();
    if (args_size == 2) {
        return std::make_shared<CTCGreedyDecoderSeqLen>(new_args.at(0),
                                                        new_args.at(1),
                                                        m_merge_repeated,
                                                        m_classes_index_type,
                                                        m_sequence_length_type);
    } else if (args_size == 3) {
        return std::make_shared<CTCGreedyDecoderSeqLen>(new_args.at(0),
                                                        new_args.at(1),
                                                        new_args.at(2),
                                                        m_merge_repeated,
                                                        m_classes_index_type,
                                                        m_sequence_length_type);
    } else {
        OPENVINO_THROW("Incorrect number of arguments");
    }
}
}  // namespace ov
