// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/ctc_greedy_decoder.hpp"

#include <ctc_greedy_decoder_shape_inference.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(ov::op::v0::CTCGreedyDecoder);

op::CTCGreedyDecoder::CTCGreedyDecoder(const Output<Node>& input,
                                       const Output<Node>& seq_len,
                                       const bool ctc_merge_repeated)
    : Op({input, seq_len}),
      m_ctc_merge_repeated(ctc_merge_repeated) {
    constructor_validate_and_infer_types();
}

void op::CTCGreedyDecoder::validate_and_infer_types() {
    OV_OP_SCOPE(v0_CTCGreedyDecoder_validate_and_infer_types);
    const auto& logits_pshape = get_input_partial_shape(0);
    const auto& seq_mask_pshape = get_input_partial_shape(1);
    const auto& input_et = get_input_element_type(0);

    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};
    std::vector<ov::PartialShape> input_shapes = {logits_pshape, seq_mask_pshape};
    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, input_et, output_shapes[0]);
}

bool op::CTCGreedyDecoder::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_CTCGreedyDecoder_visit_attributes);
    visitor.on_attribute("ctc_merge_repeated", m_ctc_merge_repeated);
    return true;
}

shared_ptr<Node> op::CTCGreedyDecoder::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_CTCGreedyDecoder_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return make_shared<CTCGreedyDecoder>(new_args.at(0), new_args.at(1), m_ctc_merge_repeated);
}
