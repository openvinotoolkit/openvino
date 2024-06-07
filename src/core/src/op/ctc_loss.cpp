// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/ctc_loss.hpp"

#include "ctc_loss_shape_inference.hpp"
#include "itt.hpp"

namespace ov {
op::v4::CTCLoss::CTCLoss(const Output<Node>& logits,
                         const Output<Node>& logit_length,
                         const Output<Node>& labels,
                         const Output<Node>& label_length,
                         const bool preprocess_collapse_repeated,
                         const bool ctc_merge_repeated,
                         const bool unique)
    : Op({logits, logit_length, labels, label_length}),
      preprocess_collapse_repeated_(preprocess_collapse_repeated),
      ctc_merge_repeated_(ctc_merge_repeated),
      unique_(unique) {
    constructor_validate_and_infer_types();
}

op::v4::CTCLoss::CTCLoss(const Output<Node>& logits,
                         const Output<Node>& logit_length,
                         const Output<Node>& labels,
                         const Output<Node>& label_length,
                         const Output<Node>& blank_index,
                         const bool preprocess_collapse_repeated,
                         const bool ctc_merge_repeated,
                         const bool unique)
    : Op({logits, logit_length, labels, label_length, blank_index}),
      preprocess_collapse_repeated_(preprocess_collapse_repeated),
      ctc_merge_repeated_(ctc_merge_repeated),
      unique_(unique) {
    constructor_validate_and_infer_types();
}

void op::v4::CTCLoss::validate_and_infer_types() {
    OV_OP_SCOPE(v4_CTCLoss_validate_and_infer_types);
    // check types of input tensors
    const auto& logits_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          logits_type.is_real(),
                          "The data type for ",
                          ctc_loss::shape_names[0],
                          " is expected to be a floating point type. Got: ",
                          logits_type);

    for (size_t i = 1; i < get_input_size(); ++i) {
        const auto& input_et = get_input_element_type(i);
        NODE_VALIDATION_CHECK(this,
                              input_et.is_integral_number(),
                              "The ",
                              ctc_loss::shape_names[i],
                              " type is expected to be an integer type. Got: ",
                              input_et);
    }

    const auto output_shapes = shape_infer(this, ov::util::get_node_input_partial_shapes(*this));
    set_output_type(0, logits_type, output_shapes[0]);
}

bool op::v4::CTCLoss::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_CTCLoss_visit_attributes);
    visitor.on_attribute("preprocess_collapse_repeated", preprocess_collapse_repeated_);
    visitor.on_attribute("ctc_merge_repeated", ctc_merge_repeated_);
    visitor.on_attribute("unique", unique_);
    return true;
}

std::shared_ptr<Node> op::v4::CTCLoss::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_CTCLoss_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return std::make_shared<CTCLoss>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         preprocess_collapse_repeated_,
                                         ctc_merge_repeated_,
                                         unique_);
    } else if (new_args.size() == 5) {
        return std::make_shared<CTCLoss>(new_args.at(0),
                                         new_args.at(1),
                                         new_args.at(2),
                                         new_args.at(3),
                                         new_args.at(4),
                                         preprocess_collapse_repeated_,
                                         ctc_merge_repeated_,
                                         unique_);
    } else {
        OPENVINO_THROW("Incorrect number of arguments");
    }
}
}  // namespace ov
