// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/ctc_loss.hpp"

#include <ctc_loss_shape_inference.hpp>

#include "itt.hpp"

using namespace std;
using namespace ngraph;

BWDCMP_RTTI_DEFINITION(op::v4::CTCLoss);

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
    const auto& logit_length_type = get_input_element_type(1);
    const auto& labels_type = get_input_element_type(2);
    const auto& label_length_type = get_input_element_type(3);

    NODE_VALIDATION_CHECK(this,
                          logits_type.is_real(),
                          "The data type for logits is expected to be a floating point type. Got: ",
                          logits_type);

    NODE_VALIDATION_CHECK(this,
                          logit_length_type.is_integral_number(),
                          "The logit length type is expected to be an integer type. Got: ",
                          logit_length_type);

    NODE_VALIDATION_CHECK(this,
                          labels_type.is_integral_number(),
                          "The labels type is expected to be an integer type. Got: ",
                          labels_type);

    NODE_VALIDATION_CHECK(this,
                          label_length_type.is_integral_number(),
                          "The label length type is expected to be an integer type. Got: ",
                          label_length_type);

    // check optional input type: blank index
    if (get_input_size() == 5) {
        const auto& blank_index_type = get_input_element_type(4);
        NODE_VALIDATION_CHECK(this,
                              blank_index_type.is_integral_number(),
                              "The blank index type is expected to be an integer type. Got: ",
                              blank_index_type);
    }

    const auto& logits_pshape = get_input_partial_shape(0);
    const auto& logit_length_pshape = get_input_partial_shape(1);
    const auto& labels_pshape = get_input_partial_shape(2);
    const auto& label_length_pshape = get_input_partial_shape(3);

    std::vector<ov::PartialShape> input_shapes;
    if (get_input_size() == 5) {
        const auto& blank_index_pshape = get_input_partial_shape(4);
        input_shapes = {logits_pshape, logit_length_pshape, labels_pshape, label_length_pshape, blank_index_pshape};
    } else {
        input_shapes = {logits_pshape, logit_length_pshape, labels_pshape, label_length_pshape};
    }
    std::vector<ov::PartialShape> output_shapes = {ov::PartialShape{}};

    shape_infer(this, input_shapes, output_shapes);
    set_output_type(0, logits_type, output_shapes[0]);
}

bool op::v4::CTCLoss::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v4_CTCLoss_visit_attributes);
    visitor.on_attribute("preprocess_collapse_repeated", preprocess_collapse_repeated_);
    visitor.on_attribute("ctc_merge_repeated", ctc_merge_repeated_);
    visitor.on_attribute("unique", unique_);
    return true;
}

shared_ptr<Node> op::v4::CTCLoss::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v4_CTCLoss_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (new_args.size() == 4) {
        return make_shared<CTCLoss>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    preprocess_collapse_repeated_,
                                    ctc_merge_repeated_,
                                    unique_);
    } else if (new_args.size() == 5) {
        return make_shared<CTCLoss>(new_args.at(0),
                                    new_args.at(1),
                                    new_args.at(2),
                                    new_args.at(3),
                                    new_args.at(4),
                                    preprocess_collapse_repeated_,
                                    ctc_merge_repeated_,
                                    unique_);
    } else {
        throw ngraph_error("Incorrect number of arguments");
    }
}
