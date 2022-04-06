// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"
#include "multiclass_nms_shape_inference.hpp"

#include "itt.hpp"

using namespace ngraph;
using namespace op::util;

BWDCMP_RTTI_DEFINITION(op::v8::MulticlassNms);
BWDCMP_RTTI_DEFINITION(op::v9::MulticlassNms);

op::v8::MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores}, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(MulticlassNms_v8_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 2, "Number of inputs must be 2 at least");

    return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
}

op::v9::MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores}, attrs) {
    constructor_validate_and_infer_types();
}

op::v9::MulticlassNms::MulticlassNms(const Output<Node>& boxes,
                                     const Output<Node>& scores,
                                     const Output<Node>& roisnum,
                                     const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores, roisnum}, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v9::MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    NGRAPH_OP_SCOPE(MulticlassNms_v9_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    switch (new_args.size()) {
    case 3:
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    default:
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
    }
}

namespace {
inline bool is_float_type_admissible(const element::Type& t) {
    return t == element::f32 || t == element::f16 || t == element::bf16;
}
}  // namespace

bool op::v9::MulticlassNms::validate() {
    const auto& nms_attrs = this->get_attrs();
    const auto output_type = nms_attrs.output_type;
    const auto nms_top_k = nms_attrs.nms_top_k;
    const auto keep_top_k = nms_attrs.keep_top_k;

    // validate dtype of each input
    NODE_VALIDATION_CHECK(this,
                          output_type == element::i64 || output_type == element::i32,
                          "Output type must be i32 or i64");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(this->get_input_element_type(0)),
                          "Expected bf16, fp16 or fp32 as element type for the 'boxes' input.");

    NODE_VALIDATION_CHECK(this,
                          is_float_type_admissible(this->get_input_element_type(1)),
                          "Expected bf16, fp16 or fp32 as element type for the 'scores' input.");

    NODE_VALIDATION_CHECK(this,
                          this->get_input_element_type(0).compatible(this->get_input_element_type(1)),
                          "Expected 'boxes', 'scores' type is same.");

    if (this->get_input_size() == 3) {
        NODE_VALIDATION_CHECK(
            this,
            this->get_input_element_type(2) == element::i64 || this->get_input_element_type(2) == element::i32,
            "Expected i64 or i32 as element type for the 'roisnum' input.");
    }

    // validate attributes
    NODE_VALIDATION_CHECK(this, nms_top_k >= -1, "The 'nms_top_k' must be great or equal -1. Got:", nms_top_k);

    NODE_VALIDATION_CHECK(this, keep_top_k >= -1, "The 'keep_top_k' must be great or equal -1. Got:", keep_top_k);

    return true;
}

void op::v9::MulticlassNms::validate_and_infer_types() {
    NGRAPH_OP_SCOPE(MulticlassNms_v9_validate_and_infer_types);

    validate();

    const auto& boxes_ps = get_input_partial_shape(0);
    const auto& scores_ps = get_input_partial_shape(1);
    std::vector<PartialShape> input_shapes = {boxes_ps, scores_ps};
    if (get_input_size() == 3) {
        const auto& roisnum_ps = get_input_partial_shape(2);
        input_shapes.push_back(roisnum_ps);
    }

    std::vector<PartialShape> output_shapes = {{Dimension::dynamic(), 6},
                                               {Dimension::dynamic(), 1},
                                               {Dimension::dynamic()}};
    shape_infer(this, input_shapes, output_shapes, false, false);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, m_output_type, output_shapes[1]);
    set_output_type(2, m_output_type, output_shapes[2]);
}
