// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/multiclass_nms.hpp"

#include "itt.hpp"
#include "multiclass_nms_shape_inference.hpp"

using namespace ngraph;
using namespace op::util;

BWDCMP_RTTI_DEFINITION(op::v8::MulticlassNms);
BWDCMP_RTTI_DEFINITION(op::v9::MulticlassNms);

// ------------------------------ V8 ------------------------------

op::v8::MulticlassNms::MulticlassNms(const Output<Node>& boxes, const Output<Node>& scores, const Attributes& attrs)
    : MulticlassNmsBase({boxes, scores}, attrs) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v8::MulticlassNms::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(MulticlassNms_v8_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 2, "Number of inputs must be 2 at least");

    return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
}

void op::v8::MulticlassNms::validate_and_infer_types() {
    OV_OP_SCOPE(MulticlassNms_v9_validate_and_infer_types);
    const auto output_type = get_attrs().output_type;

    validate();

    const auto& boxes_ps = get_input_partial_shape(0);
    const auto& scores_ps = get_input_partial_shape(1);
    std::vector<PartialShape> input_shapes = {boxes_ps, scores_ps};
    std::vector<PartialShape> output_shapes = {{Dimension::dynamic(), 6},
                                               {Dimension::dynamic(), 1},
                                               {Dimension::dynamic()}};
    shape_infer(this, input_shapes, output_shapes, false, false);
    set_output_type(0, get_input_element_type(0), output_shapes[0]);
    set_output_type(1, output_type, output_shapes[1]);
    set_output_type(2, output_type, output_shapes[2]);
}

// ------------------------------ V9 ------------------------------

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
    OV_OP_SCOPE(MulticlassNms_v9_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() == 2 || new_args.size() == 3, "Number of inputs must be 2 or 3");

    switch (new_args.size()) {
    case 3:
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), new_args.at(2), m_attrs);
    default:
        return std::make_shared<MulticlassNms>(new_args.at(0), new_args.at(1), m_attrs);
    }
}

void op::v9::MulticlassNms::validate_and_infer_types() {
    OV_OP_SCOPE(MulticlassNms_v9_validate_and_infer_types);
    const auto output_type = get_attrs().output_type;

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
    set_output_type(1, output_type, output_shapes[1]);
    set_output_type(2, output_type, output_shapes[2]);
}
