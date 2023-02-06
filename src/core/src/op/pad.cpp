// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/pad.hpp"

#include <ngraph/validation_util.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "pad_shape_inference.hpp"

using namespace std;
using namespace ngraph;

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 const Output<Node>& arg_pad_value,
                 PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, arg_pad_value}),
      m_pad_mode{pad_mode} {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    constructor_validate_and_infer_types();
}

op::v1::Pad::Pad(const Output<Node>& arg,
                 const Output<Node>& pads_begin,
                 const Output<Node>& pads_end,
                 PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, op::v0::Constant::create(arg.get_element_type(), ov::Shape{}, {0})}),
      m_pad_mode{pad_mode} {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
    constructor_validate_and_infer_types();
}

CoordinateDiff op::v1::Pad::get_pads_begin() const {
    CoordinateDiff pads_begin_coord{};
    if (auto pads_begin_const = get_constant_from_source(input_value(1))) {
        pads_begin_coord = pads_begin_const->cast_vector<ptrdiff_t>();
    }
    return pads_begin_coord;
}

CoordinateDiff op::v1::Pad::get_pads_end() const {
    CoordinateDiff pads_end_coord{};
    if (auto pads_end_const = get_constant_from_source(input_value(2))) {
        pads_end_coord = pads_end_const->cast_vector<ptrdiff_t>();
    }
    return pads_end_coord;
}

bool ngraph::op::v1::Pad::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_Pad_visit_attributes);
    visitor.on_attribute("pad_mode", m_pad_mode);
    return true;
}

void op::v1::Pad::validate_and_infer_types() {
    OV_OP_SCOPE(v1_Pad_validate_and_infer_types);
    element::Type result_et = element::dynamic;

    const auto& arg_element_type = get_input_element_type(0);
    const auto& pads_begin_element_type = get_input_element_type(1);
    const auto& pads_end_element_type = get_input_element_type(2);

    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, result_et, arg_element_type),
                          "Cannot merge element types (input arg element type: ",
                          arg_element_type,
                          ", with: ",
                          result_et,
                          ").");

    if (m_pad_mode == PadMode::CONSTANT && get_input_size() == 4) {
        const auto& arg_pad_element_type = get_input_element_type(3);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(result_et, result_et, arg_pad_element_type),
                              "Argument element types do not match (input arg element type: ",
                              arg_element_type,
                              ", arg_pad element type: ",
                              arg_pad_element_type,
                              ").");
    }

    NODE_VALIDATION_CHECK(this,
                          pads_begin_element_type.is_integral_number(),
                          "pads_begin must be an integral number, but is: ",
                          pads_begin_element_type,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          pads_end_element_type.is_integral_number(),
                          "pads_end must be an integral number, but is: ",
                          pads_end_element_type,
                          ").");

    const auto output_shapes = shape_infer(this, get_node_input_partial_shapes(*this));
    set_output_type(0, result_et, output_shapes[0]);
}

shared_ptr<Node> op::v1::Pad::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_Pad_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    if (get_input_size() == 4) {
        return make_shared<v1::Pad>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_pad_mode);
    } else {
        return make_shared<v1::Pad>(new_args.at(0), new_args.at(1), new_args.at(2), m_pad_mode);
    }
}

bool op::v1::Pad::evaluate_pad(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    const auto& data = inputs[0];
    const auto elem_size = data->get_element_type().size();

    const char* pad_value = nullptr;
    const std::vector<char> pad_zero_value(elem_size, 0);
    if (get_input_size() == 4) {
        pad_value = inputs[3]->get_data_ptr<char>();
    } else {
        pad_value = pad_zero_value.data();
    }

    // compute pads_begin and pads_end CoordinateDiffs from pads_begin
    // and pads_end shapes and reshape output to determine shape
    // (in case pads_begin and pads_end are Parameters, output is dynamic with static rank).

    op::v0::Constant pads_begin_const(inputs[1]);
    CoordinateDiff pads_begin_coord(pads_begin_const.cast_vector<ptrdiff_t>());
    op::v0::Constant pads_end_const(inputs[2]);
    CoordinateDiff pads_end_coord(pads_end_const.cast_vector<ptrdiff_t>());

    auto data_shape = data->get_shape();
    ov::Shape padded_shape(data_shape.size());
    for (size_t i = 0; i < data_shape.size(); ++i) {
        padded_shape[i] = data_shape[i] + pads_begin_coord[i] + pads_end_coord[i];
    }

    const auto& out = outputs[0];
    out->set_shape(padded_shape);

    ngraph::runtime::reference::pad(data->get_data_ptr<char>(),
                                    pad_value,
                                    out->get_data_ptr<char>(),
                                    elem_size,
                                    data->get_shape(),
                                    out->get_shape(),
                                    pads_begin_coord,
                                    pads_end_coord,
                                    get_pad_mode());

    return true;
}

bool op::v1::Pad::evaluate(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
    OV_OP_SCOPE(v1_Pad_evaluate);
    return evaluate_pad(outputs, inputs);
}

bool op::v1::Pad::has_evaluate() const {
    OV_OP_SCOPE(v1_Pad_has_evaluate);
    return true;
}

bool op::v1::Pad::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Pad_evaluate_lower);
    return ov::have_node_inputs_bounds_set(this, 1, 2) && ov::default_lower_bound_evaluator(this, output_values);
}

bool op::v1::Pad::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(v1_Pad_evaluate_upper);
    return ov::have_node_inputs_bounds_set(this, 1, 2) && ov::default_upper_bound_evaluator(this, output_values);
}

bool op::v1::Pad::evaluate_label(ov::TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(v1_Pad_evaluate_label);
    return ov::default_label_evaluator(this, output_labels);
}
