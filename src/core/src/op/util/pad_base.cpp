// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/validation_util.hpp>

#include "bound_evaluate.hpp"
#include "itt.hpp"
#include "ngraph/attribute_visitor.hpp"
#include "ngraph/except.hpp"
#include "ngraph/op/broadcast.hpp"
#include "ngraph/op/constant.hpp"
#include "ngraph/op/pad.hpp"
#include "ngraph/op/util/op_types.hpp"
#include "ngraph/runtime/reference/pad.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"
#include "pad_shape_inference.hpp"

using namespace std;
using namespace ngraph;

using namespace ov::op::util;

ov::op::util::PadBase::PadBase(const Output<Node>& arg,
                               const Output<Node>& pads_begin,
                               const Output<Node>& pads_end,
                               const Output<Node>& arg_pad_value,
                               PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, arg_pad_value}),
      m_pad_mode{pad_mode} {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
}

ov::op::util::PadBase::PadBase(const Output<Node>& arg,
                               const Output<Node>& pads_begin,
                               const Output<Node>& pads_end,
                               PadMode pad_mode)
    : Op({arg, pads_begin, pads_end, op::v0::Constant::create(arg.get_element_type(), ov::Shape{}, {0})}),
      m_pad_mode{pad_mode} {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
}

CoordinateDiff ov::op::util::PadBase::get_pads_begin() const {
    CoordinateDiff pads_begin_coord{};
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (auto pads_begin_const = get_constant_from_source(input_value(1))) {
        OPENVINO_SUPPRESS_DEPRECATED_END
        pads_begin_coord = pads_begin_const->cast_vector<ptrdiff_t>();
    }
    return pads_begin_coord;
}

CoordinateDiff ov::op::util::PadBase::get_pads_end() const {
    CoordinateDiff pads_end_coord{};
    OPENVINO_SUPPRESS_DEPRECATED_START
    if (auto pads_end_const = get_constant_from_source(input_value(2))) {
        OPENVINO_SUPPRESS_DEPRECATED_END
        pads_end_coord = pads_end_const->cast_vector<ptrdiff_t>();
    }
    return pads_end_coord;
}

bool ov::op::util::PadBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_PadBase_visit_attributes);
    visitor.on_attribute("pad_mode", m_pad_mode);
    return true;
}

void ov::op::util::PadBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_PadBase_validate_and_infer_types);
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

    OPENVINO_SUPPRESS_DEPRECATED_START
    const auto output_shapes = shape_infer(this, get_node_input_partial_shapes(*this));
    OPENVINO_SUPPRESS_DEPRECATED_END
    set_output_type(0, result_et, output_shapes[0]);
}

bool ov::op::util::PadBase::evaluate_pad(const HostTensorVector& outputs, const HostTensorVector& inputs) const {
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

bool ov::op::util::PadBase::evaluate_lower(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(util_PadBase_evaluate_lower);
    return ov::have_node_inputs_bounds_set(this, 1, 2) && ov::default_lower_bound_evaluator(this, output_values);
}

bool ov::op::util::PadBase::evaluate_upper(ov::TensorVector& output_values) const {
    OV_OP_SCOPE(util_PadBase_evaluate_upper);
    return ov::have_node_inputs_bounds_set(this, 1, 2) && ov::default_upper_bound_evaluator(this, output_values);
}

bool ov::op::util::PadBase::evaluate_label(ov::TensorLabelVector& output_labels) const {
    OV_OP_SCOPE(util_PadBase_evaluate_label);
    OPENVINO_SUPPRESS_DEPRECATED_START
    return ov::default_label_evaluator(this, output_labels);
    OPENVINO_SUPPRESS_DEPRECATED_END
}
