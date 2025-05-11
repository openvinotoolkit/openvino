// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/lrn.hpp"

#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"

namespace ov {
op::v0::LRN::LRN(const Output<Node>& arg, double alpha, double beta, double bias, size_t size)
    : LRN(arg, op::v0::Constant::create(element::i64, ov::Shape{1}, {1}), alpha, beta, bias, size) {}

op::v0::LRN::LRN(const Output<Node>& arg, const Output<Node>& axes, double alpha, double beta, double bias, size_t size)
    : Op({arg, axes}),
      m_alpha(alpha),
      m_beta(beta),
      m_bias(bias),
      m_size(size) {
    constructor_validate_and_infer_types();
}

AxisSet op::v0::LRN::get_reduction_axes() const {
    AxisSet axes{1};  // channel axis as default
    auto axes_input_node = input_value(1).get_node_shared_ptr();
    if (const auto& const_op = ov::util::get_constant_from_source(axes_input_node)) {
        axes = const_op->get_axis_set_val();
    }
    return axes;
}

void op::v0::LRN::validate_and_infer_types() {
    OV_OP_SCOPE(v0_LRN_validate_and_infer_types);
    element::Type arg_type = get_input_element_type(0);
    ov::PartialShape arg_shape = get_input_partial_shape(0);
    set_output_type(0, arg_type, arg_shape);

    const ov::PartialShape& input_shape = get_input_partial_shape(0);
    const auto input_shape_rank = input_shape.rank();

    ov::PartialShape axes_shape{ov::PartialShape::dynamic()};
    if (get_input_partial_shape(1).is_static()) {
        axes_shape = get_input_partial_shape(1);
    }

    auto axes_rank = axes_shape.rank();
    NODE_VALIDATION_CHECK(this,
                          axes_rank.compatible(1),
                          "Input axes must have rank equals 1 (axes_rank: ",
                          axes_rank,
                          ").");

    NODE_VALIDATION_CHECK(this,
                          axes_shape.is_dynamic() || input_shape_rank.is_dynamic() ||
                              axes_shape[0].get_length() <= input_shape_rank.get_length(),
                          "Number of elements of axes must be >= 0 and <= argument rank (axes_shape[0]: ",
                          axes_shape[0],
                          ").");

    if (input_shape_rank.is_static()) {
        const auto reduction_axes = get_reduction_axes();
        for (auto axis : reduction_axes) {
            NODE_VALIDATION_CHECK(this,
                                  static_cast<int64_t>(axis) < input_shape_rank.get_length(),
                                  "Reduction axis (",
                                  axis,
                                  ") is out of bounds ",
                                  "(argument shape: ",
                                  input_shape,
                                  ", reduction axes: ",
                                  reduction_axes,
                                  ")");
        }
    }

    const auto& axes_type = get_input_element_type(1);
    NODE_VALIDATION_CHECK(this,
                          axes_type.is_integral_number(),
                          "Axes input must be integral numbers, but are: ",
                          axes_type,
                          ").");
}

bool op::v0::LRN::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v0_LRN_visit_attributes);
    visitor.on_attribute("alpha", m_alpha);
    visitor.on_attribute("beta", m_beta);
    visitor.on_attribute("bias", m_bias);
    visitor.on_attribute("size", m_size);
    return true;
}

std::shared_ptr<Node> op::v0::LRN::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v0_LRN_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<op::v0::LRN>(new_args.at(0), new_args.at(1), m_alpha, m_beta, m_bias, m_size);
}
}  // namespace ov
