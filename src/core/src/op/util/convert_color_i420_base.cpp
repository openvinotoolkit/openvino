// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_i420_base.hpp"

#include "i420_shape_inference.hpp"
#include "itt.hpp"

ov::op::util::ConvertColorI420Base::ConvertColorI420Base(const Output<Node>& arg, ColorConversion format)
    : Op({arg}),
      m_format(format) {}

ov::op::util::ConvertColorI420Base::ConvertColorI420Base(const Output<Node>& arg_y,
                                                         const Output<Node>& arg_u,
                                                         const Output<Node>& arg_v,
                                                         ColorConversion format)
    : Op({arg_y, arg_u, arg_v}),
      m_format(format) {}

void ov::op::util::ConvertColorI420Base::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Convert_I420_Base_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    const auto& y_type = get_input_element_type(0);
    auto out_type = y_type;

    if (get_input_size() == 3) {
        const auto& u_type = get_input_element_type(1);
        const auto& v_type = get_input_element_type(2);

        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, u_type),
                              "Y, U, V inputs shall have compatible types, got ",
                              y_type,
                              u_type,
                              v_type);
        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, v_type),
                              "Y, U, V inputs shall have compatible types, got ",
                              y_type,
                              u_type,
                              v_type);
    }
    NODE_VALIDATION_CHECK(this,
                          is_type_supported(out_type),
                          "Input type shall have u8 or floating-point precision, got ",
                          out_type);

    set_output_type(0, out_type, output_shapes.front());
}

bool ov::op::util::ConvertColorI420Base::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool ov::op::util::ConvertColorI420Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
