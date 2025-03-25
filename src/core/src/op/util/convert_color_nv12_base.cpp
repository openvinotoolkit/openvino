// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_nv12_base.hpp"

#include "itt.hpp"
#include "nv12_shape_inference.hpp"

ov::op::util::ConvertColorNV12Base::ConvertColorNV12Base(const Output<Node>& arg, ColorConversion format)
    : Op({arg}),
      m_format(format) {}

ov::op::util::ConvertColorNV12Base::ConvertColorNV12Base(const Output<Node>& arg_y,
                                                         const Output<Node>& arg_uv,
                                                         ColorConversion format)
    : Op({arg_y, arg_uv}),
      m_format(format) {}

void ov::op::util::ConvertColorNV12Base::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Convert_NV12_Base_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    auto out_type = get_input_element_type(0);
    if (get_input_size() == 2) {
        const auto& uv_type = get_input_element_type(1);

        NODE_VALIDATION_CHECK(this,
                              element::Type::merge(out_type, out_type, uv_type),
                              "Y, UV inputs shall have compatible types, got ",
                              out_type,
                              uv_type);
    }
    NODE_VALIDATION_CHECK(this,
                          is_type_supported(out_type),
                          "Input type shall have u8 or floating-point precision, got ",
                          out_type);

    set_output_type(0, out_type, output_shapes.front());
}

bool ov::op::util::ConvertColorNV12Base::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool ov::op::util::ConvertColorNV12Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
