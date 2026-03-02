// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_to_nv12_base.hpp"

#include "itt.hpp"
#include "rgb_bgr_to_nv12_shape_inference.hpp"

ov::op::util::ConvertColorToNV12Base::ConvertColorToNV12Base(const Output<Node>& arg, ColorConversion format)
    : Op({arg}),
      m_format(format),
      m_single_plane(true) {}

ov::op::util::ConvertColorToNV12Base::ConvertColorToNV12Base(const Output<Node>& arg,
                                                             ColorConversion format,
                                                             bool single_plane)
    : Op({arg}),
      m_format(format),
      m_single_plane(single_plane) {}

void ov::op::util::ConvertColorToNV12Base::validate_and_infer_types() {
    OV_OP_SCOPE(v16_ConvertColorToNV12Base_validate_and_infer_types);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);

    auto out_type = get_input_element_type(0);

    NODE_VALIDATION_CHECK(this,
                          is_type_supported(out_type),
                          "Input type shall have u8 or floating-point precision, got ",
                          out_type);

    for (size_t i = 0; i < output_shapes.size(); i++) {
        set_output_type(i, out_type, output_shapes[i]);
    }
}

bool ov::op::util::ConvertColorToNV12Base::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("single_plane", m_single_plane);
    return true;
}

bool ov::op::util::ConvertColorToNV12Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
