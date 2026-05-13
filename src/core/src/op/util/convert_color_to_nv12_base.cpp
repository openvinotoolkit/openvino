// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_to_nv12_base.hpp"


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



bool ov::op::util::ConvertColorToNV12Base::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("single_plane", m_single_plane);
    return true;
}

bool ov::op::util::ConvertColorToNV12Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
