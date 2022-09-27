// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_i420_base.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "openvino/core/layout.hpp"

namespace i420_op {
static const size_t H_DIM = 1;
static const size_t W_DIM = 2;
static const size_t C_DIM = 3;
}  // namespace i420_op

ov::op::util::ConvertColorI420Base::ConvertColorI420Base(const Output<Node>& arg, ColorConversion format)
    : Op({arg}),
      m_format(format) {}

ov::op::util::ConvertColorI420Base::ConvertColorI420Base(const Output<Node>& arg_y,
                                                         const Output<Node>& arg_u,
                                                         const Output<Node>& arg_v,
                                                         ColorConversion format)
    : Op({arg_y, arg_u, arg_v}),
      m_format(format) {
    constructor_validate_and_infer_types();
}

void ov::op::util::ConvertColorI420Base::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Convert_I420_Base_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 1 || get_input_size() == 3,
                          "I420 conversion shall have one or 3 inputs, but it is ",
                          get_input_size());
    auto single_plane = get_input_size() == 1;
    auto y_type = get_input_element_type(0);
    const auto& shape_y = get_input_partial_shape(0);
    const auto one_channel_nhwc_shape =
        PartialShape({Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 1});
    NODE_VALIDATION_CHECK(this,
                          shape_y.compatible(one_channel_nhwc_shape),
                          "Y input shall have 4 dimensions (N, H, W, C) with channels dimension equal to 1");
    auto out_shape = shape_y;
    auto out_type = y_type;
    if (out_shape.rank().is_dynamic()) {
        out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3};
    }
    out_shape[i420_op::C_DIM] = 3;  // 3 is number of channels (R, G, B)
    if (single_plane) {
        if (shape_y.rank().is_static() && shape_y[i420_op::H_DIM].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  shape_y[i420_op::H_DIM].get_length() % 3 == 0,
                                  "I420 image height shall be divisible by 3, but it is ",
                                  shape_y[i420_op::H_DIM].get_length());
            // E.g. if input shape height is 720 for I420, then real image height is 720 * 2 / 3 = 480
            out_shape[i420_op::H_DIM] = shape_y[i420_op::H_DIM].get_length() * 2 / 3;
        }
    } else {
        auto u_type = get_input_element_type(1);
        auto v_type = get_input_element_type(2);
        NODE_VALIDATION_CHECK(this,
                              ov::element::Type::merge(out_type, out_type, u_type),
                              "Y, U, V inputs shall have compatible types, got ",
                              y_type,
                              u_type,
                              v_type);
        NODE_VALIDATION_CHECK(this,
                              ov::element::Type::merge(out_type, out_type, v_type),
                              "Y, U, V inputs shall have compatible types, got ",
                              y_type,
                              u_type,
                              v_type);
        // Validate Y/U/V shapes compatibility
        const auto& shape_u = get_input_partial_shape(1);
        NODE_VALIDATION_CHECK(this,
                              shape_u.compatible(one_channel_nhwc_shape),
                              "U input shall have 4 dimensions (N, H, W, C) with channels dimension equal to 1, got ",
                              shape_u);
        const auto& shape_v = get_input_partial_shape(2);
        NODE_VALIDATION_CHECK(this,
                              shape_v.compatible(one_channel_nhwc_shape),
                              "V input shall have 4 dimensions (N, H, W, C) with channels dimension equal to 1, got ",
                              shape_v);
        NODE_VALIDATION_CHECK(this,
                              shape_u.compatible(shape_v),
                              "U shape shall be compatible with V shape: ",
                              shape_u,
                              shape_v);
        auto shape_uv = shape_u;
        PartialShape::merge_into(shape_uv, shape_v);
        if (shape_uv.rank().is_static()) {
            shape_uv[i420_op::H_DIM] *= 2;
            shape_uv[i420_op::W_DIM] *= 2;
        }
        NODE_VALIDATION_CHECK(this,
                              shape_y.compatible(shape_uv),
                              "Y shape is inconsistent with U and V shapes: ",
                              shape_y,
                              shape_u,
                              shape_v);
        PartialShape::merge_into(out_shape, shape_uv);
    }
    NODE_VALIDATION_CHECK(this,
                          out_shape[i420_op::H_DIM].is_dynamic() || out_shape[i420_op::H_DIM].get_length() % 2 == 0,
                          "Image height must be even, but it is ",
                          out_shape[i420_op::H_DIM].get_length());
    NODE_VALIDATION_CHECK(this,
                          out_shape[i420_op::W_DIM].is_dynamic() || out_shape[i420_op::W_DIM].get_length() % 2 == 0,
                          "Image width must be even, but it is ",
                          out_shape[i420_op::W_DIM].get_length());
    NODE_VALIDATION_CHECK(this,
                          is_type_supported(out_type),
                          "Input type shall have u8 or floating-point precision, got ",
                          out_type);
    set_output_type(0, out_type, out_shape);
}

bool ov::op::util::ConvertColorI420Base::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool ov::op::util::ConvertColorI420Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
