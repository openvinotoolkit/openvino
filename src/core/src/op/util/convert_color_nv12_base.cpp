// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_nv12_base.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "openvino/core/layout.hpp"

static const size_t N_DIM = 0;
static const size_t H_DIM = 1;
static const size_t W_DIM = 2;
static const size_t C_DIM = 3;

ov::op::util::ConvertColorNV12Base::ConvertColorNV12Base(const Output<Node>& arg, ColorConversion format)
    : Op({arg}),
      m_format(format) {}

ov::op::util::ConvertColorNV12Base::ConvertColorNV12Base(const Output<Node>& arg_y,
                                                         const Output<Node>& arg_uv,
                                                         ColorConversion format)
    : Op({arg_y, arg_uv}),
      m_format(format) {
    constructor_validate_and_infer_types();
}

void ov::op::util::ConvertColorNV12Base::validate_and_infer_types() {
    OV_OP_SCOPE(v8_Convert_NV12_Base_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 1 || get_input_size() == 2,
                          "NV12 conversion shall have one or 2 inputs, but it is ",
                          get_input_size());
    auto single_plane = get_input_size() == 1;
    auto y_type = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          is_type_supported(y_type),
                          "Y input shall have u8 or floating-point precision, got ",
                          y_type);
    const auto& shape_y = get_input_partial_shape(0);
    if (shape_y.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              shape_y.rank().get_length() == 4,
                              "Y input with static shape shall have 4 dimensions (N, H, W, C)");

        NODE_VALIDATION_CHECK(this,
                              shape_y[C_DIM].is_dynamic() || shape_y[C_DIM].get_length() == 1,
                              "Y channels dimension shall be either dynamic or equal to 1. Current value is ",
                              shape_y[C_DIM].get_length());
    }
    auto out_shape = shape_y;
    auto out_type = y_type;
    if (out_shape.rank().is_dynamic()) {
        out_shape = PartialShape{Dimension::dynamic(), Dimension::dynamic(), Dimension::dynamic(), 3};
    }
    out_shape[C_DIM] = 3;  // 3 is number of channels (R, G, B)
    if (single_plane) {
        if (shape_y.rank().is_static() && shape_y[H_DIM].is_static()) {
            NODE_VALIDATION_CHECK(this,
                                  shape_y[H_DIM].get_length() % 3 == 0,
                                  "NV12 image height shall be divisible by 3, but it is ",
                                  shape_y[H_DIM].get_length());
            // E.g. if input shape height is 720 for NV12, then real image height is 720 * 2 / 3 = 480
            out_shape[H_DIM] = shape_y[H_DIM].get_length() * 2 / 3;
        }
    } else {
        auto uv_type = get_input_element_type(1);
        if (y_type.is_dynamic()) {
            NODE_VALIDATION_CHECK(this,
                                  is_type_supported(uv_type),
                                  "UV input shall have u8 or floating-point precision, got ",
                                  uv_type);
            out_type = uv_type;
        } else {
            NODE_VALIDATION_CHECK(this,
                                  uv_type.is_dynamic() || uv_type == y_type,
                                  "UV input ",
                                  uv_type,
                                  " shall have same precision as Y input ",
                                  y_type);
        }
        const auto& shape_uv = get_input_partial_shape(1);
        NODE_VALIDATION_CHECK(this,
                              shape_uv.rank().is_dynamic() || shape_uv.rank().get_length() == 4,
                              "UV input with static shape shall have 4 dimensions (N, H, W, C)");
        if (shape_y.rank().is_static() && shape_uv.rank().is_static()) {
            // Verify that height for Y input is 2 times bigger than input height for UV
            NODE_VALIDATION_CHECK(this,
                                  shape_y[H_DIM].is_dynamic() || shape_uv[H_DIM].is_dynamic() ||
                                      shape_y[H_DIM].get_length() == shape_uv[H_DIM].get_length() * 2,
                                  "Y input height shall be 2 times bigger that UV input height: Y height = ",
                                  shape_y[H_DIM].get_length(),
                                  " UV height = ",
                                  shape_uv[H_DIM].get_length());
            // Verify that width for Y input is 2 times bigger than input width for UV
            NODE_VALIDATION_CHECK(this,
                                  shape_y[W_DIM].is_dynamic() || shape_uv[W_DIM].is_dynamic() ||
                                      shape_y[W_DIM].get_length() == shape_uv[W_DIM].get_length() * 2,
                                  "Y input width shall be 2 times bigger that UV input width: Y width = ",
                                  shape_y[W_DIM].get_length(),
                                  " UV width = ",
                                  shape_uv[W_DIM].get_length());
            NODE_VALIDATION_CHECK(this,
                                  shape_uv[C_DIM].is_dynamic() || shape_uv[C_DIM].get_length() == 2,
                                  "UV channels dimension shall be either dynamic or equal to 2. Current value is ",
                                  shape_uv[C_DIM].get_length());

            NODE_VALIDATION_CHECK(this,
                                  shape_y[N_DIM].is_dynamic() || shape_uv[N_DIM].is_dynamic() ||
                                      shape_y[N_DIM].get_length() == shape_uv[N_DIM].get_length(),
                                  "Y input batch shall be same as UV input batch: Y batch = ",
                                  shape_y[N_DIM].get_length(),
                                  " UV batch = ",
                                  shape_uv[N_DIM].get_length());
        }
        // Set shape based on UV shape, if Y are dynamic
        if (shape_uv.rank().is_static()) {
            if (out_shape[N_DIM].is_dynamic()) {
                out_shape[N_DIM] = shape_uv[N_DIM];
            }
            if (out_shape[H_DIM].is_dynamic()) {
                out_shape[H_DIM] = shape_uv[H_DIM] * 2;
            }
            if (out_shape[W_DIM].is_dynamic()) {
                out_shape[W_DIM] = shape_uv[W_DIM] * 2;
            }
        }
    }
    NODE_VALIDATION_CHECK(this,
                          out_shape[H_DIM].is_dynamic() || out_shape[H_DIM].get_length() % 2 == 0,
                          "Image height must be even, but it is ",
                          out_shape[H_DIM].get_length());
    NODE_VALIDATION_CHECK(this,
                          out_shape[W_DIM].is_dynamic() || out_shape[W_DIM].get_length() % 2 == 0,
                          "Image width must be even, but it is ",
                          out_shape[W_DIM].get_length());
    set_output_type(0, out_type, out_shape);
}

bool ov::op::util::ConvertColorNV12Base::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool ov::op::util::ConvertColorNV12Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
