// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_nv12_base.hpp"

#include <memory>
#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/convert_color_nv12.hpp"
#include "openvino/core/layout.hpp"

static const size_t N_DIM = 0;
static const size_t H_DIM = 1;
static const size_t W_DIM = 2;
static const size_t C_DIM = 3;

BWDCMP_RTTI_DEFINITION(ov::op::util::ConvertColorNV12Base);

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
    NGRAPH_OP_SCOPE(v8_Convert_NV12_Base_validate_and_infer_types);

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

namespace color_convert_nv12_op {

template <ov::element::Type_t ET>
inline bool evaluate(const ov::HostTensorVector& input_values,
                     const ov::HostTensorPtr& output_value,
                     bool single_tensor,
                     ov::op::util::ConvertColorNV12Base::ColorConversion color_format) {
    using namespace ov::op::util;
    const auto& y_tensor = input_values[0];
    auto batch_size = y_tensor->get_shape()[N_DIM];
    auto image_w = y_tensor->get_shape()[W_DIM];
    auto image_h = y_tensor->get_shape()[H_DIM];
    if (single_tensor) {
        OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(input_values, 1));
        image_h = image_h * 2 / 3;
    } else {
        OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(input_values, 2));
    }
    output_value->set_shape({batch_size, image_h, image_w, 3});  // 3 is RGB
    if (single_tensor) {
        ngraph::runtime::reference::color_convert_nv12(y_tensor->get_data_ptr<ET>(),
                                                       y_tensor->get_data_ptr<ET>() + image_w * image_h,
                                                       output_value->get_data_ptr<ET>(),
                                                       batch_size,
                                                       image_h,
                                                       image_w,
                                                       image_w * image_h * 3 / 2,
                                                       image_w * image_h * 3 / 2,
                                                       color_format);
    } else {
        const auto& uv_tensor = input_values[1];
        ngraph::runtime::reference::color_convert_nv12(y_tensor->get_data_ptr<ET>(),
                                                       uv_tensor->get_data_ptr<ET>(),
                                                       output_value->get_data_ptr<ET>(),
                                                       batch_size,
                                                       image_h,
                                                       image_w,
                                                       image_w * image_h,
                                                       image_w * image_h / 2,
                                                       color_format);
    }
    return true;
}

bool evaluate_nv12_convert(const ov::HostTensorVector& input_values,
                           const ov::HostTensorPtr& output_value,
                           bool single_tensor,
                           ov::op::util::ConvertColorNV12Base::ColorConversion conv_format) {
    bool rc = false;
    switch (input_values[0]->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_nv12_convert, u8, input_values, output_value, single_tensor, conv_format);
        NGRAPH_TYPE_CASE(evaluate_nv12_convert, f16, input_values, output_value, single_tensor, conv_format);
        NGRAPH_TYPE_CASE(evaluate_nv12_convert, bf16, input_values, output_value, single_tensor, conv_format);
        NGRAPH_TYPE_CASE(evaluate_nv12_convert, f32, input_values, output_value, single_tensor, conv_format);
        NGRAPH_TYPE_CASE(evaluate_nv12_convert, f64, input_values, output_value, single_tensor, conv_format);
    default:
        break;
    }
    return rc;
}

}  // namespace color_convert_nv12_op

bool ov::op::util::ConvertColorNV12Base::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool ov::op::util::ConvertColorNV12Base::evaluate(const HostTensorVector& output_values,
                                                  const HostTensorVector& input_values) const {
    NGRAPH_OP_SCOPE(v0_ConvertColorNV12_evaluate);
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(output_values, 1));
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 1 || get_input_size() == 2,
                          "NV12 conversion shall have one or 2 inputs, but it is ",
                          get_input_size());
    auto single_plane = get_input_size() == 1;
    return color_convert_nv12_op::evaluate_nv12_convert(input_values, output_values[0], single_plane, m_format);
}

bool ov::op::util::ConvertColorNV12Base::has_evaluate() const {
    NGRAPH_OP_SCOPE(v0_ConvertColorNV12Base_has_evaluate);

    return is_type_supported(get_input_element_type(0));
}

bool ov::op::util::ConvertColorNV12Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
