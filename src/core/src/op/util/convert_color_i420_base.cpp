// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/convert_color_i420_base.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/runtime/reference/convert_color_nv12.hpp"
#include "openvino/core/layout.hpp"

namespace i420_op {
static const size_t N_DIM = 0;
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
    NGRAPH_OP_SCOPE(v8_Convert_I420_Base_validate_and_infer_types);

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

namespace i420_op {
namespace {

template <ov::element::Type_t ET>
inline bool evaluate(const ov::HostTensorVector& input_values,
                     const ov::HostTensorPtr& output_value,
                     bool single_tensor,
                     ov::op::util::ConvertColorI420Base::ColorConversion color_format) {
    using namespace ov::op::util;
    const auto& y_tensor = input_values[0];
    auto batch_size = y_tensor->get_shape()[N_DIM];
    auto image_w = y_tensor->get_shape()[W_DIM];
    auto image_h = y_tensor->get_shape()[H_DIM];
    if (single_tensor) {
        OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(input_values, 1));
        image_h = image_h * 2 / 3;
    } else {
        OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(input_values, 3));
    }
    output_value->set_shape({batch_size, image_h, image_w, 3});  // 3 is RGB
    if (single_tensor) {
        ngraph::runtime::reference::color_convert_i420(y_tensor->get_data_ptr<ET>(),
                                                       y_tensor->get_data_ptr<ET>() + image_w * image_h,
                                                       y_tensor->get_data_ptr<ET>() + 5 * image_w * image_h / 4,
                                                       output_value->get_data_ptr<ET>(),
                                                       batch_size,
                                                       image_h,
                                                       image_w,
                                                       image_w * image_h * 3 / 2,
                                                       image_w * image_h * 3 / 2,
                                                       color_format);
    } else {
        const auto& u_tensor = input_values[1];
        const auto& v_tensor = input_values[2];
        ngraph::runtime::reference::color_convert_i420(y_tensor->get_data_ptr<ET>(),
                                                       u_tensor->get_data_ptr<ET>(),
                                                       v_tensor->get_data_ptr<ET>(),
                                                       output_value->get_data_ptr<ET>(),
                                                       batch_size,
                                                       image_h,
                                                       image_w,
                                                       image_w * image_h,
                                                       image_w * image_h / 4,
                                                       color_format);
    }
    return true;
}

bool evaluate_i420_convert(const ov::HostTensorVector& input_values,
                           const ov::HostTensorPtr& output_value,
                           bool single_tensor,
                           ov::op::util::ConvertColorI420Base::ColorConversion conv_format) {
    bool rc = false;
    switch (input_values[0]->get_element_type()) {
        NGRAPH_TYPE_CASE(evaluate_i420_convert, u8, input_values, output_value, single_tensor, conv_format);
        NGRAPH_TYPE_CASE(evaluate_i420_convert, f32, input_values, output_value, single_tensor, conv_format);
    default:
        break;
    }
    return rc;
}

}  // namespace
}  // namespace i420_op

bool ov::op::util::ConvertColorI420Base::visit_attributes(AttributeVisitor& visitor) {
    return true;
}

bool ov::op::util::ConvertColorI420Base::evaluate(const HostTensorVector& output_values,
                                                  const HostTensorVector& input_values) const {
    NGRAPH_OP_SCOPE(v0_ConvertColorI420_evaluate);
    OPENVINO_ASSERT(ngraph::validate_host_tensor_vector(output_values, 1));
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 1 || get_input_size() == 3,
                          "I420 conversion shall have one or 3 inputs, but it is ",
                          get_input_size());
    auto single_plane = get_input_size() == 1;
    return i420_op::evaluate_i420_convert(input_values, output_values[0], single_plane, m_format);
}

bool ov::op::util::ConvertColorI420Base::has_evaluate() const {
    NGRAPH_OP_SCOPE(v0_ConvertColorI420Base_has_evaluate);

    return is_type_supported(get_input_element_type(0));
}

bool ov::op::util::ConvertColorI420Base::is_type_supported(const ov::element::Type& type) const {
    return type.is_dynamic() || type.is_real() || type == ov::element::u8;
}
