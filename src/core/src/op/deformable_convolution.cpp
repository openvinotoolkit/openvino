// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/deformable_convolution.hpp"

#include "deformable_convolution_shape_inference.hpp"
#include "itt.hpp"

namespace ov {
op::v8::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& offsets,
                                                     const Output<Node>& filters,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const op::PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group,
                                                     const bool bilinear_interpolation_pad)
    : DeformableConvolutionBase({arg, offsets, filters},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group),
      m_bilinear_interpolation_pad(bilinear_interpolation_pad) {
    constructor_validate_and_infer_types();
}

op::v8::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& offsets,
                                                     const Output<Node>& filters,
                                                     const Output<Node>& mask,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const op::PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group,
                                                     const bool bilinear_interpolation_pad)
    : DeformableConvolutionBase({arg, offsets, filters, mask},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group),
      m_bilinear_interpolation_pad(bilinear_interpolation_pad) {
    constructor_validate_and_infer_types();
}

bool op::v8::DeformableConvolution::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(DeformableConvolution_v8_visit_attributes);
    visitor.on_attribute("bilinear_interpolation_pad", m_bilinear_interpolation_pad);
    return DeformableConvolutionBase::visit_attributes(visitor);
}

void op::v8::DeformableConvolution::validate_and_infer_types() {
    OV_OP_SCOPE(DeformableConvolution_v8_validate_and_infer_types);

    const auto& data_batch_et = get_input_element_type(0);
    const auto& offsets_et = get_input_element_type(1);
    const auto& filters_et = get_input_element_type(2);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_batch_et, offsets_et) &&
                              element::Type::merge(result_et, result_et, filters_et),
                          "Element types of inputs do not match. Got: data batch (",
                          data_batch_et,
                          "), offsets (",
                          offsets_et,
                          ") and filters (",
                          filters_et,
                          ")");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    if (get_input_size() == 4) {
        element::Type mask_et = get_input_element_type(3);

        NODE_VALIDATION_CHECK(this,
                              mask_et.is_real() || mask_et.is_integral_number(),
                              "Element type of Mask input must be numeric. Got: ",
                              mask_et);
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto num_spatial = deformable_conv::calculate_num_spatial(this, input_shapes);
    if (num_spatial != convolution::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }

    const auto output_shapes = shape_infer(this, input_shapes, m_pads_begin, m_pads_end);
    set_output_type(0, result_et, output_shapes[0]);
}

std::shared_ptr<Node> op::v8::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(DeformableConvolution_v8_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    NODE_VALIDATION_CHECK(this, new_args.size() >= 3 && new_args.size() <= 4, "Number of inputs must be 3 or 4");
    switch (new_args.size()) {
    case 3:
        return std::make_shared<DeformableConvolution>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       m_strides,
                                                       m_pads_begin,
                                                       m_pads_end,
                                                       m_dilations,
                                                       m_auto_pad,
                                                       m_group,
                                                       m_deformable_group,
                                                       m_bilinear_interpolation_pad);
    default:
        return std::make_shared<DeformableConvolution>(new_args.at(0),
                                                       new_args.at(1),
                                                       new_args.at(2),
                                                       new_args.at(3),
                                                       m_strides,
                                                       m_pads_begin,
                                                       m_pads_end,
                                                       m_dilations,
                                                       m_auto_pad,
                                                       m_group,
                                                       m_deformable_group,
                                                       m_bilinear_interpolation_pad);
    }
}

op::v1::DeformableConvolution::DeformableConvolution(const Output<Node>& arg,
                                                     const Output<Node>& offsets,
                                                     const Output<Node>& filters,
                                                     const Strides& strides,
                                                     const CoordinateDiff& pads_begin,
                                                     const CoordinateDiff& pads_end,
                                                     const Strides& dilations,
                                                     const op::PadType& auto_pad,
                                                     const int64_t group,
                                                     const int64_t deformable_group)
    : DeformableConvolutionBase({arg, offsets, filters},
                                strides,
                                pads_begin,
                                pads_end,
                                dilations,
                                auto_pad,
                                group,
                                deformable_group) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::v1::DeformableConvolution::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(DeformableConvolution_v1_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<DeformableConvolution>(new_args.at(0),
                                                   new_args.at(1),
                                                   new_args.at(2),
                                                   m_strides,
                                                   m_pads_begin,
                                                   m_pads_end,
                                                   m_dilations,
                                                   m_auto_pad,
                                                   m_group,
                                                   m_deformable_group);
}

void op::v1::DeformableConvolution::validate_and_infer_types() {
    OV_OP_SCOPE(DeformableConvolution_v1_validate_and_infer_types);

    const auto& data_batch_et = get_input_element_type(0);
    const auto& offsets_et = get_input_element_type(1);
    const auto& filters_et = get_input_element_type(2);

    element::Type result_et;
    NODE_VALIDATION_CHECK(this,
                          element::Type::merge(result_et, data_batch_et, offsets_et) &&
                              element::Type::merge(result_et, result_et, filters_et),
                          "Element types of inputs do not match. Got: data batch (",
                          data_batch_et,
                          "), offsets (",
                          offsets_et,
                          ") and filters (",
                          filters_et,
                          ")");

    NODE_VALIDATION_CHECK(this,
                          result_et.is_real() || result_et.is_integral_number(),
                          "Element type of inputs must be numeric. Got: ",
                          result_et);

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto num_spatial = deformable_conv::calculate_num_spatial(this, input_shapes);
    if (num_spatial != convolution::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }

    const auto output_shapes = shape_infer(this, input_shapes, m_pads_begin, m_pads_end);
    set_output_type(0, result_et, output_shapes[0]);
}
}  // namespace ov
