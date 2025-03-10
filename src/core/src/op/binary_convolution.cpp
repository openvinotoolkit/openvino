// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/binary_convolution.hpp"

#include "binary_convolution_shape_inference.hpp"
#include "convolution_shape_inference.hpp"
#include "itt.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/axis_vector.hpp"
#include "openvino/core/coordinate_diff.hpp"

ov::op::v1::BinaryConvolution::BinaryConvolution(const Output<Node>& data,
                                                 const Output<Node>& kernel,
                                                 const Strides& strides,
                                                 const CoordinateDiff& pads_begin,
                                                 const CoordinateDiff& pads_end,
                                                 const Strides& dilations,
                                                 BinaryConvolutionMode mode,
                                                 float pad_value,
                                                 const PadType& auto_pad)
    : ConvolutionFwdPropBase({data, kernel}, strides, pads_begin, pads_end, dilations, auto_pad),
      m_mode(mode),
      m_pad_value(pad_value) {
    constructor_validate_and_infer_types();
}

ov::op::v1::BinaryConvolution::BinaryConvolution(const Output<Node>& data,
                                                 const Output<Node>& kernel,
                                                 const Strides& strides,
                                                 const CoordinateDiff& pads_begin,
                                                 const CoordinateDiff& pads_end,
                                                 const Strides& dilations,
                                                 const std::string& mode,
                                                 float pad_value,
                                                 const PadType& auto_pad)
    : ConvolutionFwdPropBase({data, kernel}, strides, pads_begin, pads_end, dilations, auto_pad),
      m_mode(mode_from_string(mode)),
      m_pad_value(pad_value) {
    constructor_validate_and_infer_types();
}

void ov::op::v1::BinaryConvolution::validate_and_infer_types() {
    OV_OP_SCOPE(v1_BinaryConvolution_validate_and_infer_types);

    const auto& data_batch_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          data_batch_et.is_real() || data_batch_et.is_integral_number(),
                          "Data batch element type must be numeric. Got: ",
                          data_batch_et);

    // TODO: Add NodeValidationCheck to filters et once u1 is supported in OpenVINO Python API
    // (#52715)
    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);

    auto num_spatial = convolution::calculate_num_spatial(this, input_shapes);
    if (num_spatial != util::num_spatial_undefined) {
        resize_attributes(num_spatial);
    }

    const auto output_shapes = shape_infer(this, input_shapes, m_pads_begin, m_pads_end);
    set_output_type(0, data_batch_et, output_shapes[0]);
}

std::shared_ptr<ov::Node> ov::op::v1::BinaryConvolution::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(v1_BinaryConvolution_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<v1::BinaryConvolution>(new_args.at(0),
                                                   new_args.at(1),
                                                   m_strides,
                                                   m_pads_begin,
                                                   m_pads_end,
                                                   m_dilations,
                                                   m_mode,
                                                   m_pad_value,
                                                   m_auto_pad);
}

bool ov::op::v1::BinaryConvolution::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(v1_BinaryConvolution_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("mode", m_mode);
    visitor.on_attribute("pad_value", m_pad_value);
    visitor.on_attribute("auto_pad", m_auto_pad);
    return true;
}

namespace ov {
template <>
OPENVINO_API EnumNames<ov::op::v1::BinaryConvolution::BinaryConvolutionMode>&
EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>::get() {
    static auto enum_names = EnumNames<op::v1::BinaryConvolution::BinaryConvolutionMode>(
        "op::v1::BinaryConvolution::BinaryConvolutionMode",
        {{"xnor-popcount", op::v1::BinaryConvolution::BinaryConvolutionMode::XNOR_POPCOUNT}});
    return enum_names;
}

AttributeAdapter<op::v1::BinaryConvolution::BinaryConvolutionMode>::~AttributeAdapter() = default;
}  // namespace ov

std::ostream& ov::operator<<(std::ostream& s, const ov::op::v1::BinaryConvolution::BinaryConvolutionMode& type) {
    return s << as_string(type);
}

ov::op::v1::BinaryConvolution::BinaryConvolutionMode ov::op::v1::BinaryConvolution::mode_from_string(
    const std::string& mode) const {
    return as_enum<BinaryConvolutionMode>(mode);
}
