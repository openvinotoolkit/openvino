// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/interpolate_base.hpp"

#include "itt.hpp"
#include "openvino/op/util/precision_sensitive_attribute.hpp"

using namespace ov;
using namespace ov::op::util;

InterpolateBase::InterpolateBase(const Output<Node>& image,
                                 const Output<Node>& scales_or_sizes,
                                 const InterpolateAttrs& attrs)
    : op::Op{{image, scales_or_sizes}},
      m_attrs{attrs} {
    ov::mark_as_precision_sensitive(input(1));
}

InterpolateBase::InterpolateBase(const Output<Node>& image,
                                 const Output<Node>& scales_or_sizes,
                                 const Output<Node>& axes,
                                 const InterpolateAttrs& attrs)
    : op::Op{{image, scales_or_sizes, axes}},
      m_attrs{attrs} {
    ov::mark_as_precision_sensitive(input(1));
}

InterpolateBase::InterpolateBase(const Output<Node>& image,
                                 const Output<Node>& output_shape,
                                 const Output<Node>& scales,
                                 const Output<Node>& axes,
                                 const InterpolateAttrs& attrs)
    : op::Op{{image, output_shape, scales, axes}},
      m_attrs{attrs} {
    ov::mark_as_precision_sensitive(input(1));
    ov::mark_as_precision_sensitive(input(2));
}

bool InterpolateBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_InterpolateBase_visit_attributes);
    visitor.on_attribute("mode", m_attrs.mode);
    visitor.on_attribute("shape_calculation_mode", m_attrs.shape_calculation_mode);
    visitor.on_attribute("coordinate_transformation_mode", m_attrs.coordinate_transformation_mode);
    visitor.on_attribute("nearest_mode", m_attrs.nearest_mode);
    visitor.on_attribute("antialias", m_attrs.antialias);
    visitor.on_attribute("pads_begin", m_attrs.pads_begin);
    visitor.on_attribute("pads_end", m_attrs.pads_end);
    visitor.on_attribute("cube_coeff", m_attrs.cube_coeff);
    return true;
}

void InterpolateBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_InterpolateBase_validate_and_infer_types);
    const element::Type input_et = get_input_element_type(0);
    NODE_VALIDATION_CHECK(this,
                          input_et == element::f32 || input_et == element::f16 || input_et == element::i8 ||
                              input_et == element::bf16 || input_et == element::u8 || input_et == element::i64 ||
                              input_et == element::i32 || input_et == element::dynamic,
                          "Input element type must be f32, f16, bf16, i8, u8, i64, i32");
}

void InterpolateBase::validate_scales_element_type(const element::Type& et) const {
    NODE_VALIDATION_CHECK(this,
                          et == element::f32 || et == element::f16 || et == element::bf16,
                          "Scales element type must be f32, f16 or bf16");
}

void InterpolateBase::validate_sizes_element_type(const element::Type& et) const {
    NODE_VALIDATION_CHECK(this,
                          et == element::i32 || et == element::i64 || et == element::u32 || et == element::u64,
                          "Sizes element type must be i32, i64, u32 or u64");
}

void InterpolateBase::validate_axes_element_type(const element::Type& et) const {
    NODE_VALIDATION_CHECK(this,
                          et == element::i64 || et == element::i32 || et == element::u32 || et == element::u64,
                          "Axes element type must be i32, i64, u32 or u64");
}

template <>
OPENVINO_API EnumNames<op::util::InterpolateBase::InterpolateMode>&
EnumNames<ov::op::util::InterpolateBase::InterpolateMode>::get() {
    static auto enum_names = EnumNames<ov::op::util::InterpolateBase::InterpolateMode>(
        "op::util::InterpolateBase::InterpolateMode",
        {{"nearest", ov::op::util::InterpolateBase::InterpolateMode::NEAREST},
         {"linear", ov::op::util::InterpolateBase::InterpolateMode::LINEAR},
         {"linear_onnx", ov::op::util::InterpolateBase::InterpolateMode::LINEAR_ONNX},
         {"cubic", ov::op::util::InterpolateBase::InterpolateMode::CUBIC},
         {"bilinear_pillow", ov::op::util::InterpolateBase::InterpolateMode::BILINEAR_PILLOW},
         {"bicubic_pillow", ov::op::util::InterpolateBase::InterpolateMode::BICUBIC_PILLOW}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<op::util::InterpolateBase::ShapeCalcMode>&
EnumNames<ov::op::util::InterpolateBase::ShapeCalcMode>::get() {
    static auto enum_names = EnumNames<ov::op::util::InterpolateBase::ShapeCalcMode>(
        "op::util::InterpolateBase::ShapeCalcMode",
        {{"sizes", ov::op::util::InterpolateBase::ShapeCalcMode::SIZES},
         {"scales", ov::op::util::InterpolateBase::ShapeCalcMode::SCALES}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<op::util::InterpolateBase::CoordinateTransformMode>&
EnumNames<ov::op::util::InterpolateBase::CoordinateTransformMode>::get() {
    static auto enum_names = EnumNames<ov::op::util::InterpolateBase::CoordinateTransformMode>(
        "op::util::InterpolateBase::CoordinateTransformMode",
        {{"half_pixel", ov::op::util::InterpolateBase::CoordinateTransformMode::HALF_PIXEL},
         {"pytorch_half_pixel", ov::op::util::InterpolateBase::CoordinateTransformMode::PYTORCH_HALF_PIXEL},
         {"asymmetric", ov::op::util::InterpolateBase::CoordinateTransformMode::ASYMMETRIC},
         {"tf_half_pixel_for_nn", ov::op::util::InterpolateBase::CoordinateTransformMode::TF_HALF_PIXEL_FOR_NN},
         {"align_corners", ov::op::util::InterpolateBase::CoordinateTransformMode::ALIGN_CORNERS}});
    return enum_names;
}

template <>
OPENVINO_API EnumNames<op::util::InterpolateBase::NearestMode>&
EnumNames<ov::op::util::InterpolateBase::NearestMode>::get() {
    static auto enum_names = EnumNames<ov::op::util::InterpolateBase::NearestMode>(
        "op::util::InterpolateBase::NearestMode",
        {{"round_prefer_floor", ov::op::util::InterpolateBase::NearestMode::ROUND_PREFER_FLOOR},
         {"round_prefer_ceil", ov::op::util::InterpolateBase::NearestMode::ROUND_PREFER_CEIL},
         {"floor", ov::op::util::InterpolateBase::NearestMode::FLOOR},
         {"ceil", ov::op::util::InterpolateBase::NearestMode::CEIL},
         {"simple", ov::op::util::InterpolateBase::NearestMode::SIMPLE}});
    return enum_names;
}

std::ostream& ov::operator<<(std::ostream& s, const ov::op::util::InterpolateBase::InterpolateMode& type) {
    return s << as_string(type);
}

std::ostream& ov::operator<<(std::ostream& s, const ov::op::util::InterpolateBase::ShapeCalcMode& type) {
    return s << as_string(type);
}

std::ostream& ov::operator<<(std::ostream& s, const ov::op::util::InterpolateBase::CoordinateTransformMode& type) {
    return s << as_string(type);
}

std::ostream& ov::operator<<(std::ostream& s, const ov::op::util::InterpolateBase::NearestMode& type) {
    return s << as_string(type);
}
