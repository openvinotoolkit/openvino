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
