// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/deformable_convolution_base.hpp"

#include "itt.hpp"

ov::op::util::DeformableConvolutionBase::DeformableConvolutionBase(const OutputVector& arguments,
                                                                   const Strides& strides,
                                                                   const CoordinateDiff& pads_begin,
                                                                   const CoordinateDiff& pads_end,
                                                                   const Strides& dilations,
                                                                   const PadType& auto_pad,
                                                                   const int64_t group,
                                                                   const int64_t deformable_group)
    : ConvolutionBase(arguments, strides, pads_begin, pads_end, dilations, auto_pad),
      m_group(group),
      m_deformable_group(deformable_group) {}

bool ov::op::util::DeformableConvolutionBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_DeformableConvolutionBase_visit_attributes);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("dilations", m_dilations);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("group", m_group);
    visitor.on_attribute("deformable_group", m_deformable_group);
    return true;
}
