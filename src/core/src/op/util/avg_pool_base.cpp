// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/avg_pool_base.hpp"

#include "itt.hpp"

namespace ov {
namespace op {
namespace util {

AvgPoolBase::AvgPoolBase(const Output<Node>& arg,
                         const Strides& strides,
                         const Shape& pads_begin,
                         const Shape& pads_end,
                         const Shape& kernel,
                         bool exclude_pad,
                         RoundingType rounding_type,
                         const PadType& auto_pad)
    : Op{{arg}},
      m_kernel{kernel},
      m_strides{strides},
      m_pads_begin{pads_begin},
      m_pads_end{pads_end},
      m_exclude_pad{exclude_pad},
      m_auto_pad{auto_pad},
      m_rounding_type{rounding_type} {
    constructor_validate_and_infer_types();
}

void AvgPoolBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_AvgPoolBase_validate_and_infer_types);

    if (m_strides.empty()) {
        m_strides.resize(m_kernel.size(), 1);
    }

    if (m_pads_begin.empty()) {
        m_pads_begin.resize(m_kernel.size(), 0);
    }

    if (m_pads_end.empty()) {
        m_pads_end.resize(m_kernel.size(), 0);
    }
}

bool AvgPoolBase::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(util_AvgPoolBase_visit_attributes);
    visitor.on_attribute("kernel", m_kernel);
    visitor.on_attribute("strides", m_strides);
    visitor.on_attribute("pads_begin", m_pads_begin);
    visitor.on_attribute("pads_end", m_pads_end);
    visitor.on_attribute("exclude-pad", m_exclude_pad);
    visitor.on_attribute("auto_pad", m_auto_pad);
    visitor.on_attribute("rounding_type", m_rounding_type);
    return true;
}

const Shape& AvgPoolBase::get_kernel() const {
    return m_kernel;
}

void AvgPoolBase::set_kernel(const Shape& kernel) {
    m_kernel = kernel;
}

const Strides& AvgPoolBase::get_strides() const {
    return m_strides;
}

void AvgPoolBase::set_strides(const Strides& strides) {
    m_strides = strides;
}

const Shape& AvgPoolBase::get_pads_begin() const {
    return m_pads_begin;
}

void AvgPoolBase::set_pads_begin(const Shape& pads_begin) {
    m_pads_begin = pads_begin;
}

const Shape& AvgPoolBase::get_pads_end() const {
    return m_pads_end;
}

void AvgPoolBase::set_pads_end(const Shape& pads_end) {
    m_pads_end = pads_end;
}

bool AvgPoolBase::get_exclude_pad() const {
    return m_exclude_pad;
}

void AvgPoolBase::set_exclude_pad(bool exclude_pad) {
    m_exclude_pad = exclude_pad;
}

const PadType& AvgPoolBase::get_auto_pad() const {
    return m_auto_pad;
}

void AvgPoolBase::set_auto_pad(const PadType& auto_pad) {
    m_auto_pad = auto_pad;
}

RoundingType AvgPoolBase::get_rounding_type() const {
    return m_rounding_type;
}

void AvgPoolBase::set_rounding_type(RoundingType rounding_type) {
    m_rounding_type = rounding_type;
}

}  // namespace util
}  // namespace op
}  // namespace ov
