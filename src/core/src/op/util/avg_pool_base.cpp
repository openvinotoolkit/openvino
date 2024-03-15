// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/avg_pool_base.hpp"

#include "itt.hpp"

ov::op::util::AvgPoolBase::AvgPoolBase(const Output<Node>& arg,
                                       const Strides& strides,
                                       const Shape& pads_begin,
                                       const Shape& pads_end,
                                       const Shape& kernel,
                                       bool exclude_pad,
                                       op::RoundingType rounding_type,
                                       const PadType& auto_pad)
    : Op({arg}),
      m_kernel(kernel),
      m_strides(strides),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_exclude_pad(exclude_pad),
      m_auto_pad(auto_pad),
      m_rounding_type(rounding_type) {
    constructor_validate_and_infer_types();
}

void ov::op::util::AvgPoolBase::validate_and_infer_types() {
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

bool ov::op::util::AvgPoolBase::visit_attributes(AttributeVisitor& visitor) {
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
