// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/util/max_pool_base.hpp"

#include "itt.hpp"

ov::op::util::MaxPoolBase::MaxPoolBase(const Output<Node>& arg,
                                       const Strides& strides,
                                       const ov::Shape& pads_begin,
                                       const ov::Shape& pads_end,
                                       const ov::Shape& kernel,
                                       const op::RoundingType rounding_type,
                                       const op::PadType auto_pad)
    : Op({arg}),
      m_kernel(kernel),
      m_strides(strides),
      m_pads_begin(pads_begin),
      m_pads_end(pads_end),
      m_auto_pad(auto_pad),
      m_rounding_type(rounding_type) {
    constructor_validate_and_infer_types();
}

void ov::op::util::MaxPoolBase::validate_and_infer_types() {
    OV_OP_SCOPE(util_MaxPoolBase_validate_and_infer_types);

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

void ov::op::util::MaxPoolBase::set_pads_end(Shape pads_end) {
    m_pads_end = std::move(pads_end);
}
