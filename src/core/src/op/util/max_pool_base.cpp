// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/op/util/max_pool_base.hpp"

#include <ngraph/validation_util.hpp>

#include "itt.hpp"
#include "ngraph/shape.hpp"

using namespace std;

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

    if (0 == m_strides.size()) {
        m_strides = Strides(m_kernel.size(), 1);
    }

    if (0 == m_pads_begin.size()) {
        m_pads_begin = ov::Shape(m_kernel.size(), 0);
    }

    if (0 == m_pads_end.size()) {
        m_pads_end = ov::Shape(m_kernel.size(), 0);
    }

    const PartialShape& arg_shape = get_input_partial_shape(0);

    NODE_VALIDATION_CHECK(
        this,
        arg_shape.rank().compatible(3) || arg_shape.rank().compatible(4) || arg_shape.rank().compatible(5),
        "Expected a 3D, 4D or 5D tensor for the input. Got: ",
        arg_shape);

    if (arg_shape.rank().is_static()) {
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_end.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected pads_end size to be equal to input size - 2. Got: ",
                              m_pads_end.size());

        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_pads_begin.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected pads_begin size to be equal to input size - 2. Got: ",
                              m_pads_begin.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_kernel.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected kernel size to be equal to input size - 2. Got: ",
                              m_kernel.size());
        NODE_VALIDATION_CHECK(this,
                              static_cast<int64_t>(m_strides.size()) == arg_shape.rank().get_max_length() - 2,
                              "Expected strides size to be equal to input size - 2. Got: ",
                              m_strides.size());
    }
}

ov::PartialShape ov::op::util::MaxPoolBase::infer_output_shape(const Strides& dilations) {
    OV_OP_SCOPE(util_MaxPoolBase_infer_output_shape);

    const auto& arg_shape = get_input_partial_shape(0);

    bool update_auto_padding_succeed = true;

    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER) {
        const auto filter_dilations = dilations.empty() ? Strides(m_kernel.size(), 1) : dilations;
        update_auto_padding_succeed = update_auto_padding(arg_shape, filter_dilations, m_pads_end, m_pads_begin);
    }
    if (m_auto_pad == PadType::VALID) {
        m_pads_end = ov::Shape(m_pads_end.size(), 0);
        m_pads_begin = ov::Shape(m_pads_begin.size(), 0);
    }

    auto output_shape = PartialShape::dynamic();
    if (update_auto_padding_succeed) {
        CoordinateDiff pads_begin(m_pads_begin.begin(), m_pads_begin.end());
        CoordinateDiff pads_end(m_pads_end.begin(), m_pads_end.end());
        output_shape = ngraph::infer_batched_pooling_forward(this,
                                                             get_input_partial_shape(0),
                                                             pads_begin,
                                                             pads_end,
                                                             m_kernel,
                                                             m_strides,
                                                             true,
                                                             m_rounding_type == op::RoundingType::CEIL,
                                                             dilations);
    } else {
        if (arg_shape.rank().is_static() && arg_shape.rank().get_max_length() > 0) {
            output_shape = std::vector<Dimension>(arg_shape.rank().get_max_length(), Dimension::dynamic());
            if (arg_shape[0].is_static()) {
                output_shape[0] = arg_shape[0];  // batch size
            }
            if (arg_shape[1].is_static()) {
                output_shape[1] = arg_shape[1];  // channel size
            }
        }
    }

    return output_shape;
}

bool ov::op::util::MaxPoolBase::update_auto_padding(const PartialShape& in_shape,
                                                    const Strides& filter_dilations,
                                                    ov::Shape& new_pads_end,
                                                    ov::Shape& new_pads_begin) const {
    bool update_auto_padding_succeed = true;
    if (m_auto_pad == PadType::SAME_UPPER || m_auto_pad == PadType::SAME_LOWER) {
        CoordinateDiff pads_end, pads_begin;
        update_auto_padding_succeed = ngraph::try_apply_auto_padding(in_shape,
                                                                     m_kernel,
                                                                     m_strides,
                                                                     filter_dilations,
                                                                     m_auto_pad,
                                                                     pads_end,
                                                                     pads_begin);
        new_pads_end = ov::Shape(pads_end.begin(), pads_end.end());
        new_pads_begin = ov::Shape(pads_begin.begin(), pads_begin.end());
    }
    return update_auto_padding_succeed;
}
