// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/strided_slice_ie.hpp"

#include <algorithm>
#include <vector>
#include <memory>
#include <ngraph/ops.hpp>
#include <ngraph/opsets/opset1.hpp>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::StridedSliceIE::type_info;

op::StridedSliceIE::StridedSliceIE(const Output <Node> &data, const Output <Node> &begin, const Output <Node> &end,
                                   const Output <Node> &strides, const std::vector<int64_t> &begin_mask,
                                   const std::vector<int64_t> &end_mask, const std::vector<int64_t> &new_axis_mask,
                                   const std::vector<int64_t> &shrink_axis_mask,
                                   const std::vector<int64_t> &ellipsis_mask)
    : Op({data, begin, end, strides})
    , m_begin_mask(begin_mask)
    , m_end_mask(end_mask)
    , m_new_axis_mask(new_axis_mask)
    , m_shrink_axis_mask(shrink_axis_mask)
    , m_ellipsis_mask(ellipsis_mask) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::StridedSliceIE::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<op::StridedSliceIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_begin_mask,
            m_end_mask, m_new_axis_mask, m_shrink_axis_mask, m_ellipsis_mask);
}

void op::StridedSliceIE::validate_and_infer_types() {
    const auto& begin_mask_et = input_value(1).get_element_type();
    const auto& end_mask_et = input_value(2).get_element_type();
    const auto& strides_et = input_value(3).get_element_type();

    NODE_VALIDATION_CHECK(this,
                          begin_mask_et.is_integral_number(),
                          "Begin mask must have i32 type, but its: ",
                          begin_mask_et);

    NODE_VALIDATION_CHECK(this,
                          end_mask_et == element::i32,
                          "End mask must have i32 type, but its: ",
                          end_mask_et);

    NODE_VALIDATION_CHECK(this,
                          strides_et.is_integral_number(),
                          "Strides must have i32 type, but its: ",
                          strides_et);

    // Calculate output shape via opset1::StridedSlice operation
    auto slice = std::make_shared<opset1::StridedSlice>(input_value(0), input_value(1), input_value(2), input_value(3),
            m_begin_mask, m_end_mask, m_new_axis_mask, m_shrink_axis_mask, m_ellipsis_mask);
    set_output_type(0, slice->output(0).get_element_type(), slice->output(0).get_partial_shape());
}
