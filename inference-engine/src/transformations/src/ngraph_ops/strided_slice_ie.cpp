// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_ops/strided_slice_ie.hpp"

#include <algorithm>
#include <vector>
#include <memory>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::StridedSliceIE::type_info;

op::StridedSliceIE::StridedSliceIE(const Output <Node> &data, const Output <Node> &begin, const Output <Node> &end,
                                   const Output <Node> &strides, const std::vector<int64_t> &begin_mask,
                                   const std::vector<int64_t> &end_mask, const std::vector<int64_t> &new_axis_mask,
                                   const std::vector<int64_t> &shrink_axis_mask,
                                   const std::vector<int64_t> &ellipsis_mask,
                                   const Shape& output_shape)
                                   : Op({data, begin, end, strides}),
                                   m_begin_mask(begin_mask),
                                   m_end_mask(end_mask),
                                   m_new_axis_mask(new_axis_mask),
                                   m_shrink_axis_mask(shrink_axis_mask),
                                   m_ellipsis_mask(ellipsis_mask),
                                   m_output_shape(output_shape) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<Node> op::StridedSliceIE::copy_with_new_args(const NodeVector& new_args) const {
    if (new_args.size() != 4) {
        throw ngraph_error("Incorrect number of new arguments");
    }

    return make_shared<StridedSliceIE>(new_args.at(0), new_args.at(1), new_args.at(2), new_args.at(3), m_begin_mask,
            m_end_mask, m_new_axis_mask, m_shrink_axis_mask, m_ellipsis_mask, m_output_shape);
}

void op::StridedSliceIE::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), PartialShape(m_output_shape));
}
