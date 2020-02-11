// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "onehot_ie.hpp"

#include <memory>
#include <ie_parameter.hpp>

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::OneHotIE::type_info;

op::OneHotIE::OneHotIE(const std::shared_ptr<ngraph::Node>& input, int axis, int depth, float on_value, float off_value, element::Type type)
        : Op("OneHotIE", {input}), m_axis(axis), m_depth(depth), m_on_value(on_value), m_off_value(off_value), m_type(type) {
    constructor_validate_and_infer_types();
}

void op::OneHotIE::validate_and_infer_types() {
    const PartialShape& arg_shape = get_input_partial_shape(0);

    if (arg_shape.is_dynamic()) {
        set_output_type(0, m_type, PartialShape::dynamic());
    } else {
        Shape output_shape = arg_shape.to_shape();
        int normalized_axis = m_axis;
        if (m_axis < 0)
            normalized_axis = m_axis + static_cast<int>(arg_shape.to_shape().size());
        output_shape.insert(output_shape.begin() + normalized_axis, m_depth);
        set_output_type(0, m_type, output_shape);
    }
}

shared_ptr<Node> op::OneHotIE::copy_with_new_args(const NodeVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::OneHotIE>(new_args.at(0), m_axis, m_depth, m_on_value, m_off_value, m_type);
}
