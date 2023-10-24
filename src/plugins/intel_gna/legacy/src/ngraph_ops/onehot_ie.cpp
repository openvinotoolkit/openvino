// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/onehot_ie.hpp"

#include <memory>

using namespace std;
using namespace ngraph;

op::OneHotIE::OneHotIE(const Output<ngraph::Node>& input,
                       int axis,
                       int depth,
                       float on_value,
                       float off_value,
                       element::Type type)
    : Op({input}),
      m_type(type),
      m_axis(axis),
      m_depth(depth),
      m_off_value(off_value),
      m_on_value(on_value) {
    constructor_validate_and_infer_types();
}

void op::OneHotIE::validate_and_infer_types() {
    const PartialShape& arg_shape = get_input_partial_shape(0);

    if (arg_shape.rank().is_dynamic()) {
        set_output_type(0, m_type, PartialShape::dynamic());
    } else {
        vector<Dimension> output_shape{arg_shape};
        int normalized_axis = m_axis;
        if (m_axis < 0)
            normalized_axis = m_axis + static_cast<int>(arg_shape.size());
        output_shape.insert(output_shape.begin() + normalized_axis, m_depth);
        set_output_type(0, m_type, output_shape);
    }
}

shared_ptr<Node> op::OneHotIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<op::OneHotIE>(new_args.at(0), m_axis, m_depth, m_on_value, m_off_value, m_type);
}

bool op::OneHotIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("axis", m_axis);
    visitor.on_attribute("depth", m_depth);
    visitor.on_attribute("off_value", m_off_value);
    visitor.on_attribute("on_value", m_on_value);
    return true;
}
