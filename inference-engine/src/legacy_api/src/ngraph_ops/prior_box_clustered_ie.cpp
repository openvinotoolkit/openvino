// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/prior_box_clustered_ie.hpp"

#include <memory>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PriorBoxClusteredIE::type_info;

op::PriorBoxClusteredIE::PriorBoxClusteredIE(const Output<Node>& input, const Output<Node>& image,
                                             const PriorBoxClusteredAttrs& attrs)
    : Op({input, image}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::PriorBoxClusteredIE::validate_and_infer_types() {
    if (get_input_partial_shape(0).is_dynamic() || get_input_partial_shape(1).is_dynamic()) {
        set_output_type(0, element::f32, PartialShape::dynamic(3));
        return;
    }

    auto input_shape = get_input_shape(0);
    auto image_shape = get_input_shape(1);

    size_t num_priors = m_attrs.widths.size();

    set_output_type(0, element::f32, Shape {1, 2, 4 * input_shape[2] * input_shape[3] * num_priors});
}

std::shared_ptr<Node> op::PriorBoxClusteredIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxClusteredIE>(new_args.at(0), new_args.at(1), m_attrs);
}

bool op::PriorBoxClusteredIE::visit_attributes(AttributeVisitor& visitor)
{
    float step = 0;

    visitor.on_attribute("step", step);
    visitor.on_attribute("step_w", m_attrs.step_widths);
    visitor.on_attribute("step_h", m_attrs.step_heights);
    if(step != 0) {
        // deserialization: if step_w/h is 0 replace it with step
        if (m_attrs.step_widths == 0) {
            m_attrs.step_widths = step;
        }
        if (m_attrs.step_heights == 0) {
            m_attrs.step_heights = step;
        }
    }
    visitor.on_attribute("width", m_attrs.widths);
    visitor.on_attribute("height", m_attrs.heights);
    visitor.on_attribute("clip", m_attrs.clip);
    visitor.on_attribute("offset", m_attrs.offset);
    visitor.on_attribute("variance", m_attrs.variances);
    return true;
}
