// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "legacy/ngraph_ops/prior_box_ie.hpp"

#include <memory>

#include "ngraph/op/constant.hpp"

using namespace std;
using namespace ngraph;

constexpr NodeTypeInfo op::PriorBoxIE::type_info;

op::PriorBoxIE::PriorBoxIE(const Output<Node>& input, const Output<Node>& image, const PriorBoxAttrs& attrs)
    : Op({input, image}), m_attrs(attrs) {
    constructor_validate_and_infer_types();
}

void op::PriorBoxIE::validate_and_infer_types() {
    if (get_input_partial_shape(0).is_dynamic() || get_input_partial_shape(1).is_dynamic()) {
        set_output_type(0, element::f32, PartialShape::dynamic(3));
        return;
    }
    auto input_shape = get_input_shape(0);
    auto image_shape = get_input_shape(1);

    set_output_type(0, element::f32, Shape {
        1, 2, 4 * input_shape[2] * input_shape[3] * static_cast<size_t>(op::PriorBox::number_of_priors(m_attrs))});
}

std::shared_ptr<Node> op::PriorBoxIE::clone_with_new_inputs(const OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return make_shared<PriorBoxIE>(new_args.at(0), new_args.at(1), m_attrs);
}

bool op::PriorBoxIE::visit_attributes(AttributeVisitor& visitor) {
    visitor.on_attribute("min_size", m_attrs.min_size);
    visitor.on_attribute("max_size", m_attrs.max_size);
    visitor.on_attribute("aspect_ratio", m_attrs.aspect_ratio);
    visitor.on_attribute("density", m_attrs.density);
    visitor.on_attribute("fixed_ratio", m_attrs.fixed_ratio);
    visitor.on_attribute("fixed_size", m_attrs.fixed_size);
    visitor.on_attribute("clip", m_attrs.clip);
    visitor.on_attribute("flip", m_attrs.flip);
    visitor.on_attribute("step", m_attrs.step);
    visitor.on_attribute("offset", m_attrs.offset);
    visitor.on_attribute("variance", m_attrs.variance);
    visitor.on_attribute("scale_all_sizes", m_attrs.scale_all_sizes);
    return true;
}
