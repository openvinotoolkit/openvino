// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "power_static.hpp"

constexpr ngraph::NodeTypeInfo MKLDNNPlugin::PowerStaticNode::type_info;

MKLDNNPlugin::PowerStaticNode::PowerStaticNode(const ngraph::Output<Node> &data,
                                               const float &power,
                                               const float &scale,
                                               const float &shift,
                                               const ngraph::element::Type output_type)
    : Op({data}), scale(scale), power(power), shift(shift), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> MKLDNNPlugin::PowerStaticNode::clone_with_new_inputs(const ngraph::OutputVector &new_args) const {
    if (new_args.size() != 1) {
        throw ngraph::ngraph_error("Incorrect number of new arguments");
    }

    return std::make_shared<MKLDNNPlugin::PowerStaticNode>(new_args.at(0), this->power, this->scale, this->shift, this->m_output_type);
}

void MKLDNNPlugin::PowerStaticNode::validate_and_infer_types() {
    set_output_type(0, m_output_type == ngraph::element::undefined ? get_input_element_type(0) : m_output_type, get_input_partial_shape(0));
}

bool MKLDNNPlugin::PowerStaticNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("scale", scale);
    visitor.on_attribute("power", power);
    visitor.on_attribute("shift", shift);
    return true;
}
