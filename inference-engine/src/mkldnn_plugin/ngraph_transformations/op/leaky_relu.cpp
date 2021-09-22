// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "leaky_relu.hpp"

constexpr ngraph::NodeTypeInfo MKLDNNPlugin::LeakyReluNode::type_info;

MKLDNNPlugin::LeakyReluNode::LeakyReluNode(const ngraph::Output<ngraph::Node> &data,
                                           const float &negative_slope,
                                           const ngraph::element::Type output_type)
    : Op({data}), m_negative_slope(negative_slope), m_output_type(output_type) {
    constructor_validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> MKLDNNPlugin::LeakyReluNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MKLDNNPlugin::LeakyReluNode>(new_args.at(0), m_negative_slope, m_output_type);
}

void MKLDNNPlugin::LeakyReluNode::validate_and_infer_types() {
    set_output_type(
        0,
        m_output_type == ngraph::element::undefined ? get_input_element_type(0) : m_output_type,
        get_input_partial_shape(0));
}

bool MKLDNNPlugin::LeakyReluNode::visit_attributes(ngraph::AttributeVisitor &visitor) {
    visitor.on_attribute("negative_slope", m_negative_slope);
    return true;
}
