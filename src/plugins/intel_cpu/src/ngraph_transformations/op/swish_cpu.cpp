// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "swish_cpu.hpp"

MKLDNNPlugin::SwishNode::SwishNode(const ngraph::Output<ngraph::Node> & input, const float alpha)
        : Op({input}), m_alpha(alpha) {
    validate_and_infer_types();
}

std::shared_ptr<ngraph::Node> MKLDNNPlugin::SwishNode::clone_with_new_inputs(const ngraph::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<MKLDNNPlugin::SwishNode>(new_args.at(0), m_alpha);
}

bool MKLDNNPlugin::SwishNode::visit_attributes(ngraph::AttributeVisitor& visitor) {
    visitor.on_attribute("alpha", m_alpha);
    return true;
}

void MKLDNNPlugin::SwishNode::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}

float MKLDNNPlugin::SwishNode::get_alpha() const {
    return m_alpha;
}

