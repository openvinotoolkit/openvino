// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "device_subgraph.hpp"

using namespace ov::op::util;

ov::hetero::op::DeviceSubgraph::DeviceSubgraph(const ov::OutputVector& args,
                                               const std::shared_ptr<ov::Model>& body,
                                               const std::string& affinity)
    : SubGraphOp(args),
      _affinity{affinity} {
    SubGraphOp::set_function(body);
    constructor_validate_and_infer_types();
    for (size_t i = 0; i < body->get_parameters().size(); ++i)
        m_input_descriptions[0].push_back(std::make_shared<InvariantInputDescription>(i, i));
    for (size_t i = 0; i < body->get_output_size(); ++i)
        m_output_descriptions[0].push_back(std::make_shared<BodyOutputDescription>(i, i));
}

ov::hetero::op::DeviceSubgraph::DeviceSubgraph(const ov::NodeVector& args,
                                               const std::shared_ptr<ov::Model>& body,
                                               const std::string& affinity)
    : DeviceSubgraph(as_output_vector(args), body, affinity) {}

std::shared_ptr<ov::Node> ov::hetero::op::DeviceSubgraph::clone_with_new_inputs(const ov::OutputVector& inputs) const {
    return std::make_shared<DeviceSubgraph>(inputs, body().clone(), _affinity);
}

void ov::hetero::op::DeviceSubgraph::validate_and_infer_types() {
    ov::ParameterVector old_parameters;
    for (auto op : body_ptr()->get_parameters()) {
        old_parameters.push_back(op);
    }

    for (size_t i = 0; i < get_input_size(); ++i) {
        body_ptr()->replace_parameter(
            i,
            std::make_shared<ov::op::v0::Parameter>(get_input_element_type(i), get_input_partial_shape(i)));
    }

    body_ptr()->validate_nodes_and_infer_types();

    for (size_t i = 0; i < body_ptr()->get_parameters().size(); i++) {
        body_ptr()->get_parameters()[i]->set_friendly_name(old_parameters[i]->get_friendly_name());
    }

    set_output_size(body_ptr()->get_output_size());
    for (size_t i = 0; i < get_output_size(); ++i) {
        set_output_type(i, body_ptr()->get_output_element_type(i), body_ptr()->get_output_partial_shape(i));
    }
}

bool ov::hetero::op::DeviceSubgraph::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("affinity", _affinity);
    visitor.on_attribute("body", body_ptr());
    visitor.on_attribute("input_descriptions", m_input_descriptions[0]);
    visitor.on_attribute("output_descriptions", m_output_descriptions[0]);
    return true;
}