// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/extract_constants.hpp"

#include "openvino/opsets/opset1.hpp"
#include "snippets/itt.hpp"


bool ov::snippets::pass::ExtractConstants::run_on_subgraph(const std::shared_ptr<op::Subgraph>& subgraph) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractConstants");
    auto body = subgraph->body_ptr();

    ParameterVector new_parameters;
    OutputVector new_external_inputs = subgraph->input_values();

    for (auto& op : body->get_ops()) {
        auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
        if (!constant || ov::shape_size(constant->get_shape()) == 1ul)
            continue;

        const auto child = constant->get_output_target_inputs(0).begin()->get_node()->shared_from_this();
        if (ov::snippets::op::Subgraph::constant_input_should_be_inside_body(child))
            continue;

        auto parameter = std::make_shared<ov::op::v0::Parameter>(constant->get_element_type(), constant->get_shape());
        ov::replace_output_update_name(constant->output(0), parameter->output(0));

        new_external_inputs.push_back(constant);
        new_parameters.push_back(parameter);
    }

    if (new_parameters.size() != 0) {
        body->add_parameters(new_parameters);
        body->validate_nodes_and_infer_types();
        subgraph->set_arguments(new_external_inputs);
        return true;
    }

    return false;
}
