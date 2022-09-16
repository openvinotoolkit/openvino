// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <snippets/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

#include "snippets/snippets_isa.hpp"
#include "snippets/pass/convert_constants.hpp"
#include "snippets/op/subgraph.hpp"


ngraph::snippets::pass::ConvertConstantsToScalars::ConvertConstantsToScalars() {
    MATCHER_SCOPE(ConvertConstantsToScalars);
    auto constants = std::make_shared<pattern::op::Label>(pattern::any_input(),
                                                    [](std::shared_ptr<Node> n) {
                                                        return ngraph::is_type<ov::op::v0::Constant>(n);
                                                    });
    ngraph::graph_rewrite_callback callback = [this](ngraph::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::op::ConvertConstantsToScalars")
        auto constant = as_type_ptr<ov::op::v0::Constant>(m.get_match_root());
        auto scalar = std::make_shared<snippets::op::Scalar>(*constant);
        scalar->set_friendly_name(constant->get_friendly_name());
        ngraph::copy_runtime_info(constant, scalar);
        ngraph::replace_node(constant, scalar);

        return true;
    };
    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(constants), callback);
}

ngraph::snippets::pass::ConvertConstantsToParameters::ConvertConstantsToParameters() {
    MATCHER_SCOPE(ConvertConstantsToParameters);
    register_matcher(std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ngraph::snippets::op::Subgraph>(), matcher_name),
        [this](ngraph::pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ConvertConstantsToParameters");
        auto root = m.get_match_root();

        auto subgraph = ov::as_type_ptr<ngraph::snippets::op::Subgraph>(root);
        auto body = subgraph->get_body();

        ParameterVector new_parameters;
        OutputVector new_external_inputs = subgraph->input_values();

        for (auto& op : body->get_ops()) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(op);
            if (!(constant && ngraph::shape_size(constant->get_shape()) != 1ul))
                continue;

            auto parameter = std::make_shared<opset1::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());
            parameter->set_friendly_name(constant->get_friendly_name());
            ngraph::copy_runtime_info(constant, parameter);
            constant->output(0).replace(parameter->output(0));

            new_external_inputs.push_back(constant);
            new_parameters.push_back(parameter);
        }

        if (new_parameters.size() == 0)
            return false;

        body->add_parameters(new_parameters);
        body->validate_nodes_and_infer_types();
        subgraph->set_arguments(new_external_inputs);
        return true;
    });
}
