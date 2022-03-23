// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/constant_folding.hpp"

#include <memory>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "transformations/utils/utils.hpp"
#include "transformations/op_conversions/fq_decomposition.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/itt.hpp"

#include "ngraph/pass/visualize_tree.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::ConstantFolding, "Snippets::ConstantFolding", 0);

namespace ngraph {
namespace snippets {
namespace pass {

namespace {
void parameters_to_constants(std::shared_ptr<ov::Model>& body, const std::unordered_map<std::string, std::shared_ptr<opset1::Constant>>& constant_input_ids) {
    auto ops = body->get_ops();
    for (auto& op : ops) {
        auto parameter = as_type_ptr<ngraph::opset1::Parameter>(op);
        if (parameter == nullptr) {
            continue;
        }

        auto it = constant_input_ids.find(parameter->get_friendly_name());
        if (it == constant_input_ids.end()) {
            continue;
        }

        const auto& subgraph_constant = it->second;
        auto body_constant = subgraph_constant->clone_with_new_inputs({});

        body_constant->set_friendly_name(parameter->get_friendly_name());
        for (auto input : parameter->output(0).get_target_inputs()) {
            input.replace_source_output(body_constant->output(0));
        }

        body->remove_parameter(parameter);
    }
    body->validate_nodes_and_infer_types();
}

void constants_to_parameters(
    std::shared_ptr<ngraph::snippets::op::Subgraph>& subgraph,
    std::shared_ptr<ov::Model>& body,
    const std::unordered_map<std::string, std::shared_ptr<opset1::Parameter>>& parameter_input_ids) {
    std::vector<ngraph::Output<Node>> new_inputs;
    new_inputs.reserve(subgraph->get_input_size());
    for (auto i = 0; i < subgraph->get_input_size(); ++i) {
        auto input = subgraph->get_input_source_output(i);
        if (!is_type<opset1::Constant>(input.get_node_shared_ptr())) {
            new_inputs.push_back(input);
        }
    }

    auto ops = body->get_ops();
    for (auto& op : ops) {
        auto constant = as_type_ptr<ngraph::opset1::Constant>(op);
        if ((constant == nullptr) || (ngraph::shape_size(constant->get_output_shape(0)) == 1ul)) {
            continue;
        }

        new_inputs.push_back(constant);
        auto parameter = std::make_shared<opset1::Parameter>(constant->get_element_type(), constant->output(0).get_partial_shape());

        parameter->set_friendly_name(constant->get_friendly_name());
        for (auto input : constant->output(0).get_target_inputs()) {
            input.replace_source_output(parameter->output(0));
        }

        body->add_parameters(ParameterVector{parameter});
    }
    body->validate_nodes_and_infer_types();

    const auto new_subgraph = subgraph->clone_with_new_inputs(new_inputs);
    replace_node(subgraph, new_subgraph);

    new_subgraph->set_friendly_name(subgraph->get_friendly_name());
    copy_runtime_info(subgraph, new_subgraph);
}

} // namespace

ConstantFolding::ConstantFolding() {
    auto wrapper = ngraph::pattern::wrap_type<ngraph::snippets::op::Subgraph>();

    ngraph::graph_rewrite_callback callback = [this](pattern::Matcher& m) {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::ConstantFolding");

        auto subgraph = ngraph::as_type_ptr<ngraph::snippets::op::Subgraph>(m.get_match_root());
        if (transformation_callback(subgraph)) {
            return false;
        }

        auto body = subgraph->get_body();

        std::unordered_map<std::string, std::shared_ptr<opset1::Constant>> constant_input_ids;
        std::unordered_map<std::string, std::shared_ptr<opset1::Parameter>> parameter_input_ids;
        for (auto i = 0;  i < subgraph->get_input_size(); ++i) {
            auto constant = as_type_ptr<opset1::Constant>(subgraph->get_input_node_shared_ptr(i));
            if (constant != nullptr) {
                constant_input_ids.emplace(constant->get_friendly_name(), constant);
                continue;
            }

            auto parameter = as_type_ptr<opset1::Parameter>(subgraph->get_input_node_shared_ptr(i));
            if (parameter != nullptr) {
                parameter_input_ids.emplace(parameter->get_friendly_name(), parameter);
                continue;
            }
        }
        parameters_to_constants(body, constant_input_ids);

        ngraph::pass::Manager manager(get_pass_config());
        manager.register_pass<ngraph::pass::ConstantFolding>();
        manager.run_passes(body);

        constants_to_parameters(subgraph, body, parameter_input_ids);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(wrapper, "snippets::pass::ConstantFolding");
    this->register_matcher(m, callback);
}

} // namespace pass
} // namespace snippets
} // namespace ngraph