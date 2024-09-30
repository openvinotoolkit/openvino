// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/pass/mlp_tokenization.hpp"
#include "snippets/pass/collapse_subgraph.hpp"

#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/opsets/opset4.hpp"
#include "snippets/itt.hpp"
#include "snippets/utils/tokenization_utils.hpp"

ov::snippets::pass::TokenizeMLPSnippets::TokenizeMLPSnippets(const SnippetsTokenization::Config& config) {
    MATCHER_SCOPE(TokenizeMLPSnippets);
    using namespace ov::pass::pattern;
    auto constant = wrap_type<ov::op::v0::Constant>();
    // todo: check transpose_b = true
    // auto fc_matmul = ov::pass::pattern::wrap_type<ov::opset1::MatMul>({any_input(), constant});

    ov::pass::pattern::op::NodePredicate fc_pred = [](const std::shared_ptr<ov::Node>& n){
        if (auto mm = ov::as_type_ptr<ov::opset1::MatMul>(n)) {
            return ov::is_type<opset1::Constant>(mm->get_input_node_shared_ptr(1)) &&
                   mm->get_transpose_b();
        }
        return false;
    };
    auto fc_matmul = std::make_shared<ov::pass::pattern::op::Label>(any_input(), fc_pred);

    auto callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher &m) {
        OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::op::TokenizeMLPSnippets")
        const auto last_matmul = m.get_match_root();
        if (transformation_callback(last_matmul)) {
            return false;
        }

        NodeVector fused_nodes {last_matmul};
        auto fuse_single_in_chain = [&fused_nodes](const std::shared_ptr<ov::Node> start) {
            auto in = start;
            while (TokenizeSnippets::AppropriateForSubgraph(in) && in->get_input_size() == 1) {
                fused_nodes.push_back(in);
                in = in->get_input_node_shared_ptr(0);
            }
            return in;
        };
        const auto last_not_fused = fuse_single_in_chain(last_matmul->get_input_node_shared_ptr(0));
        // Next node to be fused must have 2 inputs. Abort if not true
        if (!TokenizeSnippets::AppropriateForSubgraph(last_not_fused) || last_not_fused->get_input_size() != 2)
            return false;
        fused_nodes.push_back(last_not_fused);

        const auto left_not_fused = fuse_single_in_chain(last_not_fused->get_input_node_shared_ptr(0));
        const auto right_not_fused = fuse_single_in_chain(last_not_fused->get_input_node_shared_ptr(1));
        // Eltwise fusing chains must be interrupted by FullyConnected nodes
        if (!m.match(left_not_fused) || !m.match(right_not_fused))
            return false;
        fused_nodes.push_back(left_not_fused);
        fused_nodes.push_back(right_not_fused);


        /* ====== Subgraph creation ======= */

        ov::OutputVector body_inputs, subgraph_inputs;
        ov::ParameterVector body_parameters;
        ov::ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;

        auto create_body_inputs = [&](const std::shared_ptr<ov::Node>& node) -> void {
            for (size_t i = 0; i < node->get_input_size(); ++i) {
                const auto input = node->input(i);
                const auto parent = input.get_source_output().get_node_shared_ptr();
                if (std::find(ordered_ops.begin(), ordered_ops.end(), parent) == ordered_ops.end()) {
                    auto parameter = std::make_shared<ov::opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
                    body_parameters.push_back(parameter);
                    body_parameters.back()->set_friendly_name(input.get_node()->get_friendly_name());
                    body_inputs.push_back(parameter->output(0));

                    subgraph_inputs.push_back(input.get_source_output());

                    node->input(i).replace_source_output(parameter);
                }
            }
        };

        for (const auto& op : ordered_ops) {
            create_body_inputs(op);
            op->clear_control_dependencies();
            fused_names += op->get_friendly_name() + ",";
        }

        for (const auto& output : last_node->outputs()) {
            subgraph_result_inputs.push_back(output.get_target_inputs());
        }
        for (const auto& output : last_node->outputs()) {
            body_results.push_back(std::make_shared<ov::opset1::Result>(last_node->output(output.get_index())));
        }

        if (body_results.size() != subgraph_result_inputs.size()) {
            OPENVINO_THROW("body results and node results size mismatch during subgraph collapse");
        }

        auto body = op::create_body(last_node->get_friendly_name(), body_results, body_parameters);
        auto subgraph = std::make_shared<op::Subgraph>(subgraph_inputs, body);
        // Copy runtime info from last node to subgraph - to copy topological order
        copy_runtime_info(last_node, subgraph);
        subgraph->set_friendly_name(last_node->get_friendly_name());

        for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
            for (const auto& target_input : subgraph_result_inputs[i]) {
                target_input.replace_source_output(subgraph->output(i));
            }
        }
        op::update_out_tensor_name(subgraph);

        subgraph->validate_and_infer_types();

        auto act_body = subgraph->body_ptr();
        for (size_t i = 0; i < act_body->get_parameters().size(); i++) {
            act_body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }
        subgraph->get_rt_info()["originalLayersNames"] = fused_names;
        subgraph->set_virtual_port_count(hidden_virtual_ports_count);

        // mark the Subgraph as Completed to not allow Snippets to include any nodes into the MHA Subgraph in common Tokenization
        SetSnippetsSubgraphType(subgraph, SnippetsSubgraphType::Completed);





        ov::replace_node();

    };

    auto matcher = std::make_shared<ov::pass::pattern::Matcher>(fc_matmul, matcher_name);
    register_matcher(matcher, callback);
}
