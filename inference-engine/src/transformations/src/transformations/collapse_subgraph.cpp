// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/collapse_subgraph.hpp"
#include "ngraph_ops/subgraph.hpp"

#include <memory>
#include <vector>
#include <cassert>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph/pass/visualize_tree.hpp>

template <typename T>
ngraph::OutputVector as_output_vector(const T& args) {
    ngraph::OutputVector output_vector;
    for (auto arg : args) {
        output_vector.push_back(arg);
    }
    return output_vector;
}


ngraph::pass::CollapseSubgraph::CollapseSubgraph() {
    using namespace std;
    using ngraph::pattern::op::NodePredicate;

    std::vector<NodePredicate> initial_ops {
            NodePredicate([](std::shared_ptr<Node> n) { return !!as_type_ptr<opset1::Convolution>(n); }),
    };

    std::vector<NodePredicate> continuation_ops {
        //[](std::shared_ptr<Node> n) { return !!as_type_ptr<opset1::Add>(n); },
        //[](std::shared_ptr<Node> n) { return !!as_type_ptr<opset1::Relu>(n); },
        //[](std::shared_ptr<Node> n) { return !!as_type_ptr<opset1::FakeQuantize>(n); },
        //[](std::shared_ptr<Node> n) { return !!as_type_ptr<opset1::Convolution>(n); },
    };

    ngraph::graph_rewrite_callback continuation_callback = [initial_ops](ngraph::pattern::Matcher &m) {
        auto node = m.get_match_root();
        // Check if at least one of the inputs are in initial_ops list or already formed Subgraph op
        auto inputs = node->inputs();

        // Sort inputs into categories:
        vector<Input<Node>> input_initial_ops;  // inputs that matches one of initial_ops
        std::unordered_set<std::shared_ptr<Node>> input_subgraphs;    // inputs that are already subgraphs

        for (auto input : inputs) {
            auto parent = input.get_source_output().get_node_shared_ptr();
            std::cerr << "Considered: " << parent->get_friendly_name() << "\n";
            if (std::any_of(initial_ops.begin(), initial_ops.end(), [parent](NodePredicate predicate) {
                    return predicate(parent); })) {
                input_initial_ops.push_back(input);
            }
            if (auto subgraph = as_type_ptr<ngraph::op::Subgraph>(parent)) {    // FIXME: we do this check later
                input_subgraphs.insert(subgraph);
            }
        }

        std::cerr << input_initial_ops.size() << " " << input_subgraphs.size() << "\n";

        if (true || input_initial_ops.empty()) {
            if (false && input_subgraphs.empty()) {
                // TODO Should we collapse nodes from continuation_ops list to a subgraph?
                // Probably such sub-graph if they don't find an initial node later, will be exploded
            } else {
                // There is at least one subgraph op node among inputs and the current op is also in the list of ops that can be fused
                // Get the first subgraph op node and expand the body by adding nodes from other sub-graph op nodes on inputs
                //

                // Parameters for a new body
                ParameterVector parameters;
                parameters.reserve(2*inputs.size()); // 2* is a rough estimate

                // Inputs for a new subgraph node
                OutputVector subgraph_inputs;
                subgraph_inputs.reserve(parameters.capacity());

                // Vector of producing ports, one per each input of original node that is going to be replicated in the body
                OutputVector new_inputs;
                new_inputs.reserve(inputs.size());

                for (auto input : inputs) {
                    if (auto subgraph = as_type_ptr<ngraph::op::Subgraph>(input.get_source_output().get_node_shared_ptr())) {
                        // prepare to suck existing nodes from the sub-graph into the currently constructed sub-graph
                        const auto& subgraph_parameters = subgraph->get_body()->get_parameters();
                        assert(subgraph_parameters.size() == subgraph->get_input_size());

                        // reuse parameters from the subgraph node, don't create new ones; we will assembly a new subgraph by using nodes from the old one
                        // not all the parameters from the subgraph should be replicated
                        // a parameter that consumes output of another subgraph that is consumed by newly built subgraph should be omitted
                        for (size_t i = 0; i < subgraph_parameters.size(); ++i) {
                            if (!input_subgraphs.count(subgraph->input_value(i).get_node_shared_ptr())) {
                                parameters.push_back(subgraph_parameters[i]);
                                subgraph_inputs.push_back(subgraph->input_value(i));
                            } else {
                                // TODO: Track such inputs as well because later they have to be connected to other subgraph internals
                                assert(false);
                            }
                        }

                        size_t source_output_index = input.get_source_output().get_index();
                        // output port index corresponds to Result op in a body of the subgraph with the same index
                        auto source_result = subgraph->get_body()->get_results()[source_output_index];
                        new_inputs.push_back(source_result->input_value(0));    // Result op has a single input

                        // TODO: how not to loose other consumers of the output port that don't match the current node (a side output)?
                        // TODO: handle it for each output port in each distinct subgraph op node that among the inputs separately from this loop
                    } else {
                        // create a new Parameter if the input is not a subgraph op node
                        auto new_parameter = std::make_shared<opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
                        parameters.push_back(new_parameter);
                        new_inputs.push_back(new_parameter->output(0)); // Parameter op has a single output
                        subgraph_inputs.push_back(input.get_source_output());
                    }
                }

                auto body_node = node->copy_with_new_inputs(new_inputs);

                ResultVector results;
                results.reserve(body_node->get_output_size() + input_subgraphs.size());    // rough estimate

                // All consumers for each output outside the body that corresponds to each result node inside the body
                std::vector<std::set<Input<Node>>> result_inputs;
                result_inputs.reserve(results.capacity());

                // Collect outputs for the body. The set of outputs consists of two part: the first part is side consumers
                // of subgraphs that are consumed by the currently constructed subgraph op node; the second part is body_node
                // own outputs.

                // Collect the side output of the subgraph op nodes first
                for (auto node_subgraph : input_subgraphs) {
                    auto subgraph = as_type_ptr<ngraph::op::Subgraph>(node_subgraph);
                    for (auto output : subgraph->outputs()) {
                        bool has_side_consumers = false;
                        for (auto target_input : output.get_target_inputs()) {
                            // Check if target_input is in a list of considered nodes (all sub-graphs and the node)
                            auto target_node = target_input.get_node()->shared_from_this();     // suppose there is a consumer TODO: need to worry?
                            bool is_side_consumer = !input_subgraphs.count(target_node) && target_node != node;
                            if (is_side_consumer) {
                                if (!has_side_consumers) {
                                    // Create a new Result op node inside the body
                                    // TODO: what about reuse the existing Result op nodes in subgraphs as it is done for Parameters?
                                    results.push_back(std::make_shared<opset1::Result>(
                                            subgraph->get_body()->get_results()[output.get_index()]->input_value(0)));
                                    result_inputs.push_back({});
                                    has_side_consumers = true;
                                }

                                assert(!result_inputs.back().count(target_input));
                                // save target input port outside the body
                                result_inputs.back().insert(target_input);
                            }
                        }
                    }
                }

                assert(results.size() == result_inputs.size());

                assert(node->get_output_size() == body_node->get_output_size());
                // Then collect outputs of body_node
                for (auto output : node->outputs()) {
                    results.push_back(std::make_shared<opset1::Result>(body_node->output(output.get_index())));
                    result_inputs.push_back(output.get_target_inputs());
                }

                auto new_function = std::make_shared<Function>(results, parameters);

                {
                    std::vector<std::shared_ptr<ngraph::Function>> module{new_function};
                    ngraph::pass::VisualizeTree("/localdisk/slyalin/subgraph.png").run_on_module(module);
                }

                auto subgraph = std::make_shared<op::Subgraph>(subgraph_inputs, new_function);
                copy_runtime_info(node, subgraph);
                // Cannot use replace_node because multiple nodes are replaced; make the replacement manually
                // FIXME: other propertie except data dependencies are not transfered
                for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
                    for (auto target_input : result_inputs[i]) {
                        target_input.replace_source_output(subgraph->output(i));
                    }
                }

                subgraph->validate_and_infer_types();

                std::cerr << "Replacement done by: " << subgraph->get_friendly_name() << "\n";


//                ParameterVector parameters;
//                OutputVector parent_outputs;
//                for (auto input : inputs) {
//                    auto parent = input.get_source_output().get_node_shared_ptr();
//                    if (auto subgraph = as_type_ptr<ngraph::op::Subgraph>(parent)) {
//                        auto sub_parameters = subgraph->get_body()->get_parameters();
//                        parameters.insert(parameters.end(), sub_parameters.begin(), sub_parameters.end());
//                        auto sub_inputs = subgraph->input_values();
//                        parent_outputs.insert(parent_outputs.end(), sub_inputs.begin(), sub_inputs.end());
//                    } else {
//
//                    }
//                }

// Remove this quick durty implementation
//                auto subgraph = as_type_ptr<ngraph::op::Subgraph>(inputs[0].get_source_output().get_node_shared_ptr());
//                subgraph = subgraph->copy_with_new_inputs(::as_output_vector(subgraph->inputs()));
//                auto body_result = subgraph->get_body()->get_result();
//                auto body_result_parent = body_result->input_value(0).get_node_shared_ptr();
//                auto body_node = node->copy_with_new_inputs(body_result_parent->outputs());
//                replace_node(body_result_parent, body_node);
//                replace_node(node, subgraph);
            }
        } else {
            // Crate a new Subgraph op
            // FIXME: Probably requires gathering more than one op, but for now we captured only one (the node)

            // Create parameters for all inputs
            ParameterVector parameters; parameters.reserve(inputs.size());
            for (auto input : inputs) {
                parameters.push_back(std::make_shared<opset1::Parameter>(input.get_element_type(), input.get_partial_shape()));
            }

            auto body_node = node->copy_with_new_inputs(::as_output_vector(parameters));

            ResultVector results; results.reserve(body_node->get_output_size());
            for (auto output : body_node->outputs()) {
                results.push_back(std::make_shared<opset1::Result>(output));
            }

            auto subgraph = std::make_shared<op::Subgraph>(node->input_values(), std::make_shared<Function>(results, parameters));
            copy_runtime_info(node, subgraph);
            replace_node(node, subgraph);
            std::cerr << "Replacement done by: " << subgraph->get_friendly_name() << "\n";
        }

        return true;
    };

    // Register all continuation matchers; check for initial op inside a matcher
    for (auto predicate : continuation_ops) {
        auto p_node = std::make_shared<pattern::op::Label>(element::f32, Shape{}, predicate);
        auto m = std::make_shared<ngraph::pattern::Matcher>(p_node, "CollapseSubgraphPart");
        this->add_matcher(m, continuation_callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
}