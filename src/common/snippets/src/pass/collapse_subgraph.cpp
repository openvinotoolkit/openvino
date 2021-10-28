// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/pass/filter_fused.hpp"
#include "snippets/op/subgraph.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/op/loop.hpp>

#include <memory>
#include <vector>
#include <cassert>
#include <queue>
#include <string>
#include <numeric>
#include <climits>

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::StartSubgraph, "Snippets::StartSubgraph", 0);
NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::AttachToSubgraph, "Snippets::AttachToSubgraph", 0);
NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::TokenizeSnippets, "Snippets::TokenizeSnippets", 0);

using namespace ngraph;
using namespace snippets;

namespace {

auto outputs_are_not_broadcastable(const std::shared_ptr<ngraph::Node>& node) -> bool {
    auto outputs = node->outputs();
    auto find_smallest_output_shape = [](const std::vector<ngraph::Output<ngraph::Node>>& outputs) -> ngraph::Shape {
        return std::accumulate(std::begin(outputs), std::end(outputs), ngraph::Shape(outputs.begin()->get_shape()),
            [](ngraph::Shape other_shape, ngraph::Output<ngraph::Node> output){
                return ngraph::shape_size(output.get_shape()) < ngraph::shape_size(other_shape) ? output.get_shape() : other_shape;
            });
    };
    auto ref_shape = find_smallest_output_shape(outputs);

    auto check_shapes_broadcastable = [ref_shape](const ngraph::Output<ngraph::Node>& output) -> bool {
        auto other_shape = output.get_shape();

        if (other_shape.size() != ref_shape.size()) {
            return false;
        }

        return std::inner_product(std::begin(other_shape), std::end(other_shape), std::begin(ref_shape), true,
                            std::logical_and<bool>(), [](ngraph::Shape::value_type lsh, ngraph::Shape::value_type rsh){
                                return rsh == 1 || lsh == rsh;
                            });
    };

    return std::find_if_not(std::begin(outputs), std::end(outputs), check_shapes_broadcastable) != std::end(outputs);
};

auto has_subgraph_as_input(std::shared_ptr<Node> node) -> bool {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::has_subgraph_as_input")
    for (const auto& parent : ngraph::as_node_vector(node->input_values())) {
        if (ov::is_type<snippets::op::Subgraph>(parent)) {
            return true;
        }
    }
    return false;
}

auto is_lo(std::shared_ptr<Node> n) -> bool {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::is_lo")
    auto is_lob = [](std::shared_ptr<Node> n) -> bool {
        return ov::is_type<opset1::Add>(n)
            || ov::is_type<opset1::Divide>(n)
            || ov::is_type<opset1::Equal>(n)
            || ov::is_type<opset1::FloorMod>(n)
            || ov::is_type<opset1::Greater>(n)
            || ov::is_type<opset1::GreaterEqual>(n)
            || ov::is_type<opset1::Less>(n)
            || ov::is_type<opset1::LessEqual>(n)
            || ov::is_type<opset1::LogicalAnd>(n)
            || ov::is_type<opset1::LogicalOr>(n)
            || ov::is_type<opset1::LogicalXor>(n)
            || ov::is_type<opset1::Maximum>(n)
            || ov::is_type<opset1::Minimum>(n)
            || ov::is_type<opset1::Mod>(n)
            || ov::is_type<opset1::Multiply>(n)
            || ov::is_type<opset1::NotEqual>(n)
            || ov::is_type<opset1::PRelu>(n)
            || ov::is_type<opset1::Power>(n)
            || ov::is_type<opset1::SquaredDifference>(n)
            || ov::is_type<opset1::Subtract>(n)
            || ov::is_type<opset1::Xor>(n);
    };

    auto is_lou = [](std::shared_ptr<Node> n) -> bool {
        return ov::is_type<opset1::Abs>(n)
            || ov::is_type<opset1::Clamp>(n)
            || ov::is_type<opset1::Elu>(n)
            || ov::is_type<opset1::Erf>(n)
            || ov::is_type<opset1::Exp>(n)
            || ov::is_type<opset1::LogicalNot>(n)
            || ov::is_type<opset1::Negative>(n)
            || ov::is_type<opset1::Relu>(n)
            || ov::is_type<opset1::Sigmoid>(n)
            || ov::is_type<opset1::Sqrt>(n)
            || ov::is_type<opset1::Tanh>(n)
            || ov::is_type<ngraph::op::v4::HSwish>(n);
    };
    return is_lou(n) || is_lob(n);
}

auto has_supported_in_out(std::shared_ptr<Node> n) -> bool {
    for (auto in : n->inputs()) {
        if (in.get_tensor().get_element_type() != ngraph::element::f32) {
            return false;
        }

        if (in.get_partial_shape().is_dynamic()) {
            return false;
        }

        if (in.get_partial_shape().is_static() && in.get_shape().size() > 6) {
            return false;
        }
    }

    for (auto out : n->outputs()) {
        if (out.get_tensor().get_element_type() != ngraph::element::f32) {
            return false;
        }

        if (out.get_partial_shape().is_dynamic()) {
            return false;
        }

        if (out.get_partial_shape().is_static() && out.get_shape().size() > 6) {
            return false;
        }

        for (auto in_out : out.get_target_inputs()) {
            if (ov::is_type<ngraph::op::v5::Loop>(in_out.get_node()->shared_from_this())) {
                return false;
            }
        }
    }

    return true;
}

auto has_result_child(std::shared_ptr<Node> node) -> bool {
    for (const auto &child : node->get_users()) {
        if (ov::is_type<ngraph::opset1::Result>(child)) {
            return true;
        }
    }
    return false;
}

auto get_num_result_children(std::shared_ptr<Node> node) -> size_t {
    size_t result = 0;
    for (const auto &child : node->get_users()) {
        if (ov::is_type<ngraph::opset1::Result>(child)) {
            result++;
        }
    }
    return result;
}
// Need to update tensor name manually, since MKLDNNGraph::Replicate() looks at input.get_tensor().get_name();
// If subgraph->get_output_size() == 1, then the name will be restored correctly from the node name
// todo: remove this function when MKLDNNGraph::Replicate() will rely only on node->get_friendly_name()
auto update_out_tensor_name(std::shared_ptr<ngraph::snippets::op::Subgraph> subgraph) -> void {
    if (subgraph->get_output_size() != 1) {
        bool not_set = true;
        for (unsigned int i = 0; i < subgraph->get_output_size() && not_set; i++) {
            for (auto &in : subgraph->get_output_target_inputs(i)) {
                if (ov::is_type<opset1::Result>(in.get_node())) {
                    NGRAPH_SUPPRESS_DEPRECATED_START
                    subgraph->output(i).get_tensor_ptr()->set_name(subgraph->get_friendly_name());
                    NGRAPH_SUPPRESS_DEPRECATED_END
                    not_set = false;
                    break;
                }
            }
        }
    }
}
} // namespace

bool ngraph::snippets::pass::AppropriateForSubgraph(std::shared_ptr<Node> n) {
    return is_lo(n) && has_supported_in_out(n);
}

ngraph::snippets::pass::StartSubgraph::StartSubgraph() : MatcherPass() {
    MATCHER_SCOPE(StartSubgraph);
    auto label = std::make_shared<pattern::op::Label>(pattern::any_input(),
        [](std::shared_ptr<Node> n) {
            return GetSnippetsNodeType(n) == SnippetsNodeType::SubgraphStart;
        });
    auto callback = [](ngraph::pattern::Matcher &m) -> bool {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::StartSubgraph_callback")
        auto node = m.get_match_root();
        remark(1) << "Match root (Start): "
                  << node->get_friendly_name()
                  << " " << node
                  << " Creating new snippet - no input subgraphs found" << std::endl;

        auto subgraph = op::Subgraph::wrap_node_as_subgraph(node);
        subgraph->get_rt_info()["originalLayersNames"] = ov::make_variant(node->get_friendly_name());
        ngraph::replace_node(node, subgraph);
        update_out_tensor_name(subgraph);

        remark(1) << "Replacement (new) done for: "
                  << subgraph->get_friendly_name()
                  << " with " << subgraph->inputs().size()
                  << " inputs and " << subgraph->outputs().size()
                  << " outputs and " << subgraph->get_body()->get_ops().size() << " ops total\n";
        return true;
    };
    auto matcher = std::make_shared<ngraph::pattern::Matcher>(label);
    register_matcher(matcher, callback);
}

ngraph::snippets::pass::AttachToSubgraph::AttachToSubgraph() : MatcherPass() {
    MATCHER_SCOPE(AttachToSubgraph);
    enum continuation_strategy {
        reset,
        abort
    };

    continuation_strategy strategy = continuation_strategy::reset;
    auto label = std::make_shared<pattern::op::Label>(pattern::any_input(),
        [](std::shared_ptr<Node> n) {
            return GetSnippetsNodeType(n) == SnippetsNodeType::SubgraphBody && has_subgraph_as_input(n);
        });
    ngraph::graph_rewrite_callback callback = [strategy](ngraph::pattern::Matcher &m) -> bool {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::AttachToSubgraph_callback")
        auto node = m.get_match_root();

        remark(1) << "Match root (Attach): " << node->get_friendly_name() << " " << node << std::endl;

        auto abort_with_strategy = [&node, strategy](const std::string message_reset,
                                                     const std::string message_abort = "", int priority = 3) {
            if (strategy == continuation_strategy::reset) {
                remark(priority) << message_reset << std::endl;
                auto single_node_subgraph = op::Subgraph::wrap_node_as_subgraph(node);
                single_node_subgraph->validate_and_infer_types();
                ngraph::replace_node(node, single_node_subgraph);
                return true;
            } else if (strategy == continuation_strategy::abort) {
                if (!message_abort.empty()) {
                    remark(priority) << message_abort << std::endl;
                }
            }
            return false;
        };
        // inputs that are already subgraphs
        std::unordered_set<std::shared_ptr<Node>> input_subgraphs;
        // clone bodies because we need a rollback if loop is found
        std::map<std::shared_ptr<Node>, std::shared_ptr<ngraph::Function>> clones;

        ParameterVector body_parameters;
        // inputs to merged subgraph
        OutputVector external_inputs;
        // inputs to the node before merge to subgraph
        OutputVector internal_inputs;

        auto input_values = node->input_values();
        /* 
        * Called with subgraph->input_value(i) arg and used to 
        * Check that the attached node input subgraph has the same input as the node itself.
        * If true, then ternary merge is intiated.
        *        input
        *        /   \
        *  sungraph--node
        */
        auto is_recurrent = [&input_values](const ngraph::Output<ngraph::Node>& to_find) -> bool {
            for (const auto& in : input_values) {
                if (in == to_find)
                    return true;
            }
            return false;
        };

        std::string fusedNames;
        const auto getFusedNames = [](const std::shared_ptr<Node> n) -> std::string {
            auto rt_info = n->get_rt_info();
            auto it = rt_info.find("originalLayersNames");
            if (it != rt_info.end()) {
                auto value = std::dynamic_pointer_cast<ngraph::VariantImpl<std::string>>(it->second);
                return value->get() + ",";
            } else {
                return "";
            }
        };
        /*
         * Checks if the passed node introduces loop dependency for given topological bounds (pair of maxParentOrder, minChildOrder).
         * The bounds are presumed to be without dependency. The bounds are updated if no dependency is introduced by the node.
        */
        const auto cyclicDependencyIsIntoduced = [&node](const std::shared_ptr<Node>& nodeToExamine, std::pair<int64_t, int64_t>& currentBounds) -> bool {
            auto getTopologicalOrder = [](const std::shared_ptr<Node>& n) -> int64_t {
                auto &rt = n->get_rt_info();
                const auto rinfo = rt.find("TopologicalOrder");
                assert(rinfo != rt.end() && "TopologicalOrder is not found in rt_info");
                return ov::as_type_ptr<ngraph::VariantWrapper<int64_t>>(rinfo->second)->get();
            };
            assert(currentBounds.first < currentBounds.second && "Invalid currentBounds passed");
            const auto &parentNodes = ngraph::as_node_vector(nodeToExamine->input_values());
            const int64_t maxParentOrder = std::accumulate(parentNodes.begin(), parentNodes.end(), currentBounds.first,
                                                            [&getTopologicalOrder](int64_t maxOrder, std::shared_ptr<Node> n){
                                                                if (ngraph::op::is_constant(n) || ngraph::op::is_parameter(n))
                                                                    return maxOrder;
                                                                return std::max(maxOrder, getTopologicalOrder(n));
                                                            });
            const auto &childNodes = nodeToExamine->get_users();
            // Skip the node being attached, since it will be a part of subgraph and can't introduce loop dependency
            const int64_t minChildOrder = std::accumulate(childNodes.begin(), childNodes.end(), currentBounds.second,
                                                            [&getTopologicalOrder, &node](int64_t minOrder, std::shared_ptr<Node> n){
                                                                if (ngraph::op::is_constant(n) || ngraph::op::is_parameter(n) || n == node)
                                                                    return minOrder;
                                                                return std::min(minOrder, getTopologicalOrder(n));
                                                            });
            if (maxParentOrder < minChildOrder) {
                currentBounds = std::pair<int64_t, int64_t>(maxParentOrder, minChildOrder);
                return false;
            }
            return true;
        };

        for (const auto &input_node : ngraph::as_node_vector(input_values)) {
            if (auto subgraph = ov::as_type_ptr<op::Subgraph>(input_node)) {
                if (!clones.count(input_node)) {
                    auto f = ngraph::clone_function(*subgraph->get_body().get());
                    f->set_friendly_name(subgraph->get_body()->get_friendly_name());
                    clones[input_node] = f;
                    fusedNames += getFusedNames(subgraph);
                }
            }
        }
        fusedNames += node->get_friendly_name();

        std::pair<int64_t, int64_t> currentTopoBounds {-1, LONG_MAX};
        cyclicDependencyIsIntoduced(node, currentTopoBounds);
        assert(!cyclicDependencyIsIntoduced(node, currentTopoBounds) && "Cyclic dependency is introduced by the node itself");
        for (const auto& input_value : input_values) {
            auto input_node = input_value.get_node_shared_ptr();
            if (ov::is_type<op::Subgraph>(input_node) &&
                !cyclicDependencyIsIntoduced(input_node, currentTopoBounds)) {
                auto subgraph = std::static_pointer_cast<op::Subgraph>(input_node);
                if (!input_subgraphs.count(input_node)) {
                    input_subgraphs.insert(input_node);

                    auto f = clones[input_node];
                    const auto& input_body_parameters = f->get_parameters();
                    // Todo:
                    // Some of the imput subgraphs might have common parents, so some of the input_parameters might already be
                    // in external_inputs and hence in body_parameters. Here we handle this case and remove repeated body_parameters.
                    // Would it be better to incorporate all inputs first and then remove repeated params.
                    for (size_t i = 0; i < input_body_parameters.size(); ++i) {
                        auto found = std::find(external_inputs.begin(), external_inputs.end(), subgraph->input_value(i));
                        if (found != external_inputs.end()) {
                            // Todo: here we rely on friendly_name uniqueness. Propose a different algorithm.
                            size_t current_input_index = body_parameters.size();
                            for (size_t p_ind = 0; p_ind <  body_parameters.size(); p_ind++) {
                                const auto & p = body_parameters[p_ind];
                                if (p->get_friendly_name() == found->get_node_shared_ptr()->get_friendly_name()) {
                                    current_input_index = p_ind;
                                    break;
                                }
                            }

                            if (current_input_index < body_parameters.size()) {
                                remark(13) << "replacing " << *found << " " << current_input_index << " with "
                                          << body_parameters[current_input_index] << std::endl;
                                f->replace_parameter(i, body_parameters[current_input_index]);
                            } else {
                                external_inputs.push_back(subgraph->input_value(i));
                                body_parameters.push_back(input_body_parameters[i]);
                            }
                        } else if (is_recurrent(subgraph->input_value(i))) {
                            remark(13) << "ternary merge is conducted " << subgraph->input_value(i).get_node_shared_ptr() << std::endl;

                            auto internal = input_body_parameters[i];
                            auto internal_consumers = internal->outputs();

                            if (auto to_replace_with = ov::as_type_ptr<op::Subgraph>(subgraph->get_input_node_shared_ptr(i))) {
                                 for (auto output : internal_consumers) {
                                     for (auto consumer : output.get_target_inputs()) {
                                         auto other_body = clones[subgraph->get_input_node_shared_ptr(i)];
                                         auto other_body_result = other_body->get_results()[consumer.get_source_output().get_index()];
                                         auto result_producer = other_body_result->input(0).get_source_output();

                                         consumer.replace_source_output(result_producer.get_node_shared_ptr());
                                     }
                                 }
                            } else {
                                external_inputs.push_back(subgraph->input_value(i));
                                body_parameters.push_back(input_body_parameters[i]);
                            }
                        } else {
                            external_inputs.push_back(subgraph->input_value(i));
                            body_parameters.push_back(input_body_parameters[i]);
                        }
                    }
                }

                // this is there stitching happens, get result of a copy of a body of currently processed input and put it to the new inputs
                // internal output index == external output index
                auto& input_body = clones[input_node];
                size_t source_output_index = input_value.get_index();
                auto source_result = input_body->get_results()[source_output_index];
                // Result op has a single input
                internal_inputs.push_back(source_result->input_value(0));
            } else {
                if (op::is_scalar_constant(input_node)) {
                    internal_inputs.push_back(input_node->output(0));
                } else {
                    external_inputs.push_back(input_value);
                    auto new_parameter = std::make_shared<opset1::Parameter>(input_value.get_element_type(), input_value.get_partial_shape());
                    new_parameter->set_friendly_name(input_node->get_friendly_name());
                    body_parameters.push_back(new_parameter);
                    internal_inputs.push_back(new_parameter->output(0));
                }
            }
        }
        size_t num_result_children = get_num_result_children(node);
        std::string newSubgraphName;
        if (num_result_children == 0) {
            for (const auto& n : as_node_vector(input_values))
                if (ov::is_type<op::Subgraph>(n)) {
                    newSubgraphName = n->get_friendly_name();
                    break;
                }
        } else {
            newSubgraphName = node->get_friendly_name();
        }
        for (const auto& subgraph : input_subgraphs) {
            if (has_result_child(subgraph)) {
                num_result_children++;
                newSubgraphName = subgraph->get_friendly_name();
            }
        }
        if (num_result_children > 1)
            return abort_with_strategy("New subgraph is created since too many Result children are detected");

        auto body_node = node->copy_with_new_inputs(internal_inputs);
        body_node->set_friendly_name(node->get_friendly_name());

        remark(1) << "Original node outputs = " << node->get_output_size()
                    << " body node outputs = " << body_node->get_output_size() << std::endl;

        if (node->get_output_size() != body_node->get_output_size()) {
            throw ngraph_error("original node outputs size and extracted node outputs size doesn't much");
        }

        ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;

        for (auto subgraph : input_subgraphs) {
            for (auto output : subgraph->outputs()) {
                bool first_side_consumer = true;

                for (auto target_input : output.get_target_inputs()) {
                    auto target_node = target_input.get_node()->shared_from_this();

                    if (input_subgraphs.count(target_node)) {
                        remark(13) << "ternary merge is conducted " << subgraph << " -> " << target_node << std::endl;
                    }

                    if (!input_subgraphs.count(target_node) && target_node != node) {
                        if (first_side_consumer) {
                            auto& input_subgraph_body = clones[subgraph];
                            body_results.push_back(std::make_shared<opset1::Result>(input_subgraph_body->get_results()[output.get_index()]->input_value(0)));
                            subgraph_result_inputs.push_back({});

                            first_side_consumer = false;
                        }

                        if (!!subgraph_result_inputs.back().count(target_input)) {
                            throw ngraph_error("target input added twice!!!");
                        }
                        // save target input port outside the body
                        subgraph_result_inputs.back().insert(target_input);
                    }
                }
            }
        }

        for (auto output : node->outputs()) {
            body_results.push_back(std::make_shared<opset1::Result>(body_node->output(output.get_index())));
            subgraph_result_inputs.push_back(output.get_target_inputs());
        }

        if (body_results.size() != subgraph_result_inputs.size()) {
            throw ngraph_error("body results and node results size mismatch during subgraph collaps");
        }

        if (body_parameters.size() + body_results.size() > 7) {
            const std::string message_reset = "new subgraph is created. Impossible to schedule subgraph with " +
            std::to_string(body_parameters.size()) + " inputs and " + std::to_string(body_results.size()) + " outputs.";
            const std::string message_abort = "failed to continue subgraph. Impossible to schedule subgraph with " +
            std::to_string(body_parameters.size()) + " inputs and " + std::to_string(body_results.size()) + " outputs.";
            return abort_with_strategy(message_reset, message_abort);
        }

        auto body = op::create_body(newSubgraphName, body_results, body_parameters);
        for (size_t i = 0; i < body->get_parameters().size(); i++) {
            body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }
        auto subgraph = op::build_subgraph(node, external_inputs, body, newSubgraphName);
        auto act_body = subgraph->get_body();
        for (size_t i = 0; i < act_body->get_parameters().size(); i++) {
            act_body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }

        if (subgraph->get_output_size() != subgraph_result_inputs.size()) {
            throw ngraph_error("newly create subgraph doesn't much number of results");
        }

        if (outputs_are_not_broadcastable(subgraph))
            return abort_with_strategy("New subgraph is created due to outputs of a subgraph not broadcastable.");

        for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
            for (auto target_input : subgraph_result_inputs[i]) {
                target_input.replace_source_output(subgraph->output(i));
            }
        }
        if (num_result_children == 1)
            update_out_tensor_name(subgraph);

        subgraph->validate_and_infer_types();

        auto act_body1 = subgraph->get_body();
        for (size_t i = 0; i < act_body1->get_parameters().size(); i++) {
            act_body1->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }
        subgraph->get_rt_info()["originalLayersNames"] = ov::make_variant(fusedNames);

        remark(1) << "Replacement (merge) done for: "
                    << subgraph->get_friendly_name()
                    << " with " << subgraph->inputs().size()
                    << " inputs and " << subgraph->outputs().size()
                    << " outputs and " << subgraph->get_body()->get_ops().size() << " ops total\n";

        return true;
    };
    auto matcher = std::make_shared<ngraph::pattern::Matcher>(label);
    register_matcher(matcher, callback);
}
