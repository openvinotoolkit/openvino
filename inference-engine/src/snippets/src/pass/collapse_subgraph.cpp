// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "remarks.hpp"
#include "itt.hpp"

#include "snippets/pass/collapse_subgraph.hpp"
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

NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::StartSubgraph, "CollapseSubgraph", 0);
NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::AttachToSubgraph, "CollapseSubgraph", 0);
NGRAPH_RTTI_DEFINITION(ngraph::snippets::pass::TokenizeSnippets, "CollapseSubgraph", 0);

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

auto has_cycles_of_dependencies(const std::vector<std::set<ngraph::Input<ngraph::Node>>>& results,
                                const std::vector<ngraph::Input<ngraph::Node>>& inputs) -> bool {
    auto BFS_from_to = [](ngraph::Node* from, ngraph::Node* to) -> bool {
        std::unordered_set<ngraph::Node*> visited;
        std::queue<ngraph::Node*> stack;
        stack.push(from);

        while (stack.size() > 0) {
            ngraph::Node* curr = stack.front();
            visited.insert(curr);

            if (ngraph::op::is_output(curr)) {
                return false;
            }

            stack.pop();

            if (curr != to) {
                for (const auto& next : curr->get_users()) {
                    if (visited.count(next.get()) == 0) {
                        stack.push(next.get());
                    }
                }
            } else {
                return true;
            }
        }
        return false;
    };

    for (auto& result : results) {
        for (auto& user : result) {
            for (auto& input : inputs) {
                auto source = input.get_source_output().get_node();
                auto containsLoop = BFS_from_to(user.get_node(), source);

                remark(1) <<  "checking path from "
                        << user.get_node()->get_friendly_name()
                        << " to " << source->get_friendly_name()
                        << " resulted in " << containsLoop << std::endl;

                if (containsLoop) {
                    return true;
                }
            }
        }
    }
    return false;
}

auto has_subgraph_as_input(std::shared_ptr<Node> node) -> bool {
    auto inputs = node->inputs();
    for (auto input : inputs) {
        auto parent = input.get_source_output().get_node_shared_ptr();
        if (!!as_type_ptr<snippets::op::Subgraph>(parent)) {
            return true;
        }
    }
    return false;
};

auto is_lo(std::shared_ptr<Node> n) -> bool {
    auto is_lob = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return !!as_type_ptr<opset1::Add>(n)
            || !!as_type_ptr<opset1::Divide>(n)
            || !!as_type_ptr<opset1::Equal>(n)
            || !!as_type_ptr<opset1::FloorMod>(n)
            || !!as_type_ptr<opset1::Greater>(n)
            || !!as_type_ptr<opset1::GreaterEqual>(n)
            || !!as_type_ptr<opset1::Less>(n)
            || !!as_type_ptr<opset1::LessEqual>(n)
            || !!as_type_ptr<opset1::LogicalAnd>(n)
            || !!as_type_ptr<opset1::LogicalOr>(n)
            || !!as_type_ptr<opset1::LogicalXor>(n)
            || !!as_type_ptr<opset1::Maximum>(n)
            || !!as_type_ptr<opset1::Minimum>(n)
            || !!as_type_ptr<opset1::Mod>(n)
            || !!as_type_ptr<opset1::Multiply>(n)
            || !!as_type_ptr<opset1::NotEqual>(n)
            || !!as_type_ptr<opset1::PRelu>(n)
            || !!as_type_ptr<opset1::Power>(n)
            || !!as_type_ptr<opset1::SquaredDifference>(n)
            || !!as_type_ptr<opset1::Subtract>(n)
            || !!as_type_ptr<opset1::Xor>(n);
    };

    auto is_lou = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return !!as_type_ptr<opset1::Abs>(n)
            // || !!as_type_ptr<opset1::Acos>(n)
            // || !!as_type_ptr<opset1::Asin>(n)
            // || !!as_type_ptr<opset1::Atan>(n)
            // || !!as_type_ptr<opset1::Ceiling>(n) ?
            || !!as_type_ptr<opset1::Clamp>(n)
            // || !!as_type_ptr<opset1::Cos>(n)
            // || !!as_type_ptr<opset1::Cosh>(n)
            || !!as_type_ptr<opset1::Elu>(n)
            || !!as_type_ptr<opset1::Erf>(n)
            || !!as_type_ptr<opset1::Exp>(n)
            // || !!as_type_ptr<opset1::Floor>(n) ?
            // || !!as_type_ptr<opset1::Log>(n) ?
            || !!as_type_ptr<opset1::LogicalNot>(n)
            || !!as_type_ptr<opset1::Negative>(n)
            || !!as_type_ptr<opset1::Relu>(n)
            // || !!as_type_ptr<opset1::Sign>(n) ?
            || !!as_type_ptr<opset1::Sigmoid>(n)
            // || !!as_type_ptr<opset1::Sin>(n)
            // || !!as_type_ptr<opset1::Sinh>(n)
            || !!as_type_ptr<opset1::Sqrt>(n)
            // || !!as_type_ptr<opset1::Tan>(n)
            || !!as_type_ptr<opset1::Tanh>(n);
    };

    auto is_lot = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return false;
        // return !!as_type_ptr<opset1::HardSigmoid>(n) // ternary with 2 constants
            // || !!as_type_ptr<opset1::Selu>(n); // ternary with 2 constants / or DW
    };

    auto is_fq = [](std::shared_ptr<Node> n) -> bool {
        using ngraph::as_type_ptr;
        return false;//!!as_type_ptr<opset1::FakeQuantize>(n); // 4->1
    };

    return is_lou(n) || is_lob(n) ||is_lot(n) || is_fq(n);
}

auto has_supported_in_out(std::shared_ptr<Node> n) -> bool {
    for (auto in : n->inputs()) {
        if (in.get_tensor().get_element_type() != ngraph::element::f32) {
            return false;
        }
    }

    for (auto out : n->outputs()) {
        if (out.get_tensor().get_element_type() != ngraph::element::f32) {
            return false;
        }

        for (auto in_out : out.get_target_inputs()) {
            if (!!as_type_ptr<ngraph::op::v5::Loop>(in_out.get_node()->shared_from_this())) {
                return false;
            }
        }
    }

    return true;
};

} // namespace

ngraph::snippets::pass::StartSubgraph::StartSubgraph(bool tokenize_by_node) : MatcherPass() {
    MATCHER_SCOPE(StartSubgraph);

    auto has_multiple_output_edges = [](std::shared_ptr<Node> n) -> bool {
        for (auto out : n->outputs()) {
            if (out.get_target_inputs().size() != 1) return true;
        }

        return false;
    };

    register_matcher(std::make_shared<pattern::Matcher>(
        std::make_shared<pattern::op::Label>(pattern::any_input(),
        [tokenize_by_node, has_multiple_output_edges](std::shared_ptr<Node> n) {
            return is_lo(n) &&
                   has_supported_in_out(n) &&
                   (tokenize_by_node || !has_subgraph_as_input(n)) &&
                   has_multiple_output_edges(n);
        })),
        [](ngraph::pattern::Matcher &m) -> bool {
        auto node = m.get_match_root();

        remark(1) << "Match root"
                  << node->get_friendly_name()
                  << " " << node
                  << " Creating new snippet - no input subgraphs found" << std::endl;

        auto subgraph = op::Subgraph::wrap_node_as_subgraph(node);
        ngraph::replace_node(node, subgraph);

        remark(1) << "Replacement (new) done for: "
                  << subgraph->get_friendly_name()
                  << " with " << subgraph->inputs().size()
                  << " inputs and " << subgraph->outputs().size()
                  << " outputs and " << subgraph->get_body()->get_ops().size() << " ops total\n";
        return true;
    });
}

ngraph::snippets::pass::AttachToSubgraph::AttachToSubgraph(bool tokenize_by_node) : MatcherPass() {
    MATCHER_SCOPE(AttachToSubgraph);
    enum continuation_strategy {
        reset,
        abort
    };

    continuation_strategy strategy = continuation_strategy::abort;

    ngraph::graph_rewrite_callback continuation_callback = [strategy](ngraph::pattern::Matcher &m) -> bool {
        auto node = m.get_match_root();

        remark(1) << "Match root " << node->get_friendly_name() << " " << node << std::endl;

        // inputs that are already subgraphs
        std::unordered_set<std::shared_ptr<Node>> input_subgraphs;
        // clone bodies because we need a rollback if loop is found
        std::map<std::shared_ptr<Node>, std::shared_ptr<ngraph::Function>> clones;

        ParameterVector body_parameters;
        OutputVector external_inputs;
        OutputVector internal_inputs;

        auto inputs = node->inputs();

        auto is_recurrent = [inputs](const ngraph::Output<ngraph::Node>& to_find) -> bool {
            for (auto in : inputs) {
                if (in.get_source_output().get_node_shared_ptr() == to_find.get_node_shared_ptr()) {
                    return true;
                }
            }
            return false;
        };

        auto get_input_index = [](const Output<Node>& found) -> size_t {
            for (auto& input : found.get_target_inputs()) {
                remark(13) << input.get_node() << " " << input.get_source_output() << " vs "
                    << found << found.get_node() << " : " << input.get_index() << " " << found.get_index() << std::endl;
            }

            for (auto& input : found.get_target_inputs()) {
                remark(13) << input.get_node() << " " << input.get_source_output() << " vs "
                    << found << " : " << input.get_index() << " " << found.get_index() << std::endl;
                if (as_type_ptr<op::Subgraph>(input.get_node()->shared_from_this()) != nullptr && input.get_source_output() == found) {
                    return input.get_index();
                }
            }
            return 0;
        };

        for (auto input : inputs) {
            auto input_node = input.get_source_output().get_node_shared_ptr();

            if (auto subgraph = as_type_ptr<op::Subgraph>(input_node)) {
                if (!clones.count(input_node)) {
                    auto f = ngraph::clone_function(*subgraph->get_body().get());
                    f->set_friendly_name(subgraph->get_body()->get_friendly_name());
                    clones[input_node] = f;
                }
            }
        }

        for (auto input : inputs) {
            auto input_node = input.get_source_output().get_node_shared_ptr();

            if (auto subgraph = as_type_ptr<op::Subgraph>(input_node)) {
                if (!input_subgraphs.count(input_node)) {
                    input_subgraphs.insert(input_node);

                    auto f = clones[input_node];
                    const auto& input_body_parameters = f->get_parameters();

                    for (size_t i = 0; i < input_body_parameters.size(); ++i) {
                        auto found = std::find(external_inputs.begin(), external_inputs.end(), subgraph->input_value(i));
                        if (found != external_inputs.end()) {
                            auto current_input_index = get_input_index(*found);
                            // Handling the case if multiple inputs referencing the same parameter comes from one subgraph => it's not introduced by SS.
                            // It might be better to keep track if body parameter relationship rather than that
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

                            for (auto output : internal->outputs()) {
                                for (auto consumer : output.get_target_inputs()) {
                                    if (auto to_replace_with = as_type_ptr<op::Subgraph>(subgraph->input_value(i).get_node_shared_ptr())) {
                                        auto other_body = clones[subgraph->input_value(i).get_node_shared_ptr()];
                                        auto other_body_result = other_body->get_results()[consumer.get_source_output().get_index()];
                                        auto result_producer = other_body_result->input(0).get_source_output();

                                        consumer.replace_source_output(result_producer.get_node_shared_ptr());
                                    }
                                }
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
                size_t source_output_index = input.get_source_output().get_index();
                auto source_result = input_body->get_results()[source_output_index];
                // Result op has a single input
                internal_inputs.push_back(source_result->input_value(0));
            } else {
                if (op::is_scalar_constant(input_node)) {
                    internal_inputs.push_back(input_node->output(0));
                } else {
                    external_inputs.push_back(input.get_source_output());
                    auto new_parameter = std::make_shared<opset1::Parameter>(input.get_element_type(), input.get_partial_shape());
                    new_parameter->set_friendly_name(input.get_source_output().get_node()->get_friendly_name());
                    body_parameters.push_back(new_parameter);
                    body_parameters.back()->set_friendly_name(input.get_source_output().get_node()->get_friendly_name());
                    internal_inputs.push_back(new_parameter->output(0));
                }
            }
        }

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
            if (strategy == continuation_strategy::reset) {
                remark(13) << "new subgraph is created. Impossible to schedule subgraph with "
                        << body_parameters.size() << " inputs and " << body_results.size() << " outputs." << std::endl;

                auto single_node_subgraph = op::Subgraph::wrap_node_as_subgraph(node);
                ngraph::replace_node(node, single_node_subgraph);
                return true;
            } else {
                return false;
            }
        }

        auto body = op::create_body(node->get_friendly_name(), body_results, body_parameters);
        for (size_t i = 0; i < body->get_parameters().size(); i++) {
            body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }

        auto subgraph = op::build_subgraph(node, external_inputs, body);
        auto act_body = subgraph->get_body();
        for (size_t i = 0; i < act_body->get_parameters().size(); i++) {
            act_body->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }

        if (subgraph->get_output_size() != subgraph_result_inputs.size()) {
            throw ngraph_error("newly create subgraph doesn't much number of results");
        }

        if (outputs_are_not_broadcastable(subgraph)) {
            if (strategy == continuation_strategy::reset) {
                remark(13) << "New subgraph is created due to outputs of a subgraph not broadcastable." << std::endl;

                auto single_node_subgraph = op::Subgraph::wrap_node_as_subgraph(node);
                single_node_subgraph->validate_and_infer_types();
                ngraph::replace_node(node, single_node_subgraph);
                return true;
            } else {
                return false;
            }
        }

        if (has_cycles_of_dependencies(subgraph_result_inputs, subgraph->inputs())) {
            if (strategy == continuation_strategy::reset) {
                remark(13) << "New subgraph is created due to loop dependency introduced by one of input subgraphs." << std::endl;

                auto single_node_subgraph = op::Subgraph::wrap_node_as_subgraph(node);
                single_node_subgraph->validate_and_infer_types();
                ngraph::replace_node(node, single_node_subgraph);
                return true;
            } else {
                return false;
            }
        }

        for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
            for (auto target_input : subgraph_result_inputs[i]) {
                target_input.replace_source_output(subgraph->output(i));
            }
        }

        subgraph->validate_and_infer_types();

        auto act_body1 = subgraph->get_body();
        for (size_t i = 0; i < act_body1->get_parameters().size(); i++) {
            act_body1->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }

        remark(1) << "Replacement (merge) done for: "
                    << subgraph->get_friendly_name()
                    << " with " << subgraph->inputs().size()
                    << " inputs and " << subgraph->outputs().size()
                    << " outputs and " << subgraph->get_body()->get_ops().size() << " ops total\n";

        return true;
    };

    register_matcher(std::make_shared<pattern::Matcher>(
        std::make_shared<pattern::op::Label>(pattern::any_input(),
        [](std::shared_ptr<Node> n) {
            return is_lo(n) && has_supported_in_out(n) && has_subgraph_as_input(n);
        })),
        continuation_callback);
}
