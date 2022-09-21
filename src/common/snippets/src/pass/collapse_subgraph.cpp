// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/remarks.hpp"
#include <snippets/itt.hpp>

#include "snippets/pass/collapse_subgraph.hpp"
#include "snippets/op/subgraph.hpp"
#include "snippets/utils.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/op/loop.hpp>
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>
#include <cassert>
#include <queue>
#include <string>
#include <numeric>
#include <climits>


namespace ngraph {
namespace snippets {
namespace pass {


namespace {

auto outputs_are_not_broadcastable(const std::shared_ptr<const Node>& node) -> bool {
    auto outputs = node->outputs();
    auto find_smallest_output_shape = [](const std::vector<Output<const Node>>& outputs) -> Shape {
        return std::accumulate(std::begin(outputs), std::end(outputs), ngraph::Shape(outputs.begin()->get_shape()),
            [](Shape& other_shape, const Output<const Node>& output){
                return shape_size(output.get_shape()) < shape_size(other_shape) ? output.get_shape() : other_shape;
            });
    };
    auto ref_shape = find_smallest_output_shape(outputs);

    auto check_shapes_broadcastable = [ref_shape](const Output<const Node>& output) -> bool {
        auto other_shape = output.get_shape();

        if (other_shape.size() != ref_shape.size()) {
            return false;
        }

        return std::inner_product(std::begin(other_shape), std::end(other_shape), std::begin(ref_shape), true,
                            std::logical_and<bool>(), [](Shape::value_type lsh, Shape::value_type rsh){
                                return rsh == 1 || lsh == rsh;
                            });
    };

    return std::find_if_not(std::begin(outputs), std::end(outputs), check_shapes_broadcastable) != std::end(outputs);
}

auto is_supported_op(const std::shared_ptr<const Node> &n) -> bool {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::is_supported_op")
    auto is_supported_fq_op = [](const std::shared_ptr<const Node>& n) -> bool {
        // TODO [92179]: Add support of FakeQuantize with non-constants inputs and with binarization algorithm.
        const auto fq = ov::as_type_ptr<const opset1::FakeQuantize>(n);
        return fq && fq->get_levels() != 2 &&
               is_type<opset1::Constant>(n->get_input_node_shared_ptr(1)) &&
               is_type<opset1::Constant>(n->get_input_node_shared_ptr(2)) &&
               is_type<opset1::Constant>(n->get_input_node_shared_ptr(3)) &&
               is_type<opset1::Constant>(n->get_input_node_shared_ptr(4));
    };

    auto is_supported_binary_eltwise_op = [](const std::shared_ptr<const Node> &n) -> bool {
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
            || ov::is_type<opset1::Xor>(n)
            || ov::is_type<ngraph::op::v0::Convert>(n);
    };

    auto is_supported_unary_eltwise_op = [](const std::shared_ptr<const Node> &n) -> bool {
        return ov::is_type<opset1::Abs>(n)
            || ov::is_type<opset1::Clamp>(n)
            || ov::is_type<opset1::Floor>(n)
            || ov::is_type<opset1::Ceiling>(n)
            || ov::is_type<opset1::Elu>(n)
            || ov::is_type<opset1::Erf>(n)
            || ov::is_type<opset1::Exp>(n)
            || ov::is_type<opset1::LogicalNot>(n)
            || ov::is_type<opset1::Negative>(n)
            || ov::is_type<opset1::Relu>(n)
            || ov::is_type<opset5::Round>(n)
            || ov::is_type<opset1::Sigmoid>(n)
            || ov::is_type<opset1::Sqrt>(n)
            || ov::is_type<opset1::Tanh>(n)
            || ov::is_type<ngraph::op::v0::Gelu>(n)
            || ov::is_type<ngraph::op::v7::Gelu>(n)
            || ov::is_type<ngraph::op::v4::Swish>(n)
            || ov::is_type<ngraph::op::v4::HSwish>(n);
    };
    return is_supported_fq_op(n) || is_supported_unary_eltwise_op(n) || is_supported_binary_eltwise_op(n);
}

auto has_supported_in_out(const std::shared_ptr<const Node> &n) -> bool {
    auto supported = [](descriptor::Tensor& t) -> bool {
        static const std::set<ngraph::element::Type> supported_data_types =
                { ngraph::element::f32, ngraph::element::i32, ngraph::element::bf16, ngraph::element::i8, ngraph::element::u8 };
        return t.get_partial_shape().is_static() && supported_data_types.count(t.get_element_type()) != 0;
    };
    const auto & inputs = n->inputs();
    const auto & outputs = n->outputs();
    // todo: Is this check necessary? Remove if not
    for (const auto& out : outputs) {
        for (const auto &in_out : out.get_target_inputs()) {
            if (ov::is_type<ngraph::op::v5::Loop>(in_out.get_node()->shared_from_this())) {
                return false;
            }
        }
    }
    return std::all_of(inputs.begin(), inputs.end(), [&](const Input<const Node>& in) {return  supported(in.get_tensor());}) &&
           std::all_of(outputs.begin(), outputs.end(), [&](const Output<const Node>& out) {return  supported(out.get_tensor());});
}

auto has_result_child(const std::shared_ptr<const Node> &node) -> bool {
    for (const auto &child : node->get_users()) {
        if (ov::is_type<ngraph::opset1::Result>(child)) {
            return true;
        }
    }
    return false;
}

auto get_num_result_children(const std::shared_ptr<const Node> &node) -> size_t {
    size_t result = 0;
    for (const auto &child : node->get_users()) {
        if (ov::is_type<ngraph::opset1::Result>(child)) {
            result++;
        }
    }
    return result;
}
// Need to update tensor name manually, since intel_cpu::Graph::Replicate() looks at input.get_tensor().get_name();
// If subgraph->get_output_size() == 1, then the name will be restored correctly from the node name
auto update_out_tensor_name(std::shared_ptr<ngraph::snippets::op::Subgraph> &subgraph) -> void {
    bool not_set = true;
    for (unsigned int i = 0; i < subgraph->get_output_size() && not_set; i++) {
        for (const auto &in : subgraph->get_output_target_inputs(i)) {
            if (ov::is_type<opset1::Result>(in.get_node())) {
                const auto& body_result = subgraph->get_body()->get_output_op(i);
                const auto& body_result_input = body_result->get_input_source_output(0);
                op::Subgraph::fill_empty_output_names(subgraph->output(i), body_result_input);
                not_set = false;
                break;
            }
        }
    }
}
} // namespace

bool AppropriateForSubgraph(const std::shared_ptr<const Node> &node) {
    return is_supported_op(node) && has_supported_in_out(node);
}

void SetSnippetsNodeType(const std::shared_ptr<Node> &node, SnippetsNodeType nodeType) {
    auto &rt = node->get_rt_info();
    rt["SnippetsNodeType"] = nodeType;
}

SnippetsNodeType GetSnippetsNodeType(const std::shared_ptr<const Node> &node) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::GetSnippetsNodeType")
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("SnippetsNodeType");
    if (rinfo == rt.end())
        return SnippetsNodeType::NotSet;
    return rinfo->second.as<SnippetsNodeType>();
}

void SetTopologicalOrder(const std::shared_ptr<Node> &node, int64_t order) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::SetTopologicalOrder")
    auto &rt = node->get_rt_info();
    rt["TopologicalOrder"] = order;
}

int64_t GetTopologicalOrder(const std::shared_ptr<const Node> &node) {
    auto &rt = node->get_rt_info();
    const auto rinfo = rt.find("TopologicalOrder");
    if (rinfo == rt.end())
        throw ngraph_error("Topological order is required, but not set.");
    return rinfo->second.as<int64_t>();
}

bool EnumerateNodes::run_on_model(const std::shared_ptr<ov::Model> &m) {
    OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::EnumerateNodes")
    int64_t order = 0;
    // Todo: We don't really have to set order for every node, just for subgraph parents and children would be enough
    for (auto &node : m->get_ordered_ops()) {
        SetTopologicalOrder(node, order++);
    }
    return true;
}
TokenizeSnippets::TokenizeSnippets() {
    MATCHER_SCOPE(TokenizeSnippets);
    enum continuation_strategy {
        reset,
        abort
    };

    continuation_strategy strategy = continuation_strategy::reset;
    auto label = std::make_shared<pattern::op::Label>(pattern::any_input(),
        [](const std::shared_ptr<const Node> &n) {
            return GetSnippetsNodeType(n) != SnippetsNodeType::SkippedByPlugin && AppropriateForSubgraph(n);
        });
    ngraph::graph_rewrite_callback callback = [&, strategy](ngraph::pattern::Matcher &m) -> bool {
        OV_ITT_SCOPED_TASK(ngraph::pass::itt::domains::SnippetsTransform, "Snippets::CreateSubgraph_callback")
        auto node = m.get_match_root();
        if (transformation_callback(node)) {
            return false;
        }

        remark(1) << "Match root: " << node->get_friendly_name() << " " << node << std::endl;

        const auto getFusedNames = [](const std::shared_ptr<Node>& n) -> std::string {
            auto rt_info = n->get_rt_info();
            auto it = rt_info.find("originalLayersNames");
            if (it != rt_info.end()) {
                return it->second.as<std::string>() + ",";
            }
            return "";
        };

        auto create_single_node_subgraph = [&](const std::shared_ptr<Node> &node) {
            auto subgraph = op::Subgraph::wrap_node_as_subgraph(node);
            subgraph->get_rt_info()["originalLayersNames"] = getFusedNames(node) + node->get_friendly_name();
            ngraph::replace_node(node, subgraph);
            update_out_tensor_name(subgraph);
        };

        auto abort_with_strategy = [&](const std::string& message_reset,
                                                     const std::string& message_abort = "", int priority = 3) {
            if (strategy == continuation_strategy::reset) {
                create_single_node_subgraph(node);
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
        std::map<std::shared_ptr<Node>, std::shared_ptr<ov::Model>> clones;

        ParameterVector body_parameters;
        // inputs to merged subgraph
        OutputVector external_inputs;
        // inputs to the node before merge to subgraph
        OutputVector internal_inputs;

        auto input_values = node->input_values();
        /*
        * Called with subgraph->input_value(i) arg and used to
        * Check that the attached node input subgraph has the same input as the node itself.
        * If true, then ternary merge is initiated.
        *        input
        *        /   \
        *  subgraph--node
        */
        auto is_recurrent = [&input_values](const ngraph::Output<ngraph::Node>& to_find) -> bool {
            return std::any_of(input_values.begin(), input_values.end(),
                        [&](const ov::Output<ov::Node> &in) {return in == to_find;});
        };
        /*
         * Checks if the passed node introduces loop dependency for given topological bounds (pair of maxParentOrder, minChildOrder).
         * The bounds are presumed to be without dependency. The bounds are updated if no dependency is introduced by the node.
        */
        const auto cyclicDependencyIsIntoduced = [&node](const std::shared_ptr<Node>& nodeToExamine, std::pair<int64_t, int64_t>& currentBounds) -> bool {
            assert(currentBounds.first < currentBounds.second && "Invalid currentBounds passed");
            const auto &parentNodes = ngraph::as_node_vector(nodeToExamine->input_values());
            const int64_t maxParentOrder = std::accumulate(parentNodes.begin(), parentNodes.end(), currentBounds.first,
                                                            [](int64_t maxOrder, std::shared_ptr<Node> n){
                                                                if (ngraph::op::is_constant(n) || ngraph::op::is_parameter(n))
                                                                    return maxOrder;
                                                                return std::max(maxOrder, GetTopologicalOrder(n));
                                                            });
            const auto &childNodes = nodeToExamine->get_users();
            // Skip the node being attached, since it will be a part of subgraph and can't introduce loop dependency
            const int64_t minChildOrder = std::accumulate(childNodes.begin(), childNodes.end(), currentBounds.second,
                                                            [&node](int64_t minOrder, std::shared_ptr<Node> n){
                                                                if (ngraph::op::is_constant(n) || ngraph::op::is_parameter(n) || n == node)
                                                                    return minOrder;
                                                                return std::min(minOrder, GetTopologicalOrder(n));
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
                    auto f = ov::clone_model(*subgraph->get_body().get());
                    f->set_friendly_name(subgraph->get_body()->get_friendly_name());
                    clones[input_node] = f;
                }
            }
        }
        //  If there are no input subgraphs no need to go further, just create a new one.
        if (clones.empty()) {
            create_single_node_subgraph(node);
            remark(1) << "Starting subgraph at: "  << node->get_friendly_name()
                      << " with " << node->inputs().size() << " inputs and " << node->outputs().size()
                      << " outputs" << std::endl;
            return true;
        }
        std::string fusedNames{};
        size_t num_result_children = 0;
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

                    fusedNames += getFusedNames(subgraph);

                    num_result_children += has_result_child(subgraph);
                    auto f = clones[input_node];
                    const auto& input_body_parameters = f->get_parameters();
                    // Todo:
                    //  Some of the input subgraphs might have common parents, so some of the input_parameters might already be
                    //  in external_inputs and hence in body_parameters. Here we handle this case and remove repeated body_parameters.
                    //  Would it be better to incorporate all inputs first and then remove repeated params.
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
                                // todo: In principle, we can still attach the node to the subgraph if cyclic dependency is introduced during ternary merge.
                                //  Need to support.
                                if (cyclicDependencyIsIntoduced(to_replace_with, currentTopoBounds))
                                    return abort_with_strategy("Attempt to perform recurrent merge for cyclic-dependent subgraphs. Aborting.");
                                for (const auto& output : internal_consumers) {
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

                // We cannot add new node, that is not Convert, after Convert (that is start node) to avoid arithmetic problems with conversion
                // We can add any new node in Subgraph after Convert (bacause after Input)
                //              Parameter
                //                  |
                //               Convert
                //
                // We cannot add new node, that isn't Convert, in Subgraph after existing Convert
                //              Parameter
                //                Relu
                //               Convert
                //
                // But we can add new Convert in Subgraph after existing Convert
                //              Parameter
                //                Relu
                //               Convert
                //               Convert
                //
                // Thus, We can grow subgraph only if Convert is the first node of subgraph and have to abort it's the last one and we want to add not Convert
                // We have this limitation because at the moment we support only one execution precision inside body, so
                // if there is Convert with input and output data types that aren't equal to supported exec type,
                // we can get conversion math errors
                const auto output_of_subgraph = source_result->get_input_node_shared_ptr(0);
                if (!ov::is_type<ngraph::op::v0::Convert>(node) && ov::is_type<ngraph::op::v0::Convert>(output_of_subgraph)) {
                    // Also we can add new node after < Parameter -> Convert -> Convert -> Convert >
                    auto grandparent = output_of_subgraph->get_input_node_ptr(0);
                    while (ov::is_type<ngraph::op::v0::Convert>(grandparent)) {
                        grandparent = grandparent->get_input_node_ptr(0);
                    }

                    if (!ov::is_type<ngraph::op::v0::Parameter>(grandparent)) {
                        return abort_with_strategy("Convert supports only as Input and as Result of subgraph. Aborting");
                    }
                }
                // Result op has a single input
                internal_inputs.push_back(source_result->input_value(0));
            } else {
                // We have to save explicitly FQ Constants to call ConstantFolding after Tokenization.
                // After ConstantFolding we will move remaining non-scalar Constants from body using ConvertConstantsToParameters pass
                if ((utils::is_scalar_constant(input_node)) ||
                    (ov::is_type<ov::op::v0::Constant>(input_node) && ov::is_type<ov::op::v0::FakeQuantize>(node))) {
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
        fusedNames += node->get_friendly_name();
        num_result_children += get_num_result_children(node);
        if (num_result_children > 1)
            return abort_with_strategy("New subgraph is created since too many Result children are detected");

        auto body_node = node->copy_with_new_inputs(internal_inputs);
        body_node->set_friendly_name(node->get_friendly_name());

        remark(1) << "Original node outputs = " << node->get_output_size()
                    << " body node outputs = " << body_node->get_output_size() << std::endl;

        if (node->get_output_size() != body_node->get_output_size()) {
            throw ngraph_error("original node outputs size and extracted node outputs size doesn't much");
        }

        // After some transformations, a different number of Constants for some operations may be created
        // than the actual number of Constants during tokenization.
        // To avoid unsupported number of non-scalar Constants in the future (plugin specific limitation)
        // we should calculate potentional number of non-scalar Constants that will be moved up from body.
        size_t hidden_non_scalar_constant_count = 0;
        if (const auto fq_node = ov::as_type_ptr<ov::op::v0::FakeQuantize>(node)) {
            hidden_non_scalar_constant_count += ngraph::snippets::utils::get_non_scalar_constant_count_for_fq(fq_node);
        }

        ResultVector body_results;
        std::vector<std::set<Input<Node>>> subgraph_result_inputs;

        for (auto subgraph : input_subgraphs) {
            // we should summurize non-scalar Constants count from all input subgraphs
            // because we will collapse them with our node and we should get total count of non-scalar Constants
            hidden_non_scalar_constant_count += ov::as_type_ptr<ngraph::snippets::op::Subgraph>(subgraph)->get_non_scalar_constants_count();

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

        // todo: move this plugin-specific constraint to the plugin callback
        if (body_parameters.size() + body_results.size() + hidden_non_scalar_constant_count > 12) {
            const std::string message_reset = "new subgraph is created. Impossible to schedule subgraph with " +
            std::to_string(body_parameters.size()) + " inputs, " + std::to_string(body_results.size()) + " outputs and " +
            std::to_string(hidden_non_scalar_constant_count) + " non-scalar constants.";
            const std::string message_abort = "failed to continue subgraph. Impossible to schedule subgraph with " +
            std::to_string(body_parameters.size()) + " inputs, " + std::to_string(body_results.size()) + " outputs and " +
            std::to_string(hidden_non_scalar_constant_count) + " non-scalar constants.";
            return abort_with_strategy(message_reset, message_abort);
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

        if (outputs_are_not_broadcastable(subgraph))
            return abort_with_strategy("New subgraph is created due to outputs of a subgraph not broadcastable.");

        for (size_t i = 0; i < subgraph->get_output_size(); ++i) {
            for (auto target_input : subgraph_result_inputs[i]) {
                target_input.replace_source_output(subgraph->output(i));
            }
        }
        update_out_tensor_name(subgraph);

        subgraph->validate_and_infer_types();

        auto act_body1 = subgraph->get_body();
        for (size_t i = 0; i < act_body1->get_parameters().size(); i++) {
            act_body1->get_parameters()[i]->set_friendly_name(body_parameters[i]->get_friendly_name());
        }
        subgraph->get_rt_info()["originalLayersNames"] = fusedNames;
        subgraph->set_non_scalar_constants_count(hidden_non_scalar_constant_count);

        remark(1) << "Replacement (merge) done for: "
                    << subgraph->get_friendly_name()
                    << " with " << subgraph->inputs().size()
                    << " inputs and " << subgraph->outputs().size()
                    << " outputs and " << subgraph->get_body()->get_ops().size() << " ops total\n";

        return true;
    };
    auto matcher = std::make_shared<ngraph::pattern::Matcher>(label, matcher_name);
    register_matcher(matcher, callback);
}
} // namespace pass
} // namespace snippets
} // namespace ngraph
