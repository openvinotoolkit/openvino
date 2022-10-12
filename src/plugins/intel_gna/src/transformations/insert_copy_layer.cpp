// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/insert_copy_layer.hpp"
#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ops/copy.hpp>
#include <legacy/ngraph_ops/crop_ie.hpp>
#include <openvino/core/except.hpp>

#include "gna_plugin_log.hpp"
#include "ops/util/util.hpp"

using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::ngraph_util;

NGRAPH_RTTI_DEFINITION(InsertCopyBeforeAssignLayer, "InsertCopyBeforeAssignLayer", 0);
NGRAPH_RTTI_DEFINITION(InsertCopyBeforeConcatLayer, "InsertCopyBeforeConcatLayer", 0);
NGRAPH_RTTI_DEFINITION(HandleMultiConnectedLayerToConcatAndMemory, "HandleMultiConnectedLayerToConcatAndMemory", 0);
NGRAPH_RTTI_DEFINITION(MatchNonComputationalLayers, "MatchNonComputationalLayers", 0);
NGRAPH_RTTI_DEFINITION(HandleNonFunctionalSubgraphs, "HandleNonFunctionalSubgraphs", 0);


namespace {
    void insert_copy_layer_between(std::shared_ptr<ngraph::Node> input_op,
                                   std::shared_ptr<ngraph::Node> output_op,
                                   const size_t& index) {
        NGRAPH_CHECK(input_op);
        NGRAPH_CHECK(output_op);

        auto input_op_out_index = output_op->input(index).get_source_output().get_index();
        // In this case we don't need copy layer insertion, because after insertion of aligning filter graph will include convolution layer
        // Should be removed when InsertSplitAligningFilterPass is moved to nGraph, because it should run before the copy layer insertion passes
        if (!is_aligned_split(input_op, input_op_out_index))
            return;

        auto copy_op = std::make_shared<ov::intel_gna::op::Copy>(input_op->output(input_op_out_index));
        copy_op->set_friendly_name(input_op->get_friendly_name() + "/copy_layer/" + output_op->get_friendly_name() + "." + std::to_string(index));
        ngraph::copy_runtime_info(input_op, copy_op);

        output_op->input(index).replace_source_output(copy_op);
    }
}// namespace

InsertCopyBeforeAssignLayer::InsertCopyBeforeAssignLayer() {
    MATCHER_SCOPE(InsertCopyBeforeAssignLayer);

    auto memory_op = ngraph::pattern::wrap_type<ngraph::op::ReadValueBase,
                                                ngraph::op::AssignBase>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<ngraph::Node>(m.get_match_root());

        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto matched_node_input = node->get_input_node_shared_ptr(i);
            auto current_node = get_prev_node_skipping_certain(matched_node_input, is_gna_non_functional_node);

            // Crop -> Memory, Input -> Split -> Memory, Concat -> Memory
            if ((std::dynamic_pointer_cast<ngraph::op::CropIE>(current_node) && !is_crop_affined(current_node)) ||
                std::dynamic_pointer_cast<ngraph::opset8::Concat>(current_node) ||
                std::dynamic_pointer_cast<ngraph::opset8::Split>(current_node) ||
                std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(current_node)) {
                    insert_copy_layer_between(matched_node_input, node, i);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(memory_op, matcher_name);
    this->register_matcher(m, callback);
}

InsertCopyBeforeConcatLayer::InsertCopyBeforeConcatLayer() {
    MATCHER_SCOPE(InsertCopyBeforeConcatLayer);

    auto concat_op = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<ngraph::opset8::Concat>(m.get_match_root());

        std::set<std::shared_ptr<ngraph::Node>> inputs;
        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < concat->get_input_size(); i++) {
            auto input_op = concat->get_input_node_shared_ptr(i);

            if (inputs.find(input_op) != inputs.end()) {
                insert_copy_layer_between(input_op, concat, i);
            } else {
                inputs.insert(input_op);
            }
        }

        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < concat->get_input_size(); i++) {
            auto concat_input = concat->get_input_node_shared_ptr(i);
            auto current_node = get_prev_node_skipping_certain(concat_input, is_gna_non_functional_node);

            // Crop -> Concat, Input -> Split -> Concat
            if ((std::dynamic_pointer_cast<ngraph::op::CropIE>(current_node) && !is_crop_affined(current_node)) ||
                std::dynamic_pointer_cast<ngraph::opset8::Split>(current_node) ||
                std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(current_node)) {
                insert_copy_layer_between(concat_input, concat, i);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_op, matcher_name);
    this->register_matcher(m, callback);
}

bool HandleMultiConnectedLayerToConcatAndMemory::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(HandleMultiConnectedLayerToConcatAndMemory);

    using FuncChildrenInfo = std::tuple<
        std::shared_ptr<ngraph::Node>,   // parent node
        std::shared_ptr<ngraph::Node>,   // child node
        int32_t        // input index
    >;

    // recursively searches for children functional layers skipping non-functional ones
    std::function<std::vector<FuncChildrenInfo>(std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>, int32_t)> find_func_layers =
        [&find_func_layers](std::shared_ptr<ngraph::Node> current_node, std::shared_ptr<ngraph::Node> parent_node, int32_t input_idx) {
        if (!is_gna_non_functional_node(current_node) ||
            current_node->get_output_size() == 0 ||
            current_node->output(0).get_target_inputs().size() == 0) {
            return std::vector<FuncChildrenInfo>{std::make_tuple(parent_node, current_node, input_idx)};
        }
        std::vector<FuncChildrenInfo> results;
        for (auto& child : current_node->output(0).get_target_inputs()) {
            auto next_node = std::dynamic_pointer_cast<ngraph::Node>(child.get_node()->shared_from_this());
            auto result = find_func_layers(next_node, current_node, child.get_index());
            results.insert(results.end(), result.begin(), result.end());
        }

        return results;
    };

    bool is_graph_modified = false;
    for (auto& node : f->get_ordered_ops()) {
        if (is_gna_non_functional_node(node) || std::dynamic_pointer_cast<ngraph::opset8::Constant>(node))
            continue;
        for (auto& output : node->outputs()) {
            auto input_to = output.get_target_inputs();
            if (input_to.size() < 2) continue;
            std::vector<FuncChildrenInfo> concat_nodes, memory_nodes;
            for (auto& child : input_to) {
                auto current_node = std::dynamic_pointer_cast<ngraph::Node>(child.get_node()->shared_from_this());
                auto children_info = find_func_layers(current_node, node, child.get_index());

                for (const auto &child_info : children_info) {
                    auto child = std::get<1>(child_info);

                    if (std::dynamic_pointer_cast<ngraph::opset8::Concat>(child)) {
                        concat_nodes.push_back(child_info);
                    } else if (std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(child) ||
                        std::dynamic_pointer_cast<ngraph::op::AssignBase>(child)) {
                        memory_nodes.push_back(child_info);
                    }
                }
            }

            if (memory_nodes.empty() && concat_nodes.empty()) continue;

            auto count_to_copy = memory_nodes.size() + concat_nodes.size() - (std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node) ? 0 : 1);
            // Insertion of copy to memory layers has a priority on the concat layers
            for (size_t i = 0; i < count_to_copy; i++) {
                std::shared_ptr<ngraph::Node> in_layer, out_layer;
                size_t input_id;
                std::tie(in_layer, out_layer, input_id) = (i < memory_nodes.size()) ? memory_nodes[i] : concat_nodes[i - memory_nodes.size()];
                insert_copy_layer_between(in_layer, out_layer, input_id);
            }
            is_graph_modified = true;
        }
    }

    return is_graph_modified;
}
/* The main idea, is that we match the non-computational layers
 * one-by-one when traversing graph in reverse order.
 * For each match we check, that node contains non-computational property,
 * and then assign it for each input. We have to pass the result node too, because
 * we need to insert the copy operation before it.
 * If we found the "parameter" node with both of properties, it indicates that we found the
 * non-computational subgraph, and we insert the copy layer.
 */
MatchNonComputationalLayers::MatchNonComputationalLayers() {
    MATCHER_SCOPE(MatchNonComputationalLayers);

    auto noncompute_op = ngraph::pattern::wrap_type<ngraph::opset8::Reshape,
                                                ngraph::opset8::Squeeze,
                                                ngraph::opset8::Unsqueeze,
                                                ngraph::opset8::Transpose,
                                                ngraph::op::CropIE,
                                                ngraph::opset8::Split,
                                                ngraph::opset8::VariadicSplit,
                                                ngraph::opset8::Parameter,
                                                ngraph::opset8::Constant,
                                                ngraph::opset8::Result>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<ngraph::Node>(m.get_match_root());
        if (!(is_gna_non_functional_node(node) ||
             std::dynamic_pointer_cast<ngraph::op::CropIE>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Split>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Constant>(node) ||
             std::dynamic_pointer_cast<ngraph::opset8::Result>(node))) {
                 return false;
        }

        std::string noncomp_prop("non_compute_node");
        std::string result_prop("result_vector");

        // Since we traverse graph in reverse order, the result should be one of the first nodes
        auto& rt_info = node->get_rt_info();
        auto res_node = std::dynamic_pointer_cast<ngraph::opset8::Result>(node);
        if (res_node) {
            rt_info[noncomp_prop] = true;
            // We collect the results to the vector, because it possible to have
            // two different non-computational subgraphs with different results
            rt_info[result_prop] = ngraph::ResultVector{res_node};
        }

        if (!rt_info.count(noncomp_prop) || !rt_info[noncomp_prop].as<bool>())
            return false;

        // if current node is non-computational, pass the properties for each input
        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto& input_rti = node->get_input_node_shared_ptr(i)->get_rt_info();
            input_rti[noncomp_prop] = true;
            if (input_rti.count(result_prop) && !input_rti[result_prop].as<ngraph::ResultVector>().empty()) {
                for (auto&& res : rt_info[result_prop].as<ngraph::ResultVector>()) {
                    input_rti[result_prop].as<ngraph::ResultVector>().push_back(res);
                }
            } else {
                input_rti[result_prop] = rt_info[result_prop];
            }
        }

        // Found parameter node with non-computational property, so we detected desired subgraph
        // Need to insert a copy op for each pre-result node, that runs out to this parameter
        if (std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node)) {
            auto result_vec = rt_info[result_prop].as<ngraph::ResultVector>();
            for (auto&& result_node : result_vec) {
                auto copy_out = result_node->get_input_node_shared_ptr(0);
                for (size_t i = 0; i < copy_out->get_input_size(); i++) {
                    auto copy_in = copy_out->get_input_node_shared_ptr(i);
                    if (!std::dynamic_pointer_cast<ngraph::opset8::Constant>(copy_in) &&
                        // Copy already inserted from different result
                        !std::dynamic_pointer_cast<ov::intel_gna::op::Copy>(copy_in))
                        insert_copy_layer_between(copy_in, copy_out, i);
                }
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(noncompute_op, matcher_name);
    this->register_matcher(m, callback);
}