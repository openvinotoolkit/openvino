// Copyright (C) 2022-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "insert_copy_layer.hpp"

#include <legacy/ngraph_ops/crop_ie.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <openvino/cc/ngraph/itt.hpp>
#include <openvino/core/except.hpp>
#include <openvino/opsets/opset9.hpp>
#include <ops/copy.hpp>

#include "common/graph_utils.hpp"

using namespace ov::intel_gna::pass;
using namespace ov::intel_gna::graph_utils;
using namespace ov::opset9;

static const std::string kNonCompProperty("non_compute_node");
static const std::string kResultProperty("result_vector");

NGRAPH_RTTI_DEFINITION(InsertCopyBeforeAssignLayer, "InsertCopyBeforeAssignLayer");
NGRAPH_RTTI_DEFINITION(InsertCopyBeforeConcatLayer, "InsertCopyBeforeConcatLayer");
NGRAPH_RTTI_DEFINITION(HandleMultiConnectedLayerToConcatAndMemory, "HandleMultiConnectedLayerToConcatAndMemory");
NGRAPH_RTTI_DEFINITION(MatchNonComputationalLayers, "MatchNonComputationalLayers");
NGRAPH_RTTI_DEFINITION(HandleNonFunctionalSubgraphs, "HandleNonFunctionalSubgraphs");

namespace {
void insert_copy_layer_between(std::shared_ptr<ngraph::Node> input_op,
                               std::shared_ptr<ngraph::Node> output_op,
                               const size_t& index) {
    NGRAPH_CHECK(input_op);
    NGRAPH_CHECK(output_op);

    auto input_op_out_index = output_op->input(index).get_source_output().get_index();
    // In this case we don't need copy layer insertion, because after insertion of aligning filter graph will include
    // convolution layer Should be removed when InsertSplitAligningFilterPass is moved to nGraph, because it should run
    // before the copy layer insertion passes
    if (!is_aligned_split(input_op, input_op_out_index))
        return;

    auto copy_op = std::make_shared<ov::intel_gna::op::Copy>(input_op->output(input_op_out_index));
    copy_op->set_friendly_name(input_op->get_friendly_name() + "/copy_layer/" + output_op->get_friendly_name() + "." +
                               std::to_string(index));
    ngraph::copy_runtime_info(input_op, copy_op);

    output_op->input(index).replace_source_output(copy_op);
}

class CopyInsertionForEliminatedLayersHandler {
public:
    virtual ~CopyInsertionForEliminatedLayersHandler() = default;
    bool InsertCopyBeforeIfNeeded(std::shared_ptr<ngraph::Node>& layer);

protected:
    virtual bool WillLayerBeEliminated(std::shared_ptr<ngraph::Node>& layer) = 0;
};

class BroadcastCopyInsertionHandler : public CopyInsertionForEliminatedLayersHandler {
protected:
    bool WillLayerBeEliminated(std::shared_ptr<ngraph::Node>& layer) override;
};

class TileCopyInsertionHandler : public CopyInsertionForEliminatedLayersHandler {
protected:
    bool WillLayerBeEliminated(std::shared_ptr<ngraph::Node>& layer) override;
};

bool CopyInsertionForEliminatedLayersHandler::InsertCopyBeforeIfNeeded(std::shared_ptr<ngraph::Node>& layer) {
    auto layer_input = layer->get_input_node_shared_ptr(0);

    // skip non functional layers
    auto current_node = get_prev_node_skipping_certain(layer_input, is_gna_non_functional_node);

    if (!std::dynamic_pointer_cast<Parameter>(current_node)) {
        return false;
    }

    if (WillLayerBeEliminated(layer)) {
        insert_copy_layer_between(layer_input, layer, 0);
        return true;
    }
    return false;
}

bool BroadcastCopyInsertionHandler::WillLayerBeEliminated(std::shared_ptr<ngraph::Node>& layer) {
    auto data_node = layer->input_value(0);

    // if input has dynamic shape
    if (data_node.get_partial_shape().is_dynamic()) {
        return false;
    }

    auto input_shape = data_node.get_shape();
    auto output_shape = layer->get_output_shape(0);

    // if input shape rank is higher than output shape rank return false;
    if (input_shape.size() > output_shape.size()) {
        return false;
    }

    // if size product is not the same
    if (shape_size(input_shape) != shape_size(output_shape)) {
        return false;
    }

    return true;
}

bool TileCopyInsertionHandler::WillLayerBeEliminated(std::shared_ptr<ngraph::Node>& layer) {
    static constexpr size_t index_of_repeats_shape = 1;

    auto repeats_node =
        std::dynamic_pointer_cast<Constant>(layer->input_value(index_of_repeats_shape).get_node_shared_ptr());

    if (!repeats_node)
        return false;

    auto repeats_vaues = repeats_node->cast_vector<int64_t>();

    for (const auto& value : repeats_vaues) {
        if (value != 1) {
            return false;
        }
    }

    return true;
}

std::shared_ptr<CopyInsertionForEliminatedLayersHandler> GetCopyInsertionHandler(std::shared_ptr<ngraph::Node>& layer) {
    if (std::dynamic_pointer_cast<Broadcast>(layer)) {
        return std::make_shared<BroadcastCopyInsertionHandler>();
    }

    if (std::dynamic_pointer_cast<Tile>(layer)) {
        return std::make_shared<TileCopyInsertionHandler>();
    }

    return nullptr;
}

}  // namespace

InsertCopyBeforeAssignLayer::InsertCopyBeforeAssignLayer() {
    MATCHER_SCOPE(InsertCopyBeforeAssignLayer);

    auto memory_op = ngraph::pattern::wrap_type<ngraph::op::ReadValueBase, ngraph::op::AssignBase>();

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = std::dynamic_pointer_cast<ngraph::Node>(m.get_match_root());

        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto matched_node_input = node->get_input_node_shared_ptr(i);
            auto current_node = get_prev_node_skipping_certain(matched_node_input, is_gna_non_functional_node);

            // Crop -> Memory, Input -> Split -> Memory, Concat -> Memory
            if ((std::dynamic_pointer_cast<ngraph::op::CropIE>(current_node) && !is_crop_affined(current_node)) ||
                is_concat(current_node) || std::dynamic_pointer_cast<ngraph::opset8::Split>(current_node) ||
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
        auto concat = m.get_match_root();

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

InsertCopyBeforeLayerToBeEliminated::InsertCopyBeforeLayerToBeEliminated() {
    MATCHER_SCOPE(InsertCopyBeforeLayerToBeEliminated);
    const auto constant = ngraph::pattern::wrap_type<Constant>();
    const auto broadcast_op = ngraph::pattern::wrap_type<Broadcast>({ngraph::pattern::any_input(), constant});

    const auto tile_op = ngraph::pattern::wrap_type<Tile>({ngraph::pattern::any_input(), constant});

    const auto brodcast_tile = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{broadcast_op, tile_op});

    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto node = m.get_match_root();

        auto insertion_handler = GetCopyInsertionHandler(node);

        if (!insertion_handler) {
            return false;
        }

        // Parameter -> Tile/Broadcast to Parameter -> Copy -> Tile/Broadcast
        // Parameter -> non functional -> Tile/Broadcast to Parameter -> non functional -> Copy -> Tile/Broadcast
        return insertion_handler->InsertCopyBeforeIfNeeded(node);
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(brodcast_tile, matcher_name);
    this->register_matcher(m, callback);
}

bool HandleMultiConnectedLayerToConcatAndMemory::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(HandleMultiConnectedLayerToConcatAndMemory);

    using FuncChildrenInfo = std::tuple<std::shared_ptr<ngraph::Node>,  // parent node
                                        std::shared_ptr<ngraph::Node>,  // child node
                                        int32_t                         // input index
                                        >;

    // recursively searches for children functional layers skipping non-functional ones
    std::function<std::vector<FuncChildrenInfo>(std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>, int32_t)>
        find_func_layers = [&find_func_layers](std::shared_ptr<ngraph::Node> current_node,
                                               std::shared_ptr<ngraph::Node> parent_node,
                                               int32_t input_idx) {
            if (!is_gna_non_functional_node(current_node) || current_node->get_output_size() == 0 ||
                current_node->output(0).get_target_inputs().size() == 0) {
                return std::vector<FuncChildrenInfo>{std::make_tuple(parent_node, current_node, input_idx)};
            }
            std::vector<FuncChildrenInfo> results;
            for (auto& child : current_node->output(0).get_target_inputs()) {
                auto next_node = std::dynamic_pointer_cast<ngraph::Node>(child.get_node()->shared_from_this());
                auto result = find_func_layers(next_node, current_node, static_cast<int32_t>(child.get_index()));
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
            if (input_to.size() < 2)
                continue;
            std::vector<FuncChildrenInfo> concat_nodes, memory_nodes;
            for (auto& child : input_to) {
                auto current_node = std::dynamic_pointer_cast<ngraph::Node>(child.get_node()->shared_from_this());
                auto children_info = find_func_layers(current_node, node, static_cast<int32_t>(child.get_index()));

                for (const auto& child_info : children_info) {
                    auto child = std::get<1>(child_info);

                    if (is_concat(child)) {
                        concat_nodes.push_back(child_info);
                    } else if (std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(child) ||
                               std::dynamic_pointer_cast<ngraph::op::AssignBase>(child)) {
                        memory_nodes.push_back(child_info);
                    }
                }
            }

            if (memory_nodes.empty() && concat_nodes.empty())
                continue;

            auto count_to_copy = memory_nodes.size() + concat_nodes.size() -
                                 (std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node) ? 0 : 1);
            // Insertion of copy to memory layers has a priority on the concat layers
            for (size_t i = 0; i < count_to_copy; i++) {
                std::shared_ptr<ngraph::Node> in_layer, out_layer;
                size_t input_id;
                std::tie(in_layer, out_layer, input_id) =
                    (i < memory_nodes.size()) ? memory_nodes[i] : concat_nodes[i - memory_nodes.size()];
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
        if (!(is_gna_non_functional_node(node) || std::dynamic_pointer_cast<ngraph::op::CropIE>(node) ||
              std::dynamic_pointer_cast<ngraph::opset8::Split>(node) ||
              std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(node) ||
              std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node) ||
              std::dynamic_pointer_cast<ngraph::opset8::Constant>(node) ||
              std::dynamic_pointer_cast<ngraph::opset8::Result>(node))) {
            return false;
        }

        // Since we traverse graph in reverse order, the result should be one of the first nodes
        auto& rt_info = node->get_rt_info();
        auto res_node = std::dynamic_pointer_cast<ngraph::opset8::Result>(node);
        if (res_node) {
            rt_info[kNonCompProperty] = true;
            // We collect the results to the vector, because it possible to have
            // two different non-computational subgraphs with different results
            rt_info[kResultProperty] = ngraph::ResultVector{res_node};
        }

        if (!rt_info.count(kNonCompProperty) || !rt_info[kNonCompProperty].as<bool>()) {
            return false;
        }

        // if current node is non-computational, pass the properties for each input
        for (size_t i = 0; i < node->get_input_size(); i++) {
            auto& input_rti = node->get_input_node_shared_ptr(i)->get_rt_info();
            input_rti[kNonCompProperty] = true;
            if (input_rti.count(kResultProperty) && !input_rti[kResultProperty].as<ngraph::ResultVector>().empty()) {
                for (auto&& res : rt_info[kResultProperty].as<ngraph::ResultVector>()) {
                    input_rti[kResultProperty].as<ngraph::ResultVector>().push_back(res);
                }
            } else {
                input_rti[kResultProperty] = rt_info[kResultProperty];
            }
        }
        // Found parameter node with non-computational property, so we detected desired subgraph
        // Need to insert a copy op for each pre-result node, that runs out to this parameter
        if (std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node)) {
            auto result_vec = rt_info[kResultProperty].as<ngraph::ResultVector>();
            for (auto&& result_node : result_vec) {
                auto copy_out = result_node->get_input_node_shared_ptr(0);
                for (size_t i = 0; i < copy_out->get_input_size(); i++) {
                    auto copy_in = copy_out->get_input_node_shared_ptr(i);
                    if (!std::dynamic_pointer_cast<ngraph::opset8::Constant>(copy_in) &&
                        // Copy already inserted from different result
                        !std::dynamic_pointer_cast<ov::intel_gna::op::Copy>(copy_in)) {
                        insert_copy_layer_between(copy_in, copy_out, i);
                    }
                }
            }
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(noncompute_op, matcher_name);
    this->register_matcher(m, callback);
}

bool HandleNonFunctionalSubgraphsCleanup::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(HandleNonFunctionalSubgraphsCleanup);

    std::vector<std::string> properties{kNonCompProperty, kResultProperty};

    for (const auto& node : m->get_ops()) {
        auto& rt_info = node->get_rt_info();
        for (const auto& property : properties) {
            rt_info.erase(property);
        }
    }

    return false;
}
