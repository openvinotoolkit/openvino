// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/insert_copy_layer.hpp"
#include <openvino/cc/ngraph/itt.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>
#include <ops/copy.hpp>
#include <legacy/ngraph_ops/crop_ie.hpp>
#include <ie/ie_common.h>
#include <openvino/core/except.hpp>

#include "gna_plugin_log.hpp"
#include "gna_lib_ver_selector.hpp"
#include "layers/gna_permute.hpp"

namespace GNAPluginNS {

NGRAPH_RTTI_DEFINITION(HandleMultiConnectedLayerToConcat, "HandleMultiConnectedLayerToConcat", 0);
NGRAPH_RTTI_DEFINITION(HandleLayerConnectedToConcatOrMemory, "HandleLayerConnectedToConcatOrMemory", 0);
NGRAPH_RTTI_DEFINITION(HandleNonComputationalSubgraphs, "HandleNonComputationalSubgraphs", 0);

namespace {
    void InsertCopyLayerBetween(std::shared_ptr<ngraph::Node> input_op,
                                std::shared_ptr<ngraph::Node> output_op,
                                const size_t& index) {
        NGRAPH_CHECK(input_op);
        NGRAPH_CHECK(output_op);

        auto copy_op = std::make_shared<GNAPluginNS::Copy>(input_op->output(output_op->input(index).get_source_output().get_index()));
        copy_op->set_friendly_name(input_op->get_friendly_name() + "/copy_layer/" + output_op->get_friendly_name() + "." + std::to_string(index));
        ngraph::copy_runtime_info(input_op, copy_op);

        output_op->input(index).replace_source_output(copy_op);
    }

    bool IsCropAffined(std::shared_ptr<ngraph::Node> node) {
        auto crop = std::dynamic_pointer_cast<ngraph::op::CropIE>(node);
        if (crop != nullptr && !crop->offset.empty()) {
            // currently crop layer only supports 2 bytes in int16 and int8 mode.
            // In fp32 mode this is not necessary but is useful for testing
            size_t bytesPerCropElement = 2;
            size_t cropOffset = crop->offset.back() * bytesPerCropElement;
            return (ALIGN64(cropOffset) != cropOffset);
        }
        return false;
    }

    // @brief this not only mathematically trivial, has some WA for kaldi case
    bool IsTrivialPermute(std::shared_ptr<ngraph::Node> node) {
        auto permute = std::dynamic_pointer_cast<ngraph::opset8::Transpose>(node);
        if (!permute) return false;

        auto transpose_const = std::dynamic_pointer_cast<ngraph::op::Constant>(permute->input_value(1).get_node_shared_ptr());
        if (!transpose_const) return false;

        auto node_order = transpose_const->cast_vector<int64_t>();

        if (node_order == std::vector<int64_t>({ 0, 3, 2, 1 })) {
            return true;  // supported case
        }

        if (permute->get_input_size() == 0)
            return false; // unsupported case

        auto input = permute->input(0).get_source_output().get_node_shared_ptr();
        auto input_order = permute->get_input_shape(0);
        // auto inputs = layer->insData.begin()->lock();
        // auto inputsOrder = inputs->getTensorDesc().getDims();
        // cases when all permutations happened either between 1 and X shape where no other dims in between
        auto permute_seq = genPermutations(node_order.begin(), node_order.end());
        auto input_order_transformed = input_order;
        for (auto && permute : permute_seq) {
            // check dims of permuted
            if (input_order_transformed[permute.first] == 1 &&
                input_order_transformed[permute.second] == 1) {
                return true;
            }
            if (input_order_transformed[permute.first] != 1 &&
                input_order_transformed[permute.second] != 1) {
                return false;
            }
            // check dims in between
            for (int j = std::min(permute.first, permute.second) + 1; j < std::max(permute.first, permute.second); j++) {
                if (input_order_transformed[j] != 1) {
                    return false;
                }
            }
            // apply permutation
            std::swap(input_order_transformed[permute.first], input_order_transformed[permute.second]);
        }
        return true;
    }

    bool IsNonFunctionalGNANode(std::shared_ptr<ngraph::Node> node) {
        return std::dynamic_pointer_cast<ngraph::opset8::Reshape>(node) ||
               std::dynamic_pointer_cast<ngraph::opset8::Squeeze>(node) ||
               std::dynamic_pointer_cast<ngraph::opset8::Unsqueeze>(node) ||
               IsTrivialPermute(node);
    }
}// namespace

HandleMultiConnectedLayerToConcat::HandleMultiConnectedLayerToConcat() {
    MATCHER_SCOPE(HandleMultiConnectedLayerToConcat);

    auto concat_op = ngraph::pattern::wrap_type<ngraph::opset8::Concat>();
    ngraph::matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        auto concat = std::dynamic_pointer_cast<ngraph::opset8::Concat>(m.get_match_root());
        if (!concat) return false;

        std::set<std::shared_ptr<ngraph::Node>> inputs;
        // Insert copy layers after concat inputs with multiple connections to concat
        for (size_t i = 0; i < concat->get_input_size(); i++) {
            auto input_op = concat->input(i).get_source_output().get_node_shared_ptr();

            if (inputs.find(input_op) != inputs.end()) {
                InsertCopyLayerBetween(input_op, concat, i);
            } else {
                inputs.insert(input_op);
            }
        }

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(concat_op, matcher_name);
    this->register_matcher(m, callback);
}

bool HandleLayerConnectedToConcatOrMemory::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(HandleLayerConnectedToConcatOrMemory);

    using FuncChildrenInfo = std::tuple<
        std::shared_ptr<ngraph::Node>,   // parent node
        std::shared_ptr<ngraph::Node>,   // child node
        int32_t        // input index
    >;

    // recursively searches for children functional layers skipping non-functional ones
    std::function<std::vector<FuncChildrenInfo>(std::shared_ptr<ngraph::Node>, std::shared_ptr<ngraph::Node>, int32_t)> find_func_layers =
        [&find_func_layers](std::shared_ptr<ngraph::Node> current_node, std::shared_ptr<ngraph::Node> parent_node, int32_t input_idx) {
        if (!IsNonFunctionalGNANode(current_node) ||
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
        if (std::dynamic_pointer_cast<ngraph::opset8::Constant>(node) || IsNonFunctionalGNANode(node))
            continue;

        // // Crop -> Concat, Input -> Split -> Concat and Concat -> Memory cases
        if ((std::dynamic_pointer_cast<ngraph::op::CropIE>(node) && !IsCropAffined(node)) ||
            std::dynamic_pointer_cast<ngraph::opset8::Concat>(node) ||
            std::dynamic_pointer_cast<ngraph::opset8::Split>(node) ||
            std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(node)) {
            std::vector<FuncChildrenInfo> concat_copy_tuples;
            std::vector<FuncChildrenInfo> memory_copy_tuples;
            for (auto output : node->outputs()) {
                auto inputTo = output.get_target_inputs();
                for (auto& child : inputTo) {
                    auto index = child.get_index();
                    auto children_info = find_func_layers(child.get_node()->shared_from_this(), node, index);
                    for (const auto &child_info : children_info) {
                        auto child = std::get<1>(child_info);
                        bool isConcatCase = std::dynamic_pointer_cast<ngraph::op::CropIE>(node) ||
                            std::dynamic_pointer_cast<ngraph::opset8::Split>(node) ||
                            std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(node);

                        if ((std::dynamic_pointer_cast<ngraph::opset8::Concat>(node) || isConcatCase) &&
                           (std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(child) || std::dynamic_pointer_cast<ngraph::op::AssignBase>(child))) {
                                // Concat|Split|Crop -> Memory case
                                memory_copy_tuples.push_back(child_info);
                            } else if (isConcatCase && std::dynamic_pointer_cast<ngraph::opset8::Concat>(child)) {
                                // Split|Crop -> Concat case
                                concat_copy_tuples.push_back(child_info);
                            }
                    }
                }
            }

            if (!memory_copy_tuples.empty() || !concat_copy_tuples.empty())
                is_graph_modified = true;
            for (auto& tuple : memory_copy_tuples) {
                // Concat|Split|Crop -> Memory case
                InsertCopyLayerBetween(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
            }
            for (auto& tuple : concat_copy_tuples) {
                // Split|Crop -> Concat case
                InsertCopyLayerBetween(std::get<0>(tuple), std::get<1>(tuple), std::get<2>(tuple));
            }
        }

        for (auto& output : node->outputs()) {
            auto inputTo = output.get_target_inputs();
            if (inputTo.size() < 2) continue;
            std::vector<std::pair<std::shared_ptr<ngraph::Node>, size_t>> concat_nodes, memory_nodes;
            for (auto& child : inputTo) {
                auto current_node = std::dynamic_pointer_cast<ngraph::Node>(child.get_node()->shared_from_this());
                auto copy_output_node = current_node;
                auto previous_node = node;
                auto current_index = child.get_index();

                while ((IsNonFunctionalGNANode(current_node))) {
                    if (current_node->get_output_size() == 0) break;
                    if (current_node->output(0).get_target_inputs().size()  == 0) break;
                    previous_node = current_node;
                    current_node = current_node->output(0).get_target_inputs().begin()->get_node()->shared_from_this();
                }

                if (std::dynamic_pointer_cast<ngraph::opset8::Concat>(current_node)) {
                    concat_nodes.push_back(std::make_pair(copy_output_node, current_index));
                } else if (std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(current_node) ||
                    std::dynamic_pointer_cast<ngraph::op::AssignBase>(current_node)) {
                    memory_nodes.push_back(std::make_pair(copy_output_node, current_index));
                }
            }

            if (memory_nodes.empty() && concat_nodes.empty()) continue;
            auto count_to_copy = memory_nodes.size() + concat_nodes.size() - (std::dynamic_pointer_cast<ngraph::opset8::Parameter>(node) ? 0 : 1);
            // Insertion of copy to memory layers have a priority on the concat layers
            for (size_t i = 0; i < count_to_copy; i++) {
                auto out_layer = (i < memory_nodes.size()) ? memory_nodes[i].first : concat_nodes[i - memory_nodes.size()].first;
                auto input_id = (i < memory_nodes.size()) ? memory_nodes[i].second : concat_nodes[i - memory_nodes.size()].second;
                InsertCopyLayerBetween(node, out_layer, input_id);
            }
            is_graph_modified = true;
        }
    }

    return is_graph_modified;
}


    // for (auto & l : *pLayers) {
    //     if (!l->outData.size() == 0 &&
    //         !getInputTo(l->outData[0]).size() == 0) continue;

    //     bool bNeedInsertCopyLayer = true;
    //     CNNNetDFS(l, [&l, &bNeedInsertCopyLayer](CNNLayerPtr layer) {
    //         if (!(LayerInfo(layer).isNonFunctional() || LayerInfo(layer).isSplit() || LayerInfo(layer).isCrop() || LayerInfo(layer).isInput())) {
    //             bNeedInsertCopyLayer = false;
    //         }
    //         }, true, [&bNeedInsertCopyLayer](InferenceEngine::CNNLayer* from) {
    //                 // aborting UFS if we found functional layer (excluding Splits and Crops)
    //                 return make_upstream_order(bNeedInsertCopyLayer ? from : nullptr);
    //         });

    //     if (bNeedInsertCopyLayer) {
    //         for (size_t inputIdx = 0; inputIdx < l->insData.size(); ++inputIdx) {
    //             IE_ASSERT(l->insData[inputIdx].lock() != nullptr);
    //             auto inputData = l->insData[inputIdx].lock();
    //             auto parentLayer = getCreatorLayer(inputData);
    //             IE_ASSERT(parentLayer.lock() != nullptr);
    //             InsertCopyLayer(parentLayer.lock(), l, inputIdx, this->getPassManager(), CopyLayerName);
    //         }
    //     }
    // }


bool HandleNonComputationalSubgraphs::run_on_model(const std::shared_ptr<ngraph::Function>& f) {
    RUN_ON_FUNCTION_SCOPE(HandleNonComputationalSubgraphs);
    bool is_graph_modified = false;

    for (auto& node : f->get_ordered_ops()) {
        if (!std::dynamic_pointer_cast<ngraph::opset8::Result>(node))
            continue;
        bool bNeedInsertCopyLayer = true;
        std::unordered_map<std::shared_ptr<ngraph::Node>, bool> visited;
        std::vector<std::shared_ptr<ngraph::Node>> dfs_v;
        auto start_node = node->get_input_node_shared_ptr(0);
        dfs_v.push_back(start_node);

        while (!dfs_v.empty()) {
            auto current_node = dfs_v.back();
            dfs_v.pop_back();

            if (visited[current_node])
                continue;

            if (!(IsNonFunctionalGNANode(current_node) ||
                 std::dynamic_pointer_cast<ngraph::op::CropIE>(current_node) ||
                 std::dynamic_pointer_cast<ngraph::opset8::Split>(current_node) ||
                 std::dynamic_pointer_cast<ngraph::opset8::VariadicSplit>(current_node) ||
                 std::dynamic_pointer_cast<ngraph::opset8::Parameter>(current_node) ||
                 std::dynamic_pointer_cast<ngraph::opset8::Constant>(current_node))) {
                bNeedInsertCopyLayer = false;
                break;
            }
            visited[current_node] = true;

            for (size_t i = 0; i < current_node->get_input_size(); i++) {
                auto input_node = current_node->get_input_node_shared_ptr(i);
                if (!visited[input_node])
                    dfs_v.push_back(input_node);
            }
        }

        if (bNeedInsertCopyLayer) {
            is_graph_modified = true;
            for (size_t i = 0; i < start_node->get_input_size(); i++) {
                auto prev_layer = start_node->get_input_node_shared_ptr(i);
                if (!std::dynamic_pointer_cast<ngraph::opset8::Constant>(prev_layer))
                    InsertCopyLayerBetween(prev_layer, start_node, i);
            }
        }
    }

    return is_graph_modified;
}

} // namespace GNAPluginNS
