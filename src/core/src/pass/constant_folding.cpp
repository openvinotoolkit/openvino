// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/constant_folding.hpp"

#include <openvino/cc/pass/itt.hpp>

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset3.hpp"

using namespace std;

/**
 * \brief Check if \ref ov::Output<ov::Node> can be folded base on `can_be_folded` attribute.
 *
 * \param output  Node to check.
 *
 * \return true if output can be folded otherwise false.
 */
const auto is_output_foldable = [](const ov::Output<ov::Node>& output) {
    const auto& rt_info = output.get_node()->get_rt_info();
    return !rt_info.count("can_be_folded") || rt_info.at("can_be_folded").as<bool>();
};

/**
 * \brief Create new friendly name from node.
 *
 * New friendly name has format:
 * - {node friendly name}        if output_count 1
 * - {node friendly name}.{idx}  if output count not 1.
 *
 * \param node          Node to get friendly name.
 * \param output_count  Node output count (before folding).
 * \param idx           Node current output index.
 *
 * \return std::string with new friendly name.
 */
const auto friendly_name_from = [](const ov::Node& node, const size_t output_count, const size_t idx) {
    constexpr auto single_output = static_cast<size_t>(1);

    if (single_output == output_count) {
        return node.get_friendly_name();
    } else {
        return node.get_friendly_name() + "." + std::to_string(idx);
    }
};

bool ov::pass::ConstantFolding::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(ConstantFolding);
    bool rewritten = pre_calculated_values_folding(model);

    for (const auto& node : model->get_ordered_ops()) {
        if (rewritten) {
            node->validate_and_infer_types();
        }

        OutputVector replacements(node->get_output_size());

        if (node->constant_fold(replacements, node->input_values())) {
            OPENVINO_ASSERT(!constant_folding_is_disabled(node),
                            "Node folded but constant folding disabled. Check constant_fold implementation for ",
                            node);
            OPENVINO_ASSERT(replacements.size() == node->get_output_size(),
                            "constant_fold_default returned incorrect number of replacements for ",
                            node);

            for (size_t i = 0; i < replacements.size(); ++i) {
                auto node_output = node->output(i);
                auto replacement = replacements.at(i);
                if (replacement.get_node_shared_ptr() && (node_output != replacement)) {
                    replacement.get_node()->set_friendly_name(friendly_name_from(*node, replacements.size(), i));

                    node_output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(node, replacement);

                    rewritten = true;
                }
            }
        } else {
            // recursively constant fold operators containing subgraphs (ie: TensorIterator, Loop)
            if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node)) {
                size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
                for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                    rewritten |= run_on_model(sub_graph_node->get_function(static_cast<int>(sub_graph_ind)));
                }
            }
        }
    }

    return rewritten;
}

void ov::pass::ConstantFolding::copy_runtime_info_to_target_inputs(const std::shared_ptr<Node>& node,
                                                                   const Output<Node>& replacement) {
    for (auto& input : replacement.get_target_inputs()) {
        auto consumer = input.get_node()->shared_from_this();
        copy_runtime_info({node, consumer}, consumer);
    }
}

bool ov::pass::ConstantFolding::pre_calculated_values_folding(const std::shared_ptr<ov::Model>& model) {
    // IsOutputNodeFoldable is_output_foldable;
    // To avoid excess graph traversals we have to manually propagate DisableConstantFolding with some
    // temporary attribute which indicates that the node which is marked with this attribute can't be folded because
    // it is included into not foldable sub-graph.
    for (auto&& node : model->get_ordered_ops()) {
        const auto& input_values = node->input_values();
        bool can_be_folded;

        if (constant_folding_is_disabled(node)) {
            can_be_folded = false;
        } else if (is_type<ov::opset1::ShapeOf>(node) || is_type<ov::opset3::ShapeOf>(node)) {
            // In case if node is ShapeOf operation we stop propagation of can_be_folded attribute. We have to limit
            // propagation because we can't detect borders of shape_of sub-graphs, so we propagate can_be_folded
            // attribute through all nodes including nodes on data path. So to limit the spread of attribute to other
            // shape-of sub-graphs we do not propagate it through ShapeOf nodes.
            can_be_folded = true;
        } else {
            can_be_folded = std::all_of(input_values.cbegin(), input_values.cend(), is_output_foldable);
        }
        node->get_rt_info()["can_be_folded"] = can_be_folded;
    }

    deque<shared_ptr<Node>> nodes;
    set<shared_ptr<Node>> visited;
    for (auto& r : model->get_results())
        nodes.push_back(r);
    for (auto& r : model->get_sinks())
        nodes.emplace_back(r);

    bool rewritten = false;
    while (!nodes.empty()) {
        auto curr_node = nodes.front();
        nodes.pop_front();
        if (visited.count(curr_node) || ov::is_type<ov::op::v0::Constant>(curr_node))
            continue;
        visited.insert(curr_node);

        for (auto& output : curr_node->input_values()) {
            if (is_output_foldable(output) && output.get_tensor().has_and_set_bound()) {
                auto input_node = output.get_node_shared_ptr();
                auto replacement = std::make_shared<ov::op::v0::Constant>(output.get_tensor().get_lower_value());
                if (replacement && !ov::is_type<ov::op::v0::Constant>(input_node)) {
                    replacement->set_friendly_name(
                        friendly_name_from(*input_node, input_node->get_output_size(), output.get_index()));

                    output.replace(replacement);
                    // Propagate runtime info attributes to replacement consumer nodes
                    copy_runtime_info_to_target_inputs(input_node, replacement);

                    rewritten = true;
                }
            } else {
                // continue searching
                const auto& input_node = output.get_node_shared_ptr();
                nodes.push_front(input_node);
            }
        }
    }
    return rewritten;
}

void ov::pass::disable_constant_folding(const std::shared_ptr<Node>& node) {
    node->get_rt_info().emplace(DisableConstantFolding::get_type_info_static(), DisableConstantFolding{});
}

void ov::pass::enable_constant_folding(const std::shared_ptr<Node>& node) {
    node->get_rt_info().erase(DisableConstantFolding::get_type_info_static());
}

bool ov::pass::constant_folding_is_disabled(const std::shared_ptr<Node>& node) {
    return constant_folding_is_disabled(node.get());
}

bool ov::pass::constant_folding_is_disabled(const Node* const node) {
    OPENVINO_ASSERT(node, "node is nullptr");
    return node->get_rt_info().count(DisableConstantFolding::get_type_info_static());
}
