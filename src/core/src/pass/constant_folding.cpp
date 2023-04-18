// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/constant_folding.hpp"

#include <openvino/cc/pass/itt.hpp>

#include "openvino/core/rt_info.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/op/util/sub_graph_base.hpp"
#include "openvino/opsets/opset11.hpp"

using namespace std;

/**
 * \brief Check if \ref ov::Output<ov::Node> can be folded base on `can_be_folded` attribute.
 *
 * \param output  Node to check.
 *
 * \return true if output can be folded otherwise false.
 */
const auto is_output_foldable = [](const ov::Output<ov::Node>& output, const std::string& key = "can_be_folded") {
    const auto& rt_info = output.get_node()->get_rt_info();
    return rt_info.count(key) && rt_info.at(key).as<bool>();
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

const auto fold_gather = [](const std::shared_ptr<ov::Node>& current_node) -> bool {
    auto data_out = current_node->get_input_node_shared_ptr(0);
    if (!ov::is_type<ov::op::util::ShapeOfBase>(data_out)) {
        return false;
    }
    const auto& out_pshape = data_out->get_input_partial_shape(0);
    if (out_pshape.rank().is_dynamic() || !is_output_foldable(data_out, "can_be_partially_folded")) {
        return false;
    }

    const auto indices = ov::as_type_ptr<ov::opset11::Constant>(current_node->get_input_node_shared_ptr(1));
    if (!indices)
        return false;

    const auto rank = out_pshape.rank().get_length();
    std::set<size_t> dyn_indices;

    // In some cases, Gather doesn't use dynamic dimensions from its 1st input `ShapeOf(data)`
    // in `indices` argument, so we can create a constant after ShapeOf with any values instead
    // of the dynamic dims.
    int64_t stub = 1;
    std::vector<int64_t> shape_with_stubs(rank);
    for (int64_t i = 0; i < rank; ++i) {
        if (out_pshape[i].is_dynamic()) {
            dyn_indices.insert(i);
            shape_with_stubs[i] = stub;
        } else {
            shape_with_stubs[i] = out_pshape[i].get_length();
        }
    }

    auto indices_values = indices->cast_vector<int64_t>();
    for (auto idx : indices_values) {
        if (idx < 0) {
            idx += rank;
        }
        if (dyn_indices.count(idx)) {
            return false;
        }
    }

    auto replacement = std::make_shared<ov::op::v0::Constant>(data_out->get_output_element_type(0),
                                                              ov::Shape{shape_with_stubs.size()},
                                                              shape_with_stubs);

    current_node->input(0).replace_source_output(replacement);
    auto& rt_info = data_out->get_rt_info();
    rt_info.erase("can_be_partially_folded");
    // Propagate runtime info attributes to replacement
    copy_runtime_info(data_out, replacement);
    return true;
};

/**
 * \brief Folds the inputs of the `current_node` when ConstantFolding depends on node's logic.
 * For example, when Gather has dynamic dims in the first inputs after ShapeOf, but
 * operates with non-dynamic data specified in 'indices' argument.
 *
 * \param current_node Node which inputs can be folded.
 *
 * \return true if inputs can be folded otherwise false.
 */
const auto apply_additional_folding_rules = [](const std::shared_ptr<ov::Node>& current_node) -> bool {
    if (ov::as_type_ptr<ov::op::util::GatherBase>(current_node)) {
        return fold_gather(current_node);
    }
    // no additional rules found for the node.
    return false;
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
                    // Copy runtime info from source nodes
                    // when it was not propogated during pre-calculation
                    copy_runtime_info_from_input_values(node);
                    // Propagate runtime info attributes to replacement
                    copy_runtime_info(node, replacement.get_node_shared_ptr());

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

void ov::pass::ConstantFolding::copy_runtime_info_from_input_values(const std::shared_ptr<Node>& node) {
    if (is_type<op::util::ShapeOfBase>(node)) {
        // Don't propogate names of ShapeOf source node since it is not fused itself
        return;
    }
    // Add node itself to merge original rt info with rt info of inputs
    ov::NodeVector from = {node};
    for (auto& input : node->input_values()) {
        from.push_back(input.get_node_shared_ptr());
    }
    copy_runtime_info(from, node);
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
        } else if (is_type<op::util::ShapeOfBase>(node)) {
            // In case if node is ShapeOf operation we stop propagation of can_be_folded attribute. We have to limit
            // propagation because we can't detect borders of shape_of sub-graphs, so we propagate can_be_folded
            // attribute through all nodes including nodes on data path. So to limit the spread of attribute to other
            // shape-of sub-graphs we do not propagate it through ShapeOf nodes.
            auto pshape = input_values.begin()->get_partial_shape();
            can_be_folded = pshape.is_static();

            // In some cases, we don't need that all dims in pshape to be static, e.g.
            // Gather(shape_of, indices, axis), where `indices` arg doesn't contain indices
            // of dynamic dims in `shape_of`.
            if (!can_be_folded && pshape.rank().is_static()) {
                node->get_rt_info()["can_be_partially_folded"] = true;
            }
        } else if (op::util::is_parameter(node) || op::util::is_output(node) || op::util::is_sink(node) ||
                   is_type<op::util::ReadValueBase>(node)) {
            can_be_folded = false;
        } else {
            can_be_folded = std::all_of(input_values.cbegin(), input_values.cend(), is_output_foldable);
            if (input_values.size() && can_be_folded) {
                copy_runtime_info_from_input_values(node);
            }
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
                const auto& lower = output.get_tensor().get_lower_value();
                auto replacement =
                    std::make_shared<ov::op::v0::Constant>(lower.get_element_type(), lower.get_shape(), lower.data());
                if (replacement && !ov::is_type<ov::op::v0::Constant>(input_node)) {
                    replacement->set_friendly_name(
                        friendly_name_from(*input_node, input_node->get_output_size(), output.get_index()));

                    output.replace(replacement);
                    // Propagate runtime info attributes to replacement
                    copy_runtime_info(input_node, replacement);

                    rewritten = true;
                }
            } else if (is_output_foldable(output, "can_be_partially_folded")) {
                // it's not necessary to continue searching
                // this output will be processed in apply_additional_folding_rules function if possible
                continue;
            } else {
                // continue searching
                const auto& input_node = output.get_node_shared_ptr();
                nodes.push_front(input_node);
            }
        }

        rewritten = rewritten || apply_additional_folding_rules(curr_node);
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
