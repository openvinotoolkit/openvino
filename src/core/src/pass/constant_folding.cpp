// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/constant_folding.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/constant_fold_utils.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/util/read_value_base.hpp"
#include "openvino/op/util/shape_of_base.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

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
const auto friendly_name_from = [](const ov::Node& node, const size_t output_count, const size_t idx) -> std::string {
    constexpr auto single_output = static_cast<size_t>(1);
    auto name = node.get_friendly_name();
    if (single_output != output_count) {
        name.append(".").append(std::to_string(idx));
    }
    return name;
};

static bool restore_original_input_precision(const std::shared_ptr<ov::Node>& node) {
    bool restored = false;
    if (ov::is_type<ov::op::v0::Convert>(node)) {
        auto input = node->input(0);
        ov::util::remove_original_input_precision_attribute(input);
        return restored;
    }
    for (size_t i = 0; i < node->get_input_size(); i++) {
        auto input = node->input(i);
        if (!ov::util::has_original_input_precision(input))
            continue;
        const auto original_type = ov::util::get_original_input_precision(input);
        ov::util::remove_original_input_precision_attribute(input);
        if (original_type != node->get_input_element_type(i)) {
            auto convert = std::make_shared<ov::op::v0::Convert>(node->input_value(i), original_type);
            ov::OutputVector replacements(1);
            OPENVINO_ASSERT(convert->constant_fold(replacements, convert->input_values()));
            replacements[0].get_node()->set_friendly_name(node->get_input_node_ptr(i)->get_friendly_name());
            input.replace_source_output(replacements[0]);
            restored = true;
        }
    }
    return restored;
}

class RequiresPrecisionConversion : public ov::RuntimeAttribute {
public:
    OPENVINO_RTTI("requires_precision_conversion", "0");

    bool is_copyable() const override {
        return false;
    }
};

static void mark_node_requires_precision_conversion(const std::shared_ptr<ov::Node>& node) {
    node->get_rt_info()[RequiresPrecisionConversion::get_type_info_static()] = RequiresPrecisionConversion{};
}

static bool node_has_requires_precision_conversion_attribute(const std::shared_ptr<const ov::Node>& node) {
    return node->get_rt_info().count(RequiresPrecisionConversion::get_type_info_static()) > 0;
}

static void remove_requires_precision_conversion_attribute(const std::shared_ptr<ov::Node>& node) {
    auto& rt_info = node->get_rt_info();
    auto it = rt_info.find(RequiresPrecisionConversion::get_type_info_static());
    if (it != rt_info.end()) {
        rt_info.erase(it);
    }
}

bool ov::pass::ConstantFolding::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_MODEL_SCOPE(ConstantFolding);

    bool rewritten = pre_calculated_values_folding(model);

    for (const auto& original_node : model->get_ordered_ops()) {
        auto node = original_node;
        if (!original_node->can_constant_fold(original_node->input_values())) {
            if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::MultiSubGraphOp>(node)) {
                // recursively constant fold operators containing subgraphs (ie: TensorIterator, Loop)
                size_t sub_graphs_num = sub_graph_node->get_internal_subgraphs_size();
                for (size_t sub_graph_ind = 0; sub_graph_ind < sub_graphs_num; ++sub_graph_ind) {
                    rewritten =
                        run_on_model(sub_graph_node->get_function(static_cast<int>(sub_graph_ind))) || rewritten;
                }
            }
            rewritten = restore_original_input_precision(original_node) || rewritten;
            if (rewritten) {
                original_node->validate_and_infer_types();
            }
            continue;
        }
        if (node_has_requires_precision_conversion_attribute(node)) {
            remove_requires_precision_conversion_attribute(node);
            node = util::convert_to_supported_precision(node.get());
        } else {
            rewritten = restore_original_input_precision(node) || rewritten;
        }

        if (rewritten) {
            node->validate_and_infer_types();
        }

        OutputVector replacements(node->get_output_size());
        if (node->constant_fold(replacements, node->input_values())) {
            OPENVINO_ASSERT(!constant_folding_is_disabled(original_node),
                            "Node folded but constant folding disabled. Check constant_fold implementation for ",
                            node);
            OPENVINO_ASSERT(replacements.size() == node->get_output_size(),
                            "constant_fold_default returned incorrect number of replacements for ",
                            node);

            for (size_t i = 0; i < replacements.size(); ++i) {
                auto node_output = original_node->output(i);
                const auto& replacement = replacements.at(i);
                auto replacement_ptr = replacement.get_node_shared_ptr();
                if (replacement_ptr && (node_output != replacement)) {
                    replacement_ptr->set_friendly_name(friendly_name_from(*original_node, replacements.size(), i));

                    node_output.replace(replacement);
                    // Copy runtime info from source nodes
                    // when it was not propogated during pre-calculation
                    copy_runtime_info_from_input_values(original_node);
                    // Propagate runtime info attributes to replacement
                    copy_runtime_info(original_node, replacement_ptr);

                    rewritten = true;
                }
            }
        } else {
            // if CF was unsuccessful remove original precision attribute from inputs
            bool restored = restore_original_input_precision(original_node);
            if (restored) {
                original_node->validate_and_infer_types();
                rewritten = true;
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
        bool node_has_disabled_constant_folding = constant_folding_is_disabled(node);

        // During constant folding process, current node's input precision may not match
        // the node's original input precision. After we removed evaluates for some types (like f16, bf16)
        // we need to convert constants with those types to f32. And at some point - this f32 constant may
        // become an input to a node that's not constfoldable. Then we need to convert that constant back to
        // that input's original precision.
        util::save_original_input_precisions(node);
        if (!node_has_disabled_constant_folding && util::node_requires_precision_conversion(node.get())) {
            mark_node_requires_precision_conversion(node);
        }

        if (node_has_disabled_constant_folding) {
            can_be_folded = false;
        } else if (is_type<op::util::ShapeOfBase>(node)) {
            // In case if node is ShapeOf operation we stop propagation of can_be_folded attribute. We have to limit
            // propagation because we can't detect borders of shape_of sub-graphs, so we propagate can_be_folded
            // attribute through all nodes including nodes on data path. So to limit the spread of attribute to other
            // shape-of sub-graphs we do not propagate it through ShapeOf nodes.
            can_be_folded = true;
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

    std::deque<std::shared_ptr<Node>> nodes;
    std::set<std::shared_ptr<Node>> visited;
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
