// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/validation_util.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/gather_elements.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/util/sub_graph_base.hpp>
#include <transformations/common_optimizations/shared_ops_optimization.hpp>

#include "itt.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

bool shared_node_optimization(
    const std::shared_ptr<ov::Model>& model,
    const std::unordered_map<ov::Node::type_info_t, bool (*)(const Node*, const Node*)>& rules) {
    bool rewritten = false;

    for (const auto& op : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto sub_graph_node = std::dynamic_pointer_cast<ov::op::util::SubGraphOp>(op)) {
            if (auto sub_graph = sub_graph_node->get_function()) {
                rewritten = shared_node_optimization(sub_graph, rules) || rewritten;
            }
        }
        for (auto& output : op->outputs()) {
            const auto& target_inputs = output.get_target_inputs();
            if (target_inputs.size() <= 1)
                continue;  // nothing to optimize
            std::unordered_map<ov::Node::type_info_t, std::vector<ov::Node*>> type_to_node;
            for (const auto& input : target_inputs) {
                auto node = input.get_node();
                if (node && rules.count(node->get_type_info()))
                    type_to_node[node->get_type_info()].push_back(node);
            }
            for (auto& item : type_to_node) {
                const auto& shared_nodes = item.second;
                if (shared_nodes.size() < 2)
                    continue;
                const auto& ops_type = item.first;
                const auto& are_equal = rules.at(ops_type);
                const auto& root_op = shared_nodes[0];
                for (const auto& child_op : shared_nodes) {
                    if (root_op->get_instance_id() != child_op->get_instance_id() && are_equal(root_op, child_op)) {
                        rewritten = replace_output_update_name(child_op->output(0), root_op->output(0)) || rewritten;
                    }
                }
            }
        }
    }
    return rewritten;
}

bool inputs_from_same_source_or_equal_constants(const Node* lhs, const Node* rhs) {
    if (lhs->get_input_size() != rhs->get_input_size())
        return false;
    size_t input_size = lhs->get_input_size();
    for (size_t i = 0; i < input_size; ++i) {
        if (lhs->input_value(i) == rhs->input_value(i))
            continue;
        OPENVINO_SUPPRESS_DEPRECATED_START
        auto lhs_constant = ov::get_constant_from_source(lhs->input_value(i));
        auto rhs_constant = ov::get_constant_from_source(rhs->input_value(i));
        OPENVINO_SUPPRESS_DEPRECATED_END
        if (!lhs_constant || !rhs_constant)
            return false;
        if (lhs_constant->get_shape() != rhs_constant->get_shape() ||
            ov::shape_size(lhs_constant->get_shape()) > 10)  // protection for not comparing huge constants
            return false;
        if (lhs_constant->get_element_type() != rhs_constant->get_element_type())
            return false;
        if (lhs_constant->get_element_type().is_integral()) {
            if (lhs_constant->cast_vector<int64_t>() != rhs_constant->cast_vector<int64_t>())
                return false;
        } else if (lhs_constant->get_element_type().is_real()) {
            if (lhs_constant->cast_vector<double>() != rhs_constant->cast_vector<double>())
                return false;
        } else {
            return false;
        }
    }
    return true;
}

bool concats_are_equal(const Node* lhs, const Node* rhs) {
    const auto* lhs_concat = as_type<const v0::Concat>(lhs);
    if (!lhs_concat)
        return false;
    const auto* rhs_concat = as_type<const v0::Concat>(rhs);
    if (!rhs_concat)
        return false;
    return lhs_concat->get_axis() == rhs_concat->get_axis() && inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool gather_elements_are_equal(const Node* lhs, const Node* rhs) {
    const auto* lhs_gather_elements = as_type<const v6::GatherElements>(lhs);
    if (!lhs_gather_elements)
        return false;
    const auto* rhs_gather_elements = as_type<const v6::GatherElements>(rhs);
    if (!rhs_gather_elements)
        return false;
    return lhs_gather_elements->get_axis() == rhs_gather_elements->get_axis() &&
           inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool reshapes_are_equal(const Node* lhs, const Node* rhs) {
    const auto* lhs_reshape = as_type<const v1::Reshape>(lhs);
    if (!lhs_reshape)
        return false;
    const auto* rhs_reshape = as_type<const v1::Reshape>(rhs);
    if (!rhs_reshape)
        return false;
    return lhs_reshape->get_special_zero() == rhs_reshape->get_special_zero() &&
           inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool ov::pass::SharedOpOptimization::run_on_model(const std::shared_ptr<ov::Model>& model) {
    RUN_ON_FUNCTION_SCOPE(SharedOpOptimization);
#define RECORD(op, func) \
    { op::get_type_info_static(), func }

    const std::unordered_map<ov::Node::type_info_t, bool (*)(const Node*, const Node*)> rules = {
        // no attributes
        RECORD(v8::Slice, inputs_from_same_source_or_equal_constants),
        RECORD(v0::Tile, inputs_from_same_source_or_equal_constants),
        RECORD(v1::Transpose, inputs_from_same_source_or_equal_constants),

        // with attributes
        RECORD(v0::Concat, concats_are_equal),
        RECORD(v6::GatherElements, gather_elements_are_equal),
        RECORD(v1::Reshape, reshapes_are_equal),

    };  // TODO: use visit_attributes to uniformly perform attributes check in the future and get rid of rules table
    return shared_node_optimization(model, rules);
}
