// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/shared_ops_optimization.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/op/util/sub_graph_base.hpp"

using namespace std;
using namespace ov;
using namespace ov::op;

namespace {
using rules_t = unordered_map<Node::type_info_t, bool (*)(const Node*, const Node*)>;

bool shared_node_optimization(const shared_ptr<Model>& model, const rules_t& rules) {
    bool rewritten = false;

    for (const auto& op : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto multi_subgraph_op = dynamic_pointer_cast<op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                if (sub_graph)
                    rewritten |= shared_node_optimization(sub_graph, rules);
            }
        }
        for (auto& output : op->outputs()) {
            const auto& target_inputs = output.get_target_inputs();
            if (target_inputs.size() <= 1)
                continue;  // nothing to optimize
            unordered_map<Node::type_info_t, vector<Node*>> type_to_node;
            for (const auto& input : target_inputs)
                if (auto node = input.get_node())
                    if (rules.count(node->get_type_info()))
                        type_to_node[node->get_type_info()].push_back(node);
            for (auto& item : type_to_node) {
                auto& shared_nodes = item.second;
                if (shared_nodes.size() < 2)
                    continue;
                const auto& are_equal = rules.at(item.first);

                std::vector<bool> visited_nodes(shared_nodes.size(), false);
                for (size_t i = 0; i < visited_nodes.size(); ++i) {
                    if (visited_nodes[i])
                        continue;
                    const auto& root_op = shared_nodes[i];
                    visited_nodes[i] = true;
                    for (size_t j = i + 1; j < visited_nodes.size(); ++j) {
                        if (visited_nodes[j])
                            continue;
                        const auto& child_op = shared_nodes[j];
                        if (are_equal(root_op, child_op)) {
                            rewritten |= replace_output_update_name(child_op->output(0), root_op->output(0));
                            visited_nodes[j] = true;
                        }
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
    if (lhs->get_type_info() != rhs->get_type_info())
        return false;
    size_t input_size = lhs->get_input_size();
    for (size_t i = 0; i < input_size; ++i) {
        if (lhs->input_value(i) == rhs->input_value(i))
            continue;
        auto lhs_constant = as_type_ptr<v0::Constant>(lhs->get_input_node_shared_ptr(i));
        auto rhs_constant = as_type_ptr<v0::Constant>(rhs->get_input_node_shared_ptr(i));
        if (!lhs_constant || !rhs_constant)
            return false;
        if (lhs_constant->get_element_type() != rhs_constant->get_element_type())
            return false;
        const auto& lhs_shape = lhs_constant->get_output_partial_shape(0).to_shape();
        if (lhs_shape != rhs_constant->get_output_partial_shape(0).to_shape() || shape_size(lhs_shape) > 10)
            return false;
        if (memcmp(lhs_constant->get_data_ptr(), rhs_constant->get_data_ptr(), lhs_constant->get_byte_size()) != 0)
            return false;
    }
    return true;
}

bool concats_are_equal(const Node* lhs, const Node* rhs) {
    const auto lhs_concat = as_type<const v0::Concat>(lhs);
    if (!lhs_concat)
        return false;
    const auto rhs_concat = as_type<const v0::Concat>(rhs);
    if (!rhs_concat)
        return false;
    return lhs_concat->get_axis() == rhs_concat->get_axis() && inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool gather_elements_are_equal(const Node* lhs, const Node* rhs) {
    const auto lhs_gather_elements = as_type<const v6::GatherElements>(lhs);
    if (!lhs_gather_elements)
        return false;
    const auto rhs_gather_elements = as_type<const v6::GatherElements>(rhs);
    if (!rhs_gather_elements)
        return false;
    return lhs_gather_elements->get_axis() == rhs_gather_elements->get_axis() &&
           inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool reshapes_are_equal(const Node* lhs, const Node* rhs) {
    const auto lhs_reshape = as_type<const v1::Reshape>(lhs);
    if (!lhs_reshape)
        return false;
    const auto rhs_reshape = as_type<const v1::Reshape>(rhs);
    if (!rhs_reshape)
        return false;
    return lhs_reshape->get_special_zero() == rhs_reshape->get_special_zero() &&
           inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool shapeof_are_equal(const Node* lhs, const Node* rhs) {
    auto lhs_output_et = element::i64, rhs_output_et = element::i64;
    if (const auto shape = as_type<const v3::ShapeOf>(lhs)) {
        lhs_output_et = shape->get_output_type();
    } else if (!as_type<const v0::ShapeOf>(lhs)) {
        return false;
    }
    if (const auto shape = as_type<const v3::ShapeOf>(rhs)) {
        rhs_output_et = shape->get_output_type();
    } else if (!as_type<const v0::ShapeOf>(rhs)) {
        return false;
    }
    return lhs_output_et == rhs_output_et && inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool gathers_are_equal(const Node* lhs, const Node* rhs) {
    const auto l_gather = as_type<const op::util::GatherBase>(lhs);
    const auto r_gather = as_type<const op::util::GatherBase>(rhs);
    if (!l_gather || !r_gather)
        return false;
    return l_gather->get_batch_dims() == r_gather->get_batch_dims() &&
           inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool converts_are_equal(const Node* lhs, const Node* rhs) {
    const auto l_convert = as_type<const v0::Convert>(lhs);
    const auto r_convert = as_type<const v0::Convert>(rhs);
    if (!l_convert || !r_convert)
        return false;
    return l_convert->get_destination_type() == r_convert->get_destination_type() &&
           inputs_from_same_source_or_equal_constants(lhs, rhs);
}

bool shape_of_upgrade(const shared_ptr<Model>& model) {
    bool rewritten = false;
    for (const auto& op : model->get_ordered_ops()) {
        // Recursively apply transformation for sub-graph based operations
        if (auto multi_subgraph_op = dynamic_pointer_cast<op::util::MultiSubGraphOp>(op)) {
            for (const auto& sub_graph : multi_subgraph_op->get_functions()) {
                if (sub_graph)
                    rewritten |= shape_of_upgrade(sub_graph);
            }
        } else if (auto v1_shape_of = ov::as_type_ptr<v0::ShapeOf>(op)) {
            auto v3_shape_of = std::make_shared<v3::ShapeOf>(v1_shape_of->input_value(0), element::i64);
            v3_shape_of->set_friendly_name(v1_shape_of->get_friendly_name());
            ov::replace_output_update_name(v1_shape_of, v3_shape_of);
            rewritten = true;
        }
    }
    return rewritten;
}

}  // namespace
bool pass::SharedOpOptimization::run_on_model(const shared_ptr<Model>& model) {
    RUN_ON_FUNCTION_SCOPE(SharedOpOptimization);
#define RECORD_NO_ATTRIBUTES(operation) \
    { operation::get_type_info_static(), inputs_from_same_source_or_equal_constants }
#define RECORD(operation, func) \
    { operation::get_type_info_static(), func }

    const rules_t rules = {
        // no attributes
        RECORD_NO_ATTRIBUTES(v8::Slice),
        RECORD_NO_ATTRIBUTES(v0::Squeeze),
        RECORD_NO_ATTRIBUTES(v0::Tile),
        RECORD_NO_ATTRIBUTES(v0::Unsqueeze),

        // with attributes
        RECORD(v0::Concat, concats_are_equal),
        RECORD(v0::Convert, converts_are_equal),
        RECORD(v1::Gather, gathers_are_equal),
        RECORD(v7::Gather, gathers_are_equal),
        RECORD(v8::Gather, gathers_are_equal),
        RECORD(v6::GatherElements, gather_elements_are_equal),
        RECORD(v1::Reshape, reshapes_are_equal),
        RECORD(v0::ShapeOf, shapeof_are_equal),
        RECORD(v3::ShapeOf, shapeof_are_equal),
    };  // TODO: use visit_attributes to uniformly perform attributes check in the future and get rid of rules table

    bool rewritten = shape_of_upgrade(model);
    rewritten |= shared_node_optimization(model, rules);
    return rewritten;
}
