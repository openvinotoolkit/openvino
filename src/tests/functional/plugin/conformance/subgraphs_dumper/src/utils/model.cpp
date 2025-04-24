// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"
#include "utils/model.hpp"

namespace ov {
namespace util {

std::string get_model_type(const std::shared_ptr<ov::Model>& model) {
    if (ov::util::is_dynamic_model(model)) {
        return "dynamic";
    }
    return "static";
}

std::map<std::string, ov::conformance::InputInfo>
get_input_info_by_model(const std::shared_ptr<ov::Model>& model) {
    std::map<std::string, ov::conformance::InputInfo> in_info;
    for (const auto& node : model->get_ordered_ops()) {
        ov::conformance::InputInfo::Range ranges(ov::conformance::DEFAULT_MIN_VALUE, ov::conformance::DEFAULT_MAX_VALUE);
        bool is_const = false;
        if (ov::shape_size(node->get_output_shape(0)) != 0 && ov::op::util::is_constant(node)) {
            std::shared_ptr<ov::op::v0::Constant> constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            auto const_ranges = get_const_ranges(constant,
                                                 constant->get_default_output().get_element_type());
            ranges = const_ranges;
        } else if (!ov::op::util::is_parameter(node)) {
            continue;
        }
        auto partial_shape = node->get_default_output().get_partial_shape();
        in_info.insert({node->get_friendly_name(),
                        ov::conformance::InputInfo(partial_shape, ranges.min, ranges.max, is_const)});
    }
    return in_info;
}

std::map<std::string, ov::conformance::InputInfo>
align_input_info(const std::shared_ptr<ov::Model>& model,
                 const std::shared_ptr<ov::Model>& model_ref,
                 const std::map<std::string, ov::conformance::InputInfo>& in_info,
                 const std::map<std::string, ov::conformance::InputInfo>& in_info_ref,
                 const std::unordered_map<std::string, std::string> &matched_op) {
    std::map<std::string, ov::conformance::InputInfo> updated_input_info(in_info_ref);
    for (const auto& op : model->get_ordered_ops()) {
        const auto op_name = op->get_friendly_name();
        if (!in_info.count(op_name)) {
            continue;
        }
        if (matched_op.count(op_name) && in_info_ref.count(matched_op.at(op_name))) {
            updated_input_info[matched_op.at(op_name)] = in_info.at(op_name);
        }
    }
    return updated_input_info;
}

void
get_subgraph_set_node(std::unordered_set<std::shared_ptr<ov::Node>>& nodes_to_check,
                      const std::shared_ptr<ov::Node>& node) {
    if (nodes_to_check.empty()) {
        nodes_to_check.insert(node);
    }
    for (size_t out_idx = 0; out_idx < node->outputs().size(); ++out_idx) {
        for (const auto& out : node->get_output_target_inputs(out_idx)) {
            const auto& output_node = out.get_node()->shared_from_this();
            if (ov::op::util::is_output(output_node)) {
                return;
            }
            if (!nodes_to_check.count(output_node)) {
                nodes_to_check.insert(output_node);
                get_subgraph_set_node(nodes_to_check, output_node);
            }
        }
    }
    return;
}

bool is_same_paired_op_cnt(const std::shared_ptr<ov::Model> &fist_model,
                           const std::shared_ptr<ov::Model> &second_model) {
    size_t fist_paired_op_cnt = 0;
    size_t second_paired_op_cnt = 0;

    for (auto& node : fist_model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::util::ReadValueBase>(node) ||
            ov::as_type_ptr<ov::op::util::AssignBase>(node))
            fist_paired_op_cnt++;
    }

    for (auto& node : second_model->get_ordered_ops()) {
        if (ov::as_type_ptr<ov::op::util::ReadValueBase>(node) ||
            ov::as_type_ptr<ov::op::util::AssignBase>(node))
            second_paired_op_cnt++;
    }

    return fist_paired_op_cnt == second_paired_op_cnt;
}

bool build_control_dependency(std::shared_ptr<ov::Model> &model) {
    std::map<std::string, std::pair<std::shared_ptr<ov::op::util::ReadValueBase>, std::shared_ptr<ov::op::util::AssignBase>>> dependency_pairs;
    for (auto& node : model->get_ordered_ops()) {
        if (const auto& read_value = ov::as_type_ptr<ov::op::util::ReadValueBase>(node)) {
            dependency_pairs[read_value->get_variable_id()].first = read_value;
        }

        if (const auto& assign = ov::as_type_ptr<ov::op::util::AssignBase>(node)) {
            dependency_pairs[assign->get_variable_id()].second = assign;
        }
    }

    for (auto& pair : dependency_pairs) {
        auto nodes = pair.second;
        if (nodes.first == nullptr || nodes.second == nullptr) {
            return false;
        }
        nodes.second->add_control_dependency(nodes.first);
    }
    return true;
}

}  // namespace util
}  // namespace ov
