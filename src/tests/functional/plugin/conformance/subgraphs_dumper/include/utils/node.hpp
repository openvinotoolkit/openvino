// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "op_conformance_utils/meta_info/input_info.hpp"
#include "op_conformance_utils/utils/dynamism.hpp"

#include "openvino/openvino.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/op/constant.hpp"

namespace ov {
namespace util {

template <typename dType>
inline ov::conformance::InputInfo::Range
get_const_ranges(const std::shared_ptr<ov::op::v0::Constant>& node) {
    size_t elements_count = ov::shape_size(node->get_shape());
    if (elements_count == 0) {
        throw std::runtime_error("Impossible to get const ranges! Incorrect const size!");
    }
    const auto& const_values = node->cast_vector<dType>();
    auto max = *std::max_element(const_values.begin(), const_values.end());
    auto min = *std::min_element(const_values.begin(), const_values.end());
    return ov::conformance::InputInfo::Range(static_cast<double>(min), static_cast<double>(max));
}

ov::conformance::InputInfo::Range
get_const_ranges(const std::shared_ptr<ov::op::v0::Constant>& const_node,
                 ov::element::Type elem_type);

std::map<std::string, ov::conformance::InputInfo>
get_input_info_by_node(const std::shared_ptr<ov::Node>& node);

// replace all input node by parameters and constants instead of non input mode types
// if `!is_save_const` replace only by parameters
// if `!is_copy_const_node` do not create new node with constants only as inputs 
std::shared_ptr<ov::Node> clone_node(std::shared_ptr<ov::Node> node,
                                     bool is_save_const = false,
                                     bool is_copy_const_node = false,
                                     std::string node_name = "");


std::shared_ptr<ov::op::v0::Parameter>
convert_const_to_param(const std::shared_ptr<ov::op::v0::Constant>& constant_node);

// all inputs are defined as parameters and contains detailed info in meta
std::shared_ptr<ov::Model>
generate_model_by_node(const std::shared_ptr<ov::Node>& node);

inline std::string get_node_type(const std::shared_ptr<ov::Node>& node) {
    if (ov::util::is_dynamic_node(node)) {
        return "dynamic";
    }
    return "static";
}

// static std::map<std::string, std::string> get_max_ops_versions() {
//     std::map<std::string, std::set<std::string>> unique_ops = ov::test::utils::get_unique_ops();
//     std::map<std::string, std::string> max_ops_versions;

//     for (auto op_info : unique_ops) {
//         std::vector<size_t> available_opsets;
//         for (auto opset : op_info.second) {
//             available_opsets.push_back(std::atoi(opset.c_str()));
//         }
//         size_t max_opset = *std::max_element(available_opsets.begin(), available_opsets.end());

//         max_ops_versions.insert({op_info.first, std::to_string(max_opset)});
//     }
//     return max_ops_versions;
// }

static
std::pair<std::string, std::unordered_map<std::string, std::pair<std::string, std::string>>>
get_last_opset_version_map() {
    std::unordered_map<std::string, std::pair<std::string, std::string>> operation;
    std::string latest_opset;
    for (const auto& opset : ov::get_available_opsets()) {
        auto opset_version = opset.first;
        if (opset_version.length() >= latest_opset.length() &&
            opset_version > latest_opset) {
            latest_opset = opset_version;
        }
        for (const auto& node_type_info : opset.second().get_type_info_set()) {
            auto op_name = node_type_info.name;
            auto op_version = node_type_info.get_version();
            if (operation.count(op_name)) {
                if (opset_version.length() >= operation[op_name].first.length() &&
                    opset_version > operation[op_name].first) {
                    operation[op_name] = {opset_version, op_version};
                }
            } else {
                operation.insert({op_name, {op_version, opset_version}});
            }
        }
    }
    return {latest_opset, operation};
}

inline size_t get_node_priority_by_version(const std::shared_ptr<ov::Node>& node) {
    static std::unordered_map<std::string, std::pair<std::string, std::string>> op_info;
    static std::string latest_opset;
    if (op_info.empty()) {
        std::tie(latest_opset, op_info) = get_last_opset_version_map();
    }

    size_t priority = 1;
    auto op_name = node->get_type_info().name;
    auto op_version = node->get_type_info().get_version();
    auto a = op_info[op_name];
    if (op_info[op_name].second == op_version) {
        if (op_info[op_name].first == latest_opset) {
            priority = 3;
        } else if (op_info[op_name].second == op_version) {
            priority = 2;
        }
    }

    return priority;
}
                                
inline bool is_node_to_skip(const std::shared_ptr<ov::Node>& node) {
    return ov::op::util::is_parameter(node) ||
           ov::op::util::is_constant(node) ||
           ov::op::util::is_output(node);
}

std::string get_node_version(const ov::NodeTypeInfo& node_type_info);

}  // namespace util
}  // namespace ov
