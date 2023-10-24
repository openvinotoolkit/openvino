// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "cache/meta/input_info.hpp"
#include "functional_test_utils/node_utils.hpp"
#include "functional_test_utils/summary/op_info.hpp"
#include "openvino/openvino.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "openvino/op/util/op_types.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

template <typename dType>
inline InputInfo::Range get_const_ranges(const std::shared_ptr<ov::op::v0::Constant>& node) {
    size_t elements_count = ov::shape_size(node->get_output_partial_shape(0).to_shape());
    const auto& const_values = node->cast_vector<dType>();
    auto max = *std::max_element(const_values.begin(), const_values.end());
    auto min = *std::min_element(const_values.begin(), const_values.end());
    return InputInfo::Range(static_cast<double>(min), static_cast<double>(max));
}

InputInfo::Range get_const_ranges(const std::shared_ptr<ov::op::v0::Constant>& const_node,
                                  ov::element::Type elem_type);

std::map<std::string, InputInfo> get_input_info_by_node(const std::shared_ptr<ov::Node>& node);

// replace all input node by parameters and constants instead of non input mode types
// if `!is_save_const` replace only by parameters
// if `!is_copy_const_node` do not create new node with constants only as inputs 
std::shared_ptr<ov::Node> clone_node(std::shared_ptr<ov::Node> node,
                                     bool is_save_const = false,
                                     bool is_copy_const_node = false,
                                     std::string node_name = "");


std::shared_ptr<ov::op::v0::Parameter> convert_const_to_param(const std::shared_ptr<ov::op::v0::Constant>& constant_node);

// all inputs are defined as parameters and contains detailed info in meta
std::shared_ptr<ov::Model> generate_model_by_node(const std::shared_ptr<ov::Node>& node);

inline bool is_dynamic_node(const std::shared_ptr<ov::Node>& node) {
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (node->get_input_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return true;
        }
    }
    return false;
}

inline std::string get_node_type(const std::shared_ptr<ov::Node>& node) {
    if (is_dynamic_node(node)) {
        return "dynamic";
    }
    return "static";
}

static std::map<std::string, std::string> get_max_ops_versions() {
    std::map<std::string, std::set<std::string>> unique_ops = ov::test::utils::get_unique_ops();
    std::map<std::string, std::string> max_ops_versions;

    for (auto op_info : unique_ops) {
        std::vector<size_t> available_opsets;
        for (auto opset : op_info.second) {
            available_opsets.push_back(std::atoi(opset.c_str()));
        }
        size_t max_opset = *std::max_element(available_opsets.begin(), available_opsets.end());

        max_ops_versions.insert({op_info.first, std::to_string(max_opset)});
    }
    return max_ops_versions;
}

static std::map<std::string, std::string> get_last_opset_version_map() {
    auto opset_map = ov::get_available_opsets();
    std::map<std::string, std::string> res;
    std::string opset_name = std::prev(opset_map.end())->first;
    const ov::OpSet& opset = std::prev(opset_map.end())->second();
    for (const auto& op : opset.get_type_info_set()) {
        res[op.name] = ov::test::utils::get_op_version(op.get_version());
    }

    return res;
}

inline size_t get_node_priority_by_version(const std::shared_ptr<ov::Node>& node) {
    // { op_name, max_opset_version }
    static std::map<std::string, std::string> max_ops_versions = ov::tools::subgraph_dumper::get_max_ops_versions();
    // { op_name, op_version_in_last_opset }
    static std::map<std::string, std::string> last_opset_versions_map = ov::tools::subgraph_dumper::get_last_opset_version_map();

    size_t priority = 1;
    auto type_info = node->get_type_info();
    if (max_ops_versions.count(type_info.name)) {
        std::string version_id = ov::test::utils::get_op_version(type_info.version_id);
        if (version_id == max_ops_versions[type_info.name]) {
            priority = 2;
            if (version_id == last_opset_versions_map[type_info.name]) {
                priority = 3;
            }
        }
    }

    return priority;
}
                                
inline bool is_node_to_skip(const std::shared_ptr<ov::Node>& node) {
    return ov::op::util::is_parameter(node) ||
           ov::op::util::is_constant(node) ||
           ov::op::util::is_output(node);
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov
