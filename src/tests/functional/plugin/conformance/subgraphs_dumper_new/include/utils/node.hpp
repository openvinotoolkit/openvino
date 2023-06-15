// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include <memory>

#include "openvino/openvino.hpp"

namespace ov {
namespace tools {
namespace subgraph_dumper {

inline std::map<std::string, InputInfo> get_input_info_by_node(const std::shared_ptr<ov::Node>& node) {
    std::map<std::string, InputInfo> input_info;
    for (size_t port_id = 0; port_id < node->get_input_size(); ++port_id) {
        InputInfo in_info;
        std::string input_name = node->get_friendly_name() + "_" + std::to_string(port_id);
        if (std::dynamic_pointer_cast<ov::op::v0::Constant>(node->input_value(port_id).get_node_shared_ptr())) {
            in_info.is_const = true;
            auto const_values = 
                std::dynamic_pointer_cast<ov::op::v0::Constant>(node->input_value(port_id).get_node_shared_ptr())->get_vector<double>();
            in_info.ranges.max = *std::max_element(const_values.begin(), const_values.end());
            in_info.ranges.min = *std::min_element(const_values.begin(), const_values.end());
        }
        input_info.insert({input_name, in_info});
    }
    return input_info;
}

// all inputs are defined as parameters and contains detailed info in meta
inline std::shared_ptr<ov::Model> generate_graph_by_node(const std::shared_ptr<ov::Node>& node) {
    ov::ParameterVector params;
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (ov::op::util::is_parameter(node->get_input_node_ptr(i)) ||
            ov::op::util::is_constant(node->get_input_node_ptr(i))) {
            auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(
                    node->get_input_node_shared_ptr(i));
            params.push_back(param);
        }
    }
    ov::ResultVector results;
    for (auto &out : node->outputs()) {
        results.push_back(std::make_shared<ov::op::v0::Result>(out));
    }
    return std::make_shared<ov::Model>(results, params);
}

inline std::string get_node_type(const std::shared_ptr<ov::Node>& node) {
    for (size_t i = 0; i < node->get_input_size(); ++i) {
        if (node->get_input_partial_shape(i).is_dynamic()) {
            return "dynamic";
        }
    }
    for (size_t i = 0; i < node->get_output_size(); ++i) {
        if (node->get_output_partial_shape(i).is_dynamic()) {
            return "dynamic";
        }
    }
    return "static";
}

}  // namespace subgraph_dumper
}  // namespace tools
}  // namespace ov