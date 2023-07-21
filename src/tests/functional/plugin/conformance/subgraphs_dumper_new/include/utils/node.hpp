// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "cache/meta/input_info.hpp"
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
    size_t elements_count = ov::shape_size(node->get_shape());
    const auto& const_values = node->cast_vector<dType>();
    auto max = *std::max_element(const_values.begin(), const_values.end());
    auto min = *std::min_element(const_values.begin(), const_values.end());
    return InputInfo::Range(static_cast<double>(min), static_cast<double>(max));
}

std::map<std::string, InputInfo> get_input_info_by_node(const std::shared_ptr<ov::Node>& node);

// replace all input node by parameters and constants instead of non input mode types
// if `!is_save_const` replace only by parameters
// if `!is_copy_const_node` do not create new node with constants only as inputs 
std::shared_ptr<ov::Node> clone_node(std::shared_ptr<ov::Node> node,
                                     bool is_save_const = false,
                                     bool is_copy_const_node = false,
                                     std::string node_name = "");

// all inputs are defined as parameters and contains detailed info in meta
std::shared_ptr<ov::Model> generate_model_by_node(const std::shared_ptr<ov::Node>& node);

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