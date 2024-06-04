// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/rt_info/constant_node.hpp"

#include "openvino/core/type.hpp"
#include "openvino/op/inverse.hpp"
#include "openvino/op/multinomial.hpp"
#include "openvino/op/random_uniform.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/op_types.hpp"

inline bool is_node_constant(const std::shared_ptr<ov::Node>& node) {
    if (ov::is_type<ov::op::v8::RandomUniform>(node) || ov::is_type<ov::op::v13::Multinomial>(node) ||
        ov::is_type<ov::op::v14::Inverse>(node)) {
        return false;
    } else if (ov::op::util::is_constant(node) || ov::is_type<ov::op::v0::ShapeOf>(node) ||
               ov::is_type<ov::op::v3::ShapeOf>(node)) {
        return true;
    } else {
        const auto in_size = node->input_values().size();
        size_t in_idx = 0;
        while (in_idx < in_size) {
            if (!is_marked_as_constant_node(node->get_input_node_shared_ptr(in_idx))) {
                return false;
            }
        }
        return true;
    }
}

void ov::mark_is_constant_node(const std::shared_ptr<Node>& node) {
    if (is_node_constant(node)) {
        auto& rt_info = node->get_rt_info();
        rt_info[ConstantNode::get_type_info_static()] = ConstantNode();
    }
}

void ov::mark_is_constant_node(const Output<Node>& node) {
    ov::mark_is_constant_node(node.get_node_shared_ptr());
}

bool ov::is_marked_as_constant_node(const std::shared_ptr<Node>& node) {
    const auto& rt_info = node->get_rt_info();
    return rt_info.find(ConstantNode::get_type_info_static()) != rt_info.end();
}

bool ov::is_marked_as_constant_node(const Output<Node>& node) {
    return ov::is_marked_as_constant_node(node.get_node_shared_ptr());
}
