// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "cpu_opset/common/op/fully_connected.hpp"
#include "transformations/rt_info/dequantization_node.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace intel_cpu {

bool has_matmul_with_compressed_weights(const std::shared_ptr<const ov::Model>& model) {
    bool has_decompression_multiply = false;
    auto is_decompression_multiply = [&](ov::Node* node) {
        if (auto multiply = ov::as_type<ov::op::v1::Multiply>(node)) {
            if (ov::is_dequantization_node(multiply->shared_from_this()))
                has_decompression_multiply = true;
        }
    };

    for (const auto& op : model->get_ops()) {
        if (!ov::is_type<ov::op::v0::MatMul>(op) && !ov::is_type<FullyConnectedNode>(op))
            continue;

        if (!op->get_input_element_type(0).is_real())
            continue;

        auto weights = op->input_value(1);
        if (!ov::op::util::is_on_constant_path(weights))
            continue;

        std::unordered_set<Node*> visited;
        ov::op::util::visit_constant_path(weights.get_node(), visited, is_decompression_multiply);

        if (has_decompression_multiply)
            return true;
    }
    return false;
}

// Check specific pattern:
// Constant
//     |
// Convert  Constant
//     \    /
//   Subtract   Constant
//       \     /
//       Multiply
//          |
//     Convert input Constant
//        \     /   /
//         \   /   /
//          Gather
bool is_gather_with_compressed_weights(const std::shared_ptr<const ov::Node>& node) {
    if (!ov::is_type<ov::opset8::Gather>(node)) {
        return false;
    }
    if (node->get_input_size() != 3) {
        return false;
    }

    auto is_constant_with_2d = [](const ov::Node* node) {
        const ov::Node* const_node = ov::is_type<ov::opset1::Convert>(node) ? node->get_input_node_ptr(0) : node;

        if (ov::is_type<ov::op::v0::Constant>(const_node) && const_node->get_input_size() == 0) {
            auto cur_shape = const_node->get_output_shape(0);
            if (cur_shape.size() == 2 && cur_shape[1] == 1u) {
                return true;
            }
        }
        return false;
    };

    // Check axis
    auto axis = node->get_input_node_ptr(2);
    auto axisPtr = ov::as_type<ov::op::v0::Constant>(axis);
    if (!axisPtr) {
        return false;
    }
    int32_t axis_const = axisPtr->cast_vector<int32_t>()[0];
    if (axis_const != 0) {
        return false;
    }

    // Check weights
    ov::Node* multiply = nullptr;
    auto multiply_convert = node->get_input_node_ptr(0);
    if (ov::is_type<ov::op::v0::Convert>(multiply_convert)) {
        multiply = multiply_convert->get_input_node_ptr(0);
    } else {
        multiply = node->get_input_node_ptr(0);
    }
    if (!ov::is_type<ov::op::v1::Multiply>(multiply)) {
        return false;
    }
    if (!is_constant_with_2d(multiply->get_input_node_ptr(1))) {
        return false;
    }

    auto subtract = multiply->get_input_node_ptr(0);
    if (!ov::is_type<ov::op::v1::Subtract>(subtract)) {
        return false;
    }
    if (!is_constant_with_2d(subtract->get_input_node_ptr(1))) {
        return false;
    }

    auto weights_convert = subtract->get_input_node_ptr(0);
    if (!ov::is_type<ov::op::v0::Convert>(weights_convert)) {
        return false;
    }

    auto weights = weights_convert->get_input_node_ptr(0);
    auto weights_ptr = ov::as_type<ov::op::v0::Constant>(weights);
    if (!weights_ptr) {
        return false;
    }
    auto weights_shape = weights_ptr->get_output_shape(0);
    if (weights_shape.size() != 2u) {
        return false;
    }
    return true;
}

}   // namespace intel_cpu
}   // namespace ov
