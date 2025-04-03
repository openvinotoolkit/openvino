// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/node.hpp"

namespace ov {
namespace util {
/**
 * @brief Creates default tensor name for given Node's output.
 * The name format is "node_name:output_port".
 *
 * @param output - Node's output to create name for tensor.
 * @return Default tensor name.
 */
OPENVINO_API std::string make_default_tensor_name(const Output<const Node>& output);
}  // namespace util

namespace op::util {
/**
 * @brief Set name for both node and output tensor. Any other names will be overriden by a given single name
 * @param node - node to rename
 * @param name - new name
 * @param output_port - output port to rename
 */
void OPENVINO_API set_name(ov::Node& node, const std::string& name, size_t output_port = 0);

/**
 * @brief Checks if sources of inputs of two nodes are equal
 * @param lhs - node to check input
 * @param rhs - other node to check input
 * @param input_index - input port index to get the source
 * @return true if sources share same node and output index otherwise false
 */
bool input_sources_are_equal(const std::shared_ptr<Node>& lhs,
                             const std::shared_ptr<Node>& rhs,
                             const size_t& input_index);

}  // namespace op::util
}  // namespace ov
