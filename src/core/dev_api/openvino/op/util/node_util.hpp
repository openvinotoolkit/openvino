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
}  // namespace op::util
}  // namespace ov
