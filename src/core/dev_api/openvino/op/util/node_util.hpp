// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace op {
namespace util {
/**
 * @brief Set name for both node and output tensor. Any other names will be overriden by a given single name
 * @param node - node to rename
 * @param name - new name
 * @param output_port - output port to rename
 */
void OPENVINO_API set_name(ov::Node& node, const std::string& name, size_t output_port = 0);
}  // namespace util
}  // namespace op
}  // namespace ov
