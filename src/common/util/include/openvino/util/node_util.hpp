// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node.hpp"

namespace ov {
namespace util {
/**
 * @brief Set name for both node and output tensor. Any other names will be overriden by a given single name
 * @param node - node to rename
 * @param name - new name
 * @param output_idx - idx of output to rename
 * @return - renamed node
 */
std::shared_ptr<ov::Node> set_name(std::shared_ptr<ov::Node> node, const std::string& name, size_t output_idx = 0);

// Templated method that has the same effect as not templated `set_name` but saves Op type for convenient calls chaining
template <typename T>
std::shared_ptr<T> set_name(std::shared_ptr<T> node, const std::string& name, size_t output_idx = 0) {
    set_name(std::dynamic_pointer_cast<ov::Node>(node), name, output_idx);
    return node;
}
}  // namespace util
}  // namespace ov
