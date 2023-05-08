// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/iplugin.hpp"

namespace ov {
namespace proxy {

/**
 * @brief Creates a new instance of Proxy plugin
 *
 * @param plugin shared pointer to the plugin
 */
void create_plugin(std::shared_ptr<ov::IPlugin>& plugin);

/**
 * @brief Restore one fallback order from orders of different plugins
 *
 * @param generated_order the orders from different plugins in format "A->B,C->B,D->B->E"
 *
 * @return the common fallback order (for example: A->C->D->B->E)
 */
std::string restore_order(const std::string& generated_order);

}  // namespace proxy
}  // namespace ov
