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
 * @brief Returns wrapped HW remote context
 *
 * @param context original or proxy context
 *
 * @return HW specific context
 */
const ov::RemoteContext& get_hardware_context(const ov::RemoteContext& context);

}  // namespace proxy
}  // namespace ov
