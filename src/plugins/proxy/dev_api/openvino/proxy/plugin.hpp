// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"

namespace ov {
namespace proxy {

/**
 * @brief Creates a new instance of Proxy plugin
 *
 * @param plugin shared pointer to the plugin
 */
void create_plugin(std::shared_ptr<ov::IPlugin>& plugin);

/**
 * @brief Get wrapped remote context
 *
 * @param context Remote context
 *
 * @return Original remote context
 */
ov::SoPtr<ov::IRemoteContext> get_hardware_context(const ov::SoPtr<ov::IRemoteContext>& context);

/**
 * @brief Get wrapped remote tensor
 *
 * @param tensor Remote tensor
 *
 * @return Original remote tensor
 */
ov::SoPtr<ov::ITensor> get_hardware_tensor(const ov::SoPtr<ov::ITensor>& tensor, bool unwrap = false);

}  // namespace proxy
}  // namespace ov
