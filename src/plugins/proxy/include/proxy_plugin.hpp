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

const std::shared_ptr<ov::ITensor>& get_hardware_tensor(const std::shared_ptr<ov::ITensor>& tensor);

}  // namespace proxy
}  // namespace ov
