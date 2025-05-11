// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace proxy {

// Proxy plugin configuration properties
namespace configuration {
/**
 * @brief Read-write property to set alias for hardware plugin
 * value type: string Alias name for the set of plugins
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string, PropertyMutability::RW> alias{"PROXY_CONFIGURATION_ALIAS"};

/**
 * @brief Read-write property to set devices priority in alias
 * This property allows to configure the order of devices from different low-level plugins under the proxy
 * value type: int32_t lower value means the higher priority
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<int32_t, PropertyMutability::RW> priority{"PROXY_CONFIGURATION_PRIORITY"};

/**
 * @brief Read-write property to set the fallback to other HW plugin
 * value type: string the name of hardware plugin for fallback
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string, PropertyMutability::RW> fallback{"PROXY_CONFIGURATION_FALLBACK"};

/**
 * @brief Read-write property to set internal name if proxy hides plugin under the plugin name.
 * value type: string the internal name for hardware plugin
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::string, PropertyMutability::RW> internal_name{"PROXY_CONFIGURATION_INTERNAL_NAME"};

}  // namespace configuration

/**
 * @brief Property allows to configure the low level device priorities.
 * Vector of string. String has the next format: <device_name>:<device_priority>
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RW> device_priorities{
    "PROXY_DEVICE_PRIORITIES"};

/**
 * @brief Property allows to configure the list of low level devices under the alias
 * Vector of string. String has the next format: <device_name>
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RW> alias_for{"PROXY_ALIAS_FOR"};

}  // namespace proxy
}  // namespace ov
