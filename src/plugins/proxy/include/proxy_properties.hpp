// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace proxy {

/**
 * @brief Property allows to configure the fallback priorities under the proxy plugin
 * Vector of string. String has the next format: <first_device>-><fallback_device>
 */
// static constexpr Property<std::vector<std::string>, PropertyMutability::RW> fallback{"PROXY_FALLBACK_PRIORITIES"};

/**
 * @brief Property allows to configure the low level device priorities.
 * Vector of string. String has the next format: <device_name>:<device_priority>
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RW> priorities{"PROXY_DEVICE_PRIORITIES"};

/**
 * @brief Property allows to configure the list of low level devices under the alias
 * Vector of string. String has the next format: <device_name>
 */
static constexpr Property<std::vector<std::string>, PropertyMutability::RW> alias_for{"PROXY_ALIAS_FOR"};

}  // namespace proxy
}  // namespace ov

