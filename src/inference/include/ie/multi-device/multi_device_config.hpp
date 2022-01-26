// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Multi_Device plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods
 *
 * @file multi_device_config.hpp
 */

#pragma once

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief Multi Device plugin configuration
 */
namespace MultiDeviceConfigParams {

/**
 * @def MULTI_CONFIG_KEY(name)
 * @brief A macro which provides a MULTI-mangled name for configuration key with name `name`
 */
#define MULTI_CONFIG_KEY(name) InferenceEngine::MultiDeviceConfigParams::_CONFIG_KEY(MULTI_##name)

#define DECLARE_MULTI_CONFIG_KEY(name)   DECLARE_CONFIG_KEY(MULTI_##name)
#define DECLARE_MULTI_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(MULTI_##name)

/**
 * @brief Device Priorities config option, with comma-separated devices listed in the desired priority
 */
DECLARE_MULTI_CONFIG_KEY(DEVICE_PRIORITIES);

/**
 * @brief network priority config option, the range of value is from 0 to the max integer,
 * when there are multi devices, the value is smaller, the priority is higher,
 * 0 is the highest priority. Auto plugin dispatch the network to device
 * according to priority value. when all devices are free, even if the priority value
 * is not 0, the network will be dispatched to the strongest device.
 */
DECLARE_CONFIG_KEY(AUTO_NETWORK_PRIORITY);
}  // namespace MultiDeviceConfigParams
}  // namespace InferenceEngine
