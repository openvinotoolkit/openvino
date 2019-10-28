// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Multi_Device plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods
 *
 * @file multi_device_config.hpp
 */

#pragma once

#include <string>
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

#define DECLARE_MULTI_CONFIG_KEY(name) DECLARE_CONFIG_KEY(MULTI_##name)
#define DECLARE_MULTI_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(MULTI_##name)

/**
 * @brief Device Priorities config option, with comma-separated devices listed in the desired priority
 */
DECLARE_MULTI_CONFIG_KEY(DEVICE_PRIORITIES);

}  // namespace MultiDeviceConfigParams
}  // namespace InferenceEngine
