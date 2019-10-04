// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header for properties that are passed from IE to plguins
 *        or from one plugin to another
 * @file ie_internal_plugin_config.hpp
 */
#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace InferenceEngine {

namespace InternalPluginConfigParams {

/**
* @brief shortcut for defining internal configuration keys
*/
#define IE_INTERNAL_CONFIG_KEY(name) InferenceEngine::InternalPluginConfigParams::_IE_INTERNAL_CONFIG_KEY(name)
#define _IE_INTERNAL_CONFIG_KEY(name) KEY_##name
#define DECLARE_IE_INTERNAL_CONFIG_KEY(name) static constexpr auto _IE_INTERNAL_CONFIG_KEY(name) = #name

/**
 * @brief This key should be used to mark input executable subnetworks
 */
DECLARE_IE_INTERNAL_CONFIG_KEY(SUBNETWORK_WITH_NETWORK_INPUTS);

}  // namespace InternalPluginConfigParams
}  // namespace InferenceEngine
