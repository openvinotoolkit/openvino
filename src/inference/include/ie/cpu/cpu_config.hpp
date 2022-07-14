// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for CPU plugin
 *        To use in SetConfig() method of plugins
 *
 * @file cpu_config.hpp
 */
#pragma once

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief CPU plugin configuration
 */
namespace CPUConfigParams {

/**
 * @brief shortcut for defining configuration keys
 */
#define CPU_CONFIG_KEY(name)           InferenceEngine::CPUConfigParams::_CONFIG_KEY(CPU_##name)
#define DECLARE_CPU_CONFIG_KEY(name)   DECLARE_CONFIG_KEY(CPU_##name)
#define DECLARE_CPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(CPU_##name)

/**
 * @brief The name for defining if denormals is optimized on CPU whenever it is possible
 *
 * This option let cpu plugin know if denormals is optimized where it see performance benefits from
 * getting ride of denormals computation.
 * Such option do not guarantee accuracy of the model, the accuracy in this mode should be
 * verified separately by the user and basing on performance and accuracy results it should be
 * user's decision to use this option or not to use
 * It is passed to Core::SetConfig(), this option should be used with values:
 * PluginConfigParams::YES or PluginConfigParams::NO
 * If not set explicitly, the behavior is kept in runtime enviroment where compile_model is called.
 */
DECLARE_CPU_CONFIG_KEY(DENORMALS_OPTIMIZATION);

}  // namespace CPUConfigParams
}  // namespace InferenceEngine
