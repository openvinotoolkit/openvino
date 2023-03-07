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
 * This option lets CPU plugin determine whether denormals are optimized where it expects
 * performance benefits from getting rid of denormals computation.
 * Such option does not guarantee accuracy of the model, the accuracy in this mode should be
 * verified separately by the user. Basing on performance and accuracy results, it should be
 * user's decision to use this option or not.
 * It is passed to Core::SetConfig(), this option should be used with values:
 * PluginConfigParams::YES or PluginConfigParams::NO
 * If not set explicitly, the behavior is kept in runtime enviroment where compile_model is called.
 */
DECLARE_CPU_CONFIG_KEY(DENORMALS_OPTIMIZATION);

DECLARE_CPU_CONFIG_KEY(SPARSE_WEIGHTS_DECOMPRESSION_RATE);

/**
 * @brief The name for defining processor type used for CPU inference.
 *
 *  - UNDEFINED:       Default setting. All processors can be used on one socket platform. And only processors of
 *                     physical cores can be used on two socket platform.
 *  - ALL_CORE:        All processors can be used. If hyper threading is enabled, both processor of one performance-core
 *                     can be used.
 *  - PHY_CORE_ONLY:   Only processors of physical cores can be used. If hyper threading is enabled, only one processor
 *                     of one performance-core can be used.
 *  - P_CORE_ONLY:     Only processors of performance-cores can be used. If hyper threading is enabled, both processor
 *                     of one performance-core can be used.
 *  - E_CORE_ONLY:     Only processors of efficient-cores can be used.
 *  - PHY_P_CORE_ONLY: Only processors of physical performance-cores can be used. If hyper threading is enabled, only
 *                     one processor of one performance-core can be used.
 */
DECLARE_CPU_CONFIG_KEY(PROCESSOR_TYPE);
DECLARE_CPU_CONFIG_VALUE(UNDEFINED);
DECLARE_CPU_CONFIG_VALUE(ALL_CORE);
DECLARE_CPU_CONFIG_VALUE(PHY_CORE_ONLY);
DECLARE_CPU_CONFIG_VALUE(P_CORE_ONLY);
DECLARE_CPU_CONFIG_VALUE(E_CORE_ONLY);
DECLARE_CPU_CONFIG_VALUE(PHY_P_CORE_ONLY);

}  // namespace CPUConfigParams
}  // namespace InferenceEngine
