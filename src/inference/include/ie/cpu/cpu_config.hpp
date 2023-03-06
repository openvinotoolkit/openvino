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
 *  - UNDEFINED: default setting may vary by platform and performance hints
 *  - ALL_CORE: all processors can be used. If hyper threading is enabled, both processors ofoneperformance-corewill be
 *              used.
 *  - PHY_CORE_ONLY: only one processor can be used per CPU core even with hyper threading enabled.
 *  - P_CORE_ONLY: only processors of performance-cores can be used. If hyper threading is enabled, both processors of
 *                 one performance-core will be used.
 *  - E_CORE_ONLY: only processors of efficient-cores can be used.
 *  - PHY_P_CORE_ONLY: only one processor can be used per performance-core even with hyper threading enabled.
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
