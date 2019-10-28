// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for DLIA plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file dlia_config.hpp
 */

#pragma once

#include <string>
#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief DLIA plugin metrics
 */
namespace DliaMetrics {

/**
 * @def DLIA_METRIC_VALUE(name)
 * @brief Shortcut for defining FPGA metric values
 */
#define DLIA_METRIC_VALUE(name) InferenceEngine::DliaMetrics::name
#define DECLARE_DLIA_METRIC_VALUE(name) static constexpr auto name = #name

/**
 * @brief FP11 optimization capability. It's specific for FPGA device which can perform computations in FP11 data type.
 */
DECLARE_DLIA_METRIC_VALUE(FP11);

/**
 * @brief Input Streaming capability. It's specific for FPGA device which can perform input streaming
 */
DECLARE_DLIA_METRIC_VALUE(INPUT_STREAMING);

}  // namespace DliaMetrics

/**
 * @brief DLIA plugin configuration
 */
namespace DLIAConfigParams {

/**
 * @def DLIA_CONFIG_KEY(name)
 * @brief Shortcut for defining FPGA configuration keys
 */
#define DLIA_CONFIG_KEY(name) InferenceEngine::DLIAConfigParams::_CONFIG_KEY(DLIA_##name)

#define DECLARE_DLIA_CONFIG_KEY(name) DECLARE_CONFIG_KEY(DLIA_##name)
#define DECLARE_DLIA_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(DLIA_##name)

/**
 * @brief The key to define the type of transformations for DLIA inputs and outputs.
 * DLIA use custom data layout for input and output blobs. IE DLIA Plugin provides custom
 * optimized version of transformation functions that do not use OpenMP and much more faster
 * than native DLIA functions. Values: "NO" - optimized plugin transformations
 * are used, "YES" - native DLIA transformations are used.
 */
DECLARE_DLIA_CONFIG_KEY(IO_TRANSFORMATIONS_NATIVE);

/**
 * @brief The key to define path to DLA bitstreams architectures folder
 */
DECLARE_DLIA_CONFIG_KEY(ARCH_ROOT_DIR);

/**
 * @brief The bool key to define whether theoretical performance estimation should be performed.
 * If true, the estimated performance is dumped via performance counters as "FPGA theoretical execute time"
 */
DECLARE_DLIA_CONFIG_KEY(PERF_ESTIMATION);

// TODO: Temporarily adding dlia config to test streaming feature
// Values - "YES" or "NO"
DECLARE_DLIA_CONFIG_KEY(ENABLE_STREAMING);

/**
 * @brief The bool key to define whether information messages with a reason are printed in case the layer is unsupported by DLA
 */
DECLARE_DLIA_CONFIG_KEY(DUMP_SUPPORTED_LAYERS_INFORMATION);

}  // namespace DLIAConfigParams
}  // namespace InferenceEngine
