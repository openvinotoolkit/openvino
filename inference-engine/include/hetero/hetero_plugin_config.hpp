// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Heterogeneous plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file hetero_plugin_config.hpp
 */

#pragma once

#include <string>
#include <vector>

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief Heterogeneous plugin configuration
 */
namespace HeteroConfigParams {

/**
 * @def HETERO_CONFIG_KEY(name)
 * @brief Shortcut for defining HETERO configuration keys
 */
#define HETERO_CONFIG_KEY(name) InferenceEngine::HeteroConfigParams::_CONFIG_KEY(HETERO_##name)
#define DECLARE_HETERO_CONFIG_KEY(name) DECLARE_CONFIG_KEY(HETERO_##name)
#define DECLARE_HETERO_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(HETERO_##name)

/**
 * @brief The key for enabling of dumping the topology with details of layers and details how
 * this network would be executed on different devices to the disk in GraphViz format.
 * This option should be used with values: CONFIG_VALUE(NO) (default) or CONFIG_VALUE(YES)
 */
DECLARE_HETERO_CONFIG_KEY(DUMP_GRAPH_DOT);

/**
 * @deprecated Use DLIA_CONFIG_KEY(DUMP_SUPPORTED_LAYERS_INFORMATION) FPGA configuration boolean key instead
 * @brief The bool key to define whether information messages with a reason are printed in case the layer is unsupported by DLA
 */
INFERENCE_ENGINE_DEPRECATED
DECLARE_HETERO_CONFIG_KEY(DUMP_DLA_MESSAGES);

}  // namespace HeteroConfigParams
}  // namespace InferenceEngine
