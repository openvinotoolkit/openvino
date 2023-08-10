// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Heterogeneous plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file hetero_plugin_config.hpp
 */

#pragma once

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

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
#define HETERO_CONFIG_KEY(name)         InferenceEngine::HeteroConfigParams::_CONFIG_KEY(HETERO_##name)
#define DECLARE_HETERO_CONFIG_KEY(name) DECLARE_CONFIG_KEY(HETERO_##name)

/**
 * @brief The key for enabling of dumping the topology with details of layers and details how
 * this network would be executed on different devices to the disk in GraphViz format.
 * This option should be used with values: CONFIG_VALUE(NO) (default) or CONFIG_VALUE(YES)
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_HETERO_CONFIG_KEY(DUMP_GRAPH_DOT);

}  // namespace HeteroConfigParams
}  // namespace InferenceEngine
