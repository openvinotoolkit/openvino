// Copyright (C) 2018-2022 Intel Corporation
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

namespace TemplateMetrics {

/**
 * @def TEMPLATE_METRIC_VALUE(name)
 * @brief Shortcut for defining Template metric values
 */
#define TEMPLATE_METRIC_VALUE(name) InferenceEngine::TemplateMetrics::name
#define DECLARE_TEMPLATE_METRIC_VALUE(name) static constexpr auto name = #name

// ! [public_header:metrics]
/**
 * @brief Defines whether current Template device instance supports hardware blocks for fast convolution computations.
 */
DECLARE_TEMPLATE_METRIC_VALUE(HARDWARE_CONVOLUTION);
// ! [public_header:metrics]

}  // namespace TemplateMetrics

namespace TemplateConfigParams {

/**
 * @def TEMPLATE_CONFIG_KEY(name)
 * @brief Shortcut for defining Template device configuration keys
 */
#define TEMPLATE_CONFIG_KEY(name) InferenceEngine::TemplateConfigParams::_CONFIG_KEY(TEMPLATE_##name)

#define DECLARE_TEMPLATE_CONFIG_KEY(name) DECLARE_CONFIG_KEY(TEMPLATE_##name)
#define DECLARE_TEMPLATE_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(TEMPLATE_##name)


/**
 * @brief Defines the number of throutput streams used by TEMPLATE plugin.
 */
DECLARE_TEMPLATE_CONFIG_KEY(THROUGHPUT_STREAMS);


}  // namespace TemplateConfigParams
}  // namespace InferenceEngine
