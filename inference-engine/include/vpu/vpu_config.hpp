// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_config.hpp
 */

#pragma once

#include "details/common_utils.hpp"
#include "details/myriad_config.hpp"
#include "details/hddl_config.hpp"
#include "details/hddl_metrics.hpp"

#include "ie_plugin_config.hpp"
#include "ie_api.h"

#include <string>

namespace InferenceEngine {

//
// Common options
//

/**
 * @brief Turn on HW stages usage (applicable for MyriadX devices only).
 * This option should be used with values: CONFIG_VALUE(YES) (default) or CONFIG_VALUE(NO)
 */
DECLARE_MYRIAD_CONFIG_KEY(ENABLE_HW_ACCELERATION);

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_MYRIAD_CONFIG_KEY(ENABLE_RECEIVING_TENSOR_TIME);

/**
 * @brief This option allows to pass custom layers binding xml.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported
 */
DECLARE_MYRIAD_CONFIG_KEY(CUSTOM_LAYERS);

/**
 * @brief Optimize vpu plugin execution to maximize throughput.
 * This option should be used with integer value which is the requested number of streams.
 */
DECLARE_MYRIAD_CONFIG_KEY(THROUGHPUT_STREAMS);

}  // namespace InferenceEngine
