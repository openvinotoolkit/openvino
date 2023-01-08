// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines common config subset for VPU plugins.
 * Include myriad_config.hpp or hddl_config.hpp directly.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_config.hpp
 */

#pragma once

#include <string>

#include "ie_api.h"
#include "ie_plugin_config.hpp"

#define DECLARE_VPU_CONFIG(name) static constexpr auto name = #name

namespace InferenceEngine {

//
// Common options
//

/**
 * @brief Turn on HW stages usage (applicable for MyriadX devices only).
 * The only possible values are:
 *     CONFIG_VALUE(YES) (default value)
 *     CONFIG_VALUE(NO)
 */
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_HW_ACCELERATION);

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 * The only possible values are:
 *     CONFIG_VALUE(YES)
 *     CONFIG_VALUE(NO) (default value)
 */
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_RECEIVING_TENSOR_TIME);

/**
 * @brief This option allows to pass custom layers binding xml.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported
 */
DECLARE_VPU_CONFIG(MYRIAD_CUSTOM_LAYERS);

}  // namespace InferenceEngine
