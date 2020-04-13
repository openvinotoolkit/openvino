// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for VPU plugins.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file vpu_plugin_config.hpp
 */

#pragma once

#include <string>

#include "ie_plugin_config.hpp"
#include "myriad_plugin_config.hpp"
#include "hddl_plugin_config.hpp"
#include "ie_api.h"

//
// Common options
//

#define VPU_CONFIG_KEY(name) InferenceEngine::VPUConfigParams::_CONFIG_KEY(VPU_##name)
#define VPU_CONFIG_VALUE(name) InferenceEngine::VPUConfigParams::VPU_##name

#define DECLARE_VPU_CONFIG_KEY(name) DECLARE_CONFIG_KEY(VPU_##name)
#define DECLARE_VPU_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(VPU_##name)

//
// Common metrics
//

#define VPU_METRIC(name) METRIC_KEY(VPU_##name)
#define DECLARE_VPU_METRIC(name, ...)  DECLARE_METRIC_KEY(VPU_##name, __VA_ARGS__)

namespace InferenceEngine {

/**
 * @brief VPU plugin configuration
 */
namespace VPUConfigParams {

//
// Common options
//

/**
 * @brief Turn on HW stages usage (applicable for MyriadX devices only).
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_VPU_CONFIG_KEY(HW_STAGES_OPTIMIZATION);

/**
 * @deprecated Use CONFIG_KEY(LOG_LEVEL) instead.
 * @brief The key to define log level
 */
INFERENCE_ENGINE_DEPRECATED("Use CONFIG_KEY(LOG_LEVEL) instead")
DECLARE_VPU_CONFIG_KEY(LOG_LEVEL);

/**
 * @deprecated
 * @brief The key to define normalization coefficient for the network input.
 * This option should used with be a real number. Example "255.f"
 */
INFERENCE_ENGINE_DEPRECATED("")
DECLARE_VPU_CONFIG_KEY(INPUT_NORM);

/**
 * @deprecated
 * @brief The flag to specify Bias value that is added to each element of the network input.
 * This option should used with be a real number. Example "0.1f"
 */
INFERENCE_ENGINE_DEPRECATED("")
DECLARE_VPU_CONFIG_KEY(INPUT_BIAS);

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_VPU_CONFIG_KEY(PRINT_RECEIVE_TENSOR_TIME);

/**
 * @deprecated The data scaling now works automatically.
 */
INFERENCE_ENGINE_DEPRECATED("The data scaling now works automatically")
DECLARE_VPU_CONFIG_KEY(NETWORK_CONFIG);

/**
 * @deprecated Use InputInfo::setLayout on input data from ICNNNetwork::getInputsInfo() or
 * Data::setLayout on output data from ICNNNetwork::getOutputsInfo()
 * @brief This option allows to to specify input output layouts for network layers.
 * By default, this value set to VPU_CONFIG_VALUE(AUTO) value.
 * Supported values:
 *   VPU_CONFIG_VALUE(AUTO) executable network configured to use optimal layer layout depending on available HW
 *   VPU_CONFIG_VALUE(NCHW) executable network forced to use NCHW input/output layouts
 *   VPU_CONFIG_VALUE(NHWC) executable network forced to use NHWC input/output layouts
 */
INFERENCE_ENGINE_DEPRECATED("Use InputInfo::setLayout on input data from ICNNNetwork::getInputsInfo() or" \
    "Data::setLayout on output data from ICNNNetwork::getOutputsInfo()")
DECLARE_VPU_CONFIG_KEY(COMPUTE_LAYOUT);

/**
 * @deprecated See VPU_CONFIG_KEY(COMPUTE_LAYOUT) deprecation info.
 * @brief Supported keys definition for VPU_CONFIG_KEY(COMPUTE_LAYOUT) option.
 */
INFERENCE_ENGINE_DEPRECATED("See VPU_CONFIG_KEY(COMPUTE_LAYOUT) deprecation info")
DECLARE_VPU_CONFIG_VALUE(AUTO);
INFERENCE_ENGINE_DEPRECATED("See VPU_CONFIG_KEY(COMPUTE_LAYOUT) deprecation info")
DECLARE_VPU_CONFIG_VALUE(NCHW);
INFERENCE_ENGINE_DEPRECATED("See VPU_CONFIG_KEY(COMPUTE_LAYOUT) deprecation info")
DECLARE_VPU_CONFIG_VALUE(NHWC);
INFERENCE_ENGINE_DEPRECATED("See VPU_CONFIG_KEY(COMPUTE_LAYOUT) deprecation info")
DECLARE_VPU_CONFIG_VALUE(NCDHW);
INFERENCE_ENGINE_DEPRECATED("See VPU_CONFIG_KEY(COMPUTE_LAYOUT) deprecation info")
DECLARE_VPU_CONFIG_VALUE(NDHWC);

/**
 * @brief This option allows to pass custom layers binding xml.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported
 */
DECLARE_VPU_CONFIG_KEY(CUSTOM_LAYERS);

/**
 * @brief Ignore statistic in IR by plugin.
 * Plugin could use statistic present in IR in order to try to improve calculations precision.
 * If you don't want statistic to be used enable this option.
 * This option should be used with values: CONFIG_VALUE(YES) or CONFIG_VALUE(NO) (default)
 */
DECLARE_VPU_CONFIG_KEY(IGNORE_IR_STATISTIC);

/**
 * @brief This option allows to specify protocol.
 */
DECLARE_VPU_MYRIAD_CONFIG_KEY(PROTOCOL);

/**
 * @brief Supported keys definition for VPU_MYRIAD_CONFIG_KEY(PROTOCOL) option.
 */
DECLARE_VPU_MYRIAD_CONFIG_VALUE(PCIE);
DECLARE_VPU_MYRIAD_CONFIG_VALUE(USB);

/**
 * @deprecated Use VPU_MYRIAD_CONFIG_KEY(FORCE_RESET) instead.
 */
INFERENCE_ENGINE_DEPRECATED("Use VPU_MYRIAD_CONFIG_KEY(FORCE_RESET) instead")
DECLARE_VPU_CONFIG_KEY(FORCE_RESET);

/**
 * @deprecated Use VPU_MYRIAD_CONFIG_KEY(PLATFORM) instead.
 */
INFERENCE_ENGINE_DEPRECATED("Use VPU_MYRIAD_CONFIG_KEY(PLATFORM) instead")
DECLARE_VPU_CONFIG_KEY(PLATFORM);

/**
 * @brief Supported keys definition for DECLARE_VPU_CONFIG_KEY(PLATFORM) option.
 */
INFERENCE_ENGINE_DEPRECATED("")
DECLARE_VPU_CONFIG_VALUE(2450);
INFERENCE_ENGINE_DEPRECATED("")
DECLARE_VPU_CONFIG_VALUE(2480);

}  // namespace VPUConfigParams

}  // namespace InferenceEngine
