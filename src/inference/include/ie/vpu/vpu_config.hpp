// Copyright (C) 2018-2023 Intel Corporation
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

#if !defined(IN_OV_COMPONENT) && !defined(IE_LEGACY_HEADER_INCLUDED)
#    define IE_LEGACY_HEADER_INCLUDED
#    ifdef _MSC_VER
#        pragma message( \
            "The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    else
#        warning("The Inference Engine API is deprecated and will be removed in the 2024.0 release. For instructions on transitioning to the new API, please refer to https://docs.openvino.ai/latest/openvino_2_0_transition_guide.html")
#    endif
#endif

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
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_ENABLE_HW_ACCELERATION);

/**
 * @brief The flag for adding to the profiling information the time of obtaining a tensor.
 * The only possible values are:
 *     CONFIG_VALUE(YES)
 *     CONFIG_VALUE(NO) (default value)
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_ENABLE_RECEIVING_TENSOR_TIME);

/**
 * @brief This option allows to pass custom layers binding xml.
 * If layer is present in such an xml, it would be used during inference even if the layer is natively supported
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_CUSTOM_LAYERS);

}  // namespace InferenceEngine
