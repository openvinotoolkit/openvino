// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Myriad plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file myriad_config.hpp
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

#include "vpu_config.hpp"

namespace InferenceEngine {

/**
 * @brief The flag to reset stalled devices.
 * This is a plugin scope option and must be used with the plugin's SetConfig method
 * The only possible values are:
 *     CONFIG_VALUE(YES)
 *     CONFIG_VALUE(NO) (default value)
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_ENABLE_FORCE_RESET);

/**
 * @brief This option allows to specify device memory type.
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_DDR_TYPE);

/**
 * @brief Supported keys definition for InferenceEngine::MYRIAD_DDR_TYPE option.
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_DDR_AUTO);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_DDR_MICRON_2GB);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_DDR_SAMSUNG_2GB);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_DDR_HYNIX_2GB);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_DDR_MICRON_1GB);

/**
 * @brief This option allows to specify protocol.
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_PROTOCOL);

/**
 * @brief Supported keys definition for InferenceEngine::MYRIAD_PROTOCOL option.
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_PCIE);
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_USB);

/**
 * @brief Optimize MYRIAD plugin execution to maximize throughput.
 * This option should be used with integer value which is the requested number of streams.
 * The only possible values are:
 *     1
 *     2
 *     3
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_THROUGHPUT_STREAMS);

/**
 * @brief Default key definition for InferenceEngine::MYRIAD_THROUGHPUT_STREAMS option.
 */
INFERENCE_ENGINE_1_0_DEPRECATED DECLARE_VPU_CONFIG(MYRIAD_THROUGHPUT_STREAMS_AUTO);

}  // namespace InferenceEngine
