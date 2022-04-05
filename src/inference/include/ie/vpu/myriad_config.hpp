// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for Myriad plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file myriad_config.hpp
 */

#pragma once

#include "vpu_config.hpp"

namespace InferenceEngine {

/**
 * @brief The flag to reset stalled devices.
 * This is a plugin scope option and must be used with the plugin's SetConfig method
 * The only possible values are:
 *     CONFIG_VALUE(YES)
 *     CONFIG_VALUE(NO) (default value)
 */
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_FORCE_RESET);

/**
 * @brief This option allows to specify device memory type.
 */
DECLARE_VPU_CONFIG(MYRIAD_DDR_TYPE);

/**
 * @brief Supported keys definition for InferenceEngine::MYRIAD_DDR_TYPE option.
 */
DECLARE_VPU_CONFIG(MYRIAD_DDR_AUTO);
DECLARE_VPU_CONFIG(MYRIAD_DDR_MICRON_2GB);
DECLARE_VPU_CONFIG(MYRIAD_DDR_SAMSUNG_2GB);
DECLARE_VPU_CONFIG(MYRIAD_DDR_HYNIX_2GB);
DECLARE_VPU_CONFIG(MYRIAD_DDR_MICRON_1GB);

/**
 * @brief This option allows to specify protocol.
 */
DECLARE_VPU_CONFIG(MYRIAD_PROTOCOL);

/**
 * @brief Supported keys definition for InferenceEngine::MYRIAD_PROTOCOL option.
 */
DECLARE_VPU_CONFIG(MYRIAD_PCIE);
DECLARE_VPU_CONFIG(MYRIAD_USB);

/**
 * @brief Optimize vpu plugin execution to maximize throughput.
 * This option should be used with integer value which is the requested number of streams.
 * The only possible values are:
 *     1
 *     2
 *     3
 */
DECLARE_VPU_CONFIG(MYRIAD_THROUGHPUT_STREAMS);

/**
 * @brief Default key definition for InferenceEngine::MYRIAD_THROUGHPUT_STREAMS option.
 */
DECLARE_VPU_CONFIG(MYRIAD_THROUGHPUT_STREAMS_AUTO);

}  // namespace InferenceEngine
