// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include <vpu/myriad_config.hpp>
#include <vpu/vpu_plugin_config.hpp>

namespace InferenceEngine {

//
// Compilation options
//

DECLARE_VPU_CONFIG(MYRIAD_NUMBER_OF_SHAVES);
DECLARE_VPU_CONFIG(MYRIAD_NUMBER_OF_CMX_SLICES);
DECLARE_VPU_CONFIG(MYRIAD_TILING_CMX_LIMIT_KB);

DECLARE_VPU_CONFIG(MYRIAD_TENSOR_STRIDES);

DECLARE_VPU_CONFIG(MYRIAD_IR_WITH_SCALES_DIRECTORY);
DECLARE_VPU_CONFIG(MYRIAD_DETECT_NETWORK_BATCH);
DECLARE_VPU_CONFIG(MYRIAD_COPY_OPTIMIZATION);
DECLARE_VPU_CONFIG(MYRIAD_HW_INJECT_STAGES);
DECLARE_VPU_CONFIG(MYRIAD_HW_POOL_CONV_MERGE);
DECLARE_VPU_CONFIG(MYRIAD_PACK_DATA_IN_CMX);
DECLARE_VPU_CONFIG(MYRIAD_HW_DILATION);
DECLARE_VPU_CONFIG(MYRIAD_HW_EXTRA_SPLIT);

DECLARE_VPU_CONFIG(MYRIAD_PERF_REPORT_MODE);
DECLARE_VPU_CONFIG(MYRIAD_PER_LAYER);
DECLARE_VPU_CONFIG(MYRIAD_PER_STAGE);

DECLARE_VPU_CONFIG(MYRIAD_ENABLE_MEMORY_TYPES_ANNOTATION);
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_EARLY_ELTWISE_RELU_FUSION);

/**
 * @brief Used to disable analyzeWeightableLayers pass in cases where
 * weights scaling leads to poor accuracy. Default = "YES"
 */
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_WEIGHTS_ANALYSIS);

/**
 * @brief MyriadPlugin uses heuristic algorithm to avoid accuracy degradations.
 * This algorithm tries to find the preprocessing at the beginning of the model to adjust its parameters.
 * This option should be set to "NO" if preprocessing is not a part of the model and performed separately
 * in order to avoid accuracy degradation.
 * Default is "YES"
 */
DECLARE_VPU_CONFIG(MYRIAD_CHECK_PREPROCESSING_INSIDE_MODEL);

/**
 * @brief Used to enable reshapeBeforeConvTiling pass in cases where
 * user have reshape parameter "alt_width" in IR.
 * Default is "NO".
 */
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_CUSTOM_RESHAPE_PARAM);

/**
 * @brief Default key definition for InferenceEngine::MYRIAD_NUMBER_OF_SHAVES option.
 */
DECLARE_VPU_CONFIG(MYRIAD_NUMBER_OF_SHAVES_AUTO);

/**
 * @brief Default key definition for InferenceEngine::MYRIAD_NUMBER_OF_CMX_SLICES option.
 */
DECLARE_VPU_CONFIG(MYRIAD_NUMBER_OF_CMX_SLICES_AUTO);

/**
 * @brief Default key definition for InferenceEngine::MYRIAD_TILING_CMX_LIMIT_KB option.
 */
DECLARE_VPU_CONFIG(MYRIAD_TILING_CMX_LIMIT_KB_AUTO);

/**
 * @brief Default key definition for InferenceEngine::MYRIAD_HW_INJECT_STAGES option.
 */
DECLARE_VPU_CONFIG(MYRIAD_HW_INJECT_STAGES_AUTO);

//
// Debug options
//

DECLARE_VPU_CONFIG(MYRIAD_HW_BLACK_LIST);

DECLARE_VPU_CONFIG(MYRIAD_NONE_LAYERS);
DECLARE_VPU_CONFIG(MYRIAD_IGNORE_UNKNOWN_LAYERS);

DECLARE_VPU_CONFIG(MYRIAD_DUMP_INTERNAL_GRAPH_FILE_NAME);
DECLARE_VPU_CONFIG(MYRIAD_DUMP_ALL_PASSES_DIRECTORY);
DECLARE_VPU_CONFIG(MYRIAD_DUMP_ALL_PASSES);

/**
 * @brief Used to disable reorder passes in tests to be able to precisely set
 * desired layout on every stage.
 */
DECLARE_VPU_CONFIG(MYRIAD_DISABLE_REORDER);

/**
 * @brief Used to disable convert stages in tests to be able to insert
 * convert layer with desired precision.
 */
DECLARE_VPU_CONFIG(MYRIAD_DISABLE_CONVERT_STAGES);

/**
 * @brief Used to disable permute merging pass (with setting "NO") in tests to check it preserves behaviour. Default = "YES"
 */
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_PERMUTE_MERGING);

DECLARE_VPU_CONFIG(MYRIAD_ENABLE_REPL_WITH_SCRELU);

/**
 * @brief Used to enable Tensor Iterator unrolling to get a reference for Tensor Iterator per-layer tests.
 * Default is "NO".
 */
DECLARE_VPU_CONFIG(MYRIAD_ENABLE_TENSOR_ITERATOR_UNROLLING);

/**
 * @brief Used to guarantee Tensor Iterator layer will remain in the network regardless of possible performance transformation.
 * Example of transformation: combining to RNN sequence. Needed for Tensor Iterator per-layer tests.
 * Default is "NO".
 */
DECLARE_VPU_CONFIG(MYRIAD_FORCE_PURE_TENSOR_ITERATOR);

//
// Myriad plugin options
//

DECLARE_VPU_CONFIG(MYRIAD_POWER_MANAGEMENT);
DECLARE_VPU_CONFIG(MYRIAD_POWER_FULL);
DECLARE_VPU_CONFIG(MYRIAD_POWER_INFER);
DECLARE_VPU_CONFIG(MYRIAD_POWER_STAGE);
DECLARE_VPU_CONFIG(MYRIAD_POWER_STAGE_SHAVES);
DECLARE_VPU_CONFIG(MYRIAD_POWER_STAGE_NCES);

DECLARE_VPU_CONFIG(MYRIAD_WATCHDOG);

DECLARE_VPU_CONFIG(MYRIAD_DEVICE_CONNECT_TIMEOUT);

DECLARE_VPU_CONFIG(MYRIAD_ENABLE_ASYNC_DMA);

namespace VPUConfigParams {

IE_SUPPRESS_DEPRECATED_START
// Used to update API usage in the dependent repos.
DECLARE_VPU_CONFIG_KEY(DETECT_NETWORK_BATCH);
IE_SUPPRESS_DEPRECATED_END

}  // namespace VPUConfigParams

}  // namespace InferenceEngine
