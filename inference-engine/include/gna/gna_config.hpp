// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for GNA plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file gna_config.hpp
 */

#pragma once

#include "ie_plugin_config.hpp"

namespace InferenceEngine {

/**
 * @brief GNA plugin configuration
 */
namespace GNAConfigParams {

/**
 * @def GNA_CONFIG_KEY(name)
 * @brief Shortcut for defining configuration keys
 */
#define GNA_CONFIG_KEY(name) InferenceEngine::GNAConfigParams::_CONFIG_KEY(GNA_##name)
/**
 * @def GNA_CONFIG_VALUE(name)
 * @brief Shortcut for defining configuration values
 */
#define GNA_CONFIG_VALUE(name) InferenceEngine::GNAConfigParams::GNA_##name

#define DECLARE_GNA_CONFIG_KEY(name) DECLARE_CONFIG_KEY(GNA_##name)
#define DECLARE_GNA_CONFIG_VALUE(name) DECLARE_CONFIG_VALUE(GNA_##name)

/**
* @brief Scale factor that is calculated by user, in order to use static quantisation feature
* This option should be used with floating point value serialized to string with decimal separator equals to . (dot)
* @details For multiple input case, individual scale factors can be passed, using KEY_GNA_SCALE_FACTOR[_input_layer_name]
* where input_layer can be obtained from from CNNNetwork::GetInputsInfo
*/
DECLARE_GNA_CONFIG_KEY(SCALE_FACTOR);

/**
* @brief By default gna api works with Int16 weights precision, however this can be adjusted if necessary,
* currently supported values are I16, I8
*/
DECLARE_GNA_CONFIG_KEY(PRECISION);

/**
* @brief if turned on, dump GNA firmware model into specified file
*/
DECLARE_GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE);

/**
* @brief information on GNA generation chosen for firmware model dump, can be overridden by GNA3
*/
DECLARE_GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE_GENERATION);

/**
* @brief GNA proc_type setting that should be one of GNA_AUTO, GNA_HW, GNA_SW, GNA_SW_EXACT
*/
DECLARE_GNA_CONFIG_KEY(DEVICE_MODE);

DECLARE_GNA_CONFIG_VALUE(AUTO);
DECLARE_GNA_CONFIG_VALUE(HW);
DECLARE_GNA_CONFIG_VALUE(HW_WITH_SW_FBACK);
DECLARE_GNA_CONFIG_VALUE(SW);
DECLARE_GNA_CONFIG_VALUE(SW_EXACT);
DECLARE_GNA_CONFIG_VALUE(SW_FP32);
DECLARE_GNA_CONFIG_VALUE(GEN);
DECLARE_GNA_CONFIG_VALUE(GEN_EXACT);
DECLARE_GNA_CONFIG_VALUE(SSE);
DECLARE_GNA_CONFIG_VALUE(SSE_EXACT);
DECLARE_GNA_CONFIG_VALUE(AVX1);
DECLARE_GNA_CONFIG_VALUE(AVX1_EXACT);
DECLARE_GNA_CONFIG_VALUE(AVX2);
DECLARE_GNA_CONFIG_VALUE(AVX2_EXACT);

/**
* @brief The option to override the GNA HW execution target. May be one of GNA_TARGET_2_0, GNA_TARGET_3_0.
* By default (in case of no value set) the behavior depends on GNA HW availability:
* If GNA HW is present, use the option corresponding to this HW.
* If HW is not present, use the option corresponding to the latest fully supported GNA HW generation.
* A fully supported GNA HW generation means it must be supported by booth the OV GNA Plugin and the core GNA Library.
* For the GNA Library 2.0.X.Y, the latest supported GNA HW generation corresponds to GNA_TARGET_2_0.
* For the GNA Library 2.1.X.Y, the latest supported GNA HW generation corresponds to GNA_TARGET_3_0.
* For the OV GNA Plugin 2021.4, the latest supported GNA HW generation corresponds to GNA_TARGET_3_0.
*/
DECLARE_GNA_CONFIG_KEY(EXEC_TARGET);

DECLARE_GNA_CONFIG_VALUE(TARGET_2_0);
DECLARE_GNA_CONFIG_VALUE(TARGET_3_0);

/**
* @brief The option to override the GNA HW compile target. May be one of GNA_TARGET_2_0, GNA_TARGET_3_0.
* By default the same as GNA_EXEC_TARGET.
*/
DECLARE_GNA_CONFIG_KEY(COMPILE_TARGET);

/**
* @brief if enabled produced minimum memory footprint for loaded network in GNA memory, default value is YES
*/
DECLARE_GNA_CONFIG_KEY(COMPACT_MODE);

/**
* @brief The option to enable/disable uniformly distributed PWL algorithm.
* By default (in case of NO value set) the optimized algorithm called "Recursive Descent Algorithm for Finding
* the Optimal Minimax Piecewise Linear Approximation of Convex Functions is used.
* If value is YES then simple uniform distribution used to create PWL approximation of activation functions
* Uniform distribution usually gives poor approximation with same number of segments
*/
DECLARE_GNA_CONFIG_KEY(PWL_UNIFORM_DESIGN);

/**
* @brief The option to allow to specify the maximum error percent that the optimized algorithm finding
* will use to find PWL functions.
* By default (in case of NO value set), 1.0 value is used.
*/
DECLARE_GNA_CONFIG_KEY(PWL_MAX_ERROR_PERCENT);

/**
* @brief By default, the GNA plugin uses one worker thread for inference computations.
* This parameter allows you to create up to 127 threads for software modes.
*
* Note that multithreading mode does not guarantee the same computation order as order
* of issuing. Additionally, in this case, software modes do not implement any serializations.
*/
DECLARE_GNA_CONFIG_KEY(LIB_N_THREADS);
}  // namespace GNAConfigParams

namespace Metrics {
    /**
    * @brief Metric to get a std::string of GNA Library version, usually in the form <API_REVISION>.<RELEASE_LINE>.<RELEASE>.<BUILD>
    */
    DECLARE_METRIC_KEY(GNA_LIBRARY_FULL_VERSION, std::string);
}  // namespace Metrics

namespace PluginConfigParams {

/**
 * @brief The key controls threading inside GNA Inference Engine plugin.
 *
 * It is passed to Core::SetConfig(), this option should be used with values:
 * PluginConfigParams::YES or PluginConfigParams::NO
 */
DECLARE_CONFIG_KEY(SINGLE_THREAD);

}  // namespace PluginConfigParams

}  // namespace InferenceEngine
