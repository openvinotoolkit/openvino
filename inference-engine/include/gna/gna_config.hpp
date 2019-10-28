// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header that defines advanced related properties for GNA plugin.
 * These properties should be used in SetConfig() and LoadNetwork() methods of plugins
 *
 * @file gna_config.hpp
 */

#pragma once

#include <string>
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
* @brief By default gna api work in Int16 precision, however this can be adjusted if necessary,
* currently supported values are I16, I8
*/
DECLARE_GNA_CONFIG_KEY(PRECISION);


/**
* @brief if turned on, dump GNA firmware model into specified file
*/
DECLARE_GNA_CONFIG_KEY(FIRMWARE_MODEL_IMAGE);

/**
* @brief GNA proc_type setting that should be one of GNA_AUTO, GNA_HW, GNA_SW, GNA_SW_EXACT
*/
DECLARE_GNA_CONFIG_KEY(DEVICE_MODE);

DECLARE_GNA_CONFIG_VALUE(AUTO);
DECLARE_GNA_CONFIG_VALUE(HW);
DECLARE_GNA_CONFIG_VALUE(SW);
DECLARE_GNA_CONFIG_VALUE(SW_EXACT);
DECLARE_GNA_CONFIG_VALUE(SW_FP32);

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
* @brief By default, the GNA plugin uses one worker thread for inference computations.
* This parameter allows you to create up to 127 threads for software modes.
*
* Note that multithreading mode does not guarantee the same computation order as order
* of issuing. Additionally, in this case, software modes do not implement any serializations.
*/
DECLARE_GNA_CONFIG_KEY(LIB_N_THREADS);
}  // namespace GNAConfigParams
}  // namespace InferenceEngine
