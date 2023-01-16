// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/ie_layers.h>

#include <ie/ie_input_info.hpp>
#include <string>

#include "backend/dnn_components.hpp"
#include "descriptions/gna_desc.hpp"

namespace ov {
namespace intel_gna {

/**
 * @namespace helpers contains helpers tools for gna plugin.
 */
namespace helpers {

/**
 * @brief Update expected orientation for model input of given \p inputLayer. It is needed to recognize if extra
 * transposition for input data of input layer is needed.
 *
 * @note Function check following parameters if:
 *  - there is at least one dnn input layer for given cnn layer
 *  - corresponding dnn layers operation are not kDnnInterleaveOp and not kDnnDeinterleaveOp
 *  - corresponding input layer first dimenions and product of rest of dimensions is greater than 1
 *
 * If any of conditions is not met kDnnNonInterleavedOrientation is set.
 * If all of conditions above will be met kDnnInterleavedOrientation is set.
 *
 * @param inputLayer model input layer
 * @param components layers transformed to form consumable by GNA
 * @param inputs of model
 *
 * @throws if orientations of input for multiple layers are different
 */
void updateModelInputOrientationWithoutConvolution(const InferenceEngine::CNNLayer& inputLayer,
                                                   const backend::DnnComponents& components,
                                                   GnaInputs& inputs);

/**
 * @brief Update expected orientation for model output of given \p outputName. It is needed to recognize if extra
 * transposition for output data of output layer is needed.
 *
 * @note Function checks following parameters if:
 *  - corresponding dnn layer operation is kDnnInterleaveOp or kDnnDeinterleaveOp
 *  - corresponding dnn layer is present and columns and rows numbes are bigger than 1
 *
 * If any of conditions above will be not met orientation is set to kDnnNonInterleavedOrientation.
 * If there is no corespnding dnn layer orientation is untouched.
 *
 * @param outputName name of the model output
 * @param cnnLayerName name of coresponding model input layer
 * @param components layers transformed to form consumable by GNA
 * @param outputs outputs of model
 *
 */
void updateModelOutputOrientation(const std::string& outputName,
                                  const std::string& cnnlayerName,
                                  const backend::DnnComponents& components,
                                  GnaOutputs& outputs);

}  // namespace helpers
}  // namespace intel_gna
}  // namespace ov
