// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This header file defines the list of public transformations.
 *
 * @file ie_transformations.hpp
 */

#pragma once

#include <ie_api.h>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {

/**
 * @brief The transformation finds all TensorIterator layers in the network, processes all back
 * edges that describe a connection between Result and Parameter of the TensorIterator body,
 * and inserts ReadValue layer between Parameter and the next layers after this Parameter,
 * and Assign layer after the layers before the Result layer.
 * Supported platforms: CPU, GNA.
 *
 *  The example below describes the changes to the inner part (body, back edges) of the TensorIterator layer.
 *  [] - TensorIterator body
 *  () - new layer
 *
 *  before applying the transformation:
 *  back_edge_1 -> [Parameter -> some layers ... -> Result ] -> back_edge_1
 *
 *  after applying the transformation:
 *  back_edge_1 -> [Parameter -> (ReadValue layer) -> some layers ... -> (Assign layer) ]
 *                                                              \
 *                                                               -> Result ] -> back_edge_1
 *
 *  It is recommended to use this transformation in conjunction with the Reshape feature to set sequence
 *  dimension to 1 and with the UnrollTensorIterator transformation.
 *  For convenience, we have already enabled the unconditional execution of the UnrollTensorIterator
 *  transformation when using the LowLatency transformation for CPU, GNA plugins, no action is required here.
 *  After applying both of these transformations, the resulting network can be inferred step by
 *  step, the states will store between inferences.
 *
 *    An illustrative example, not real API:
 *
 *    network->reshape(...) // Set sequence dimension to 1, recalculating shapes. Optional, depends on the network.
 *    LowLatency(network)   // Applying LowLatency and UnrollTensorIterator transformations.
 *    network->infer (...)  // Calculating new values for states.
 *    // All states are stored between inferences via Assign, ReadValue layers.
 *    network->infer (...)  // Using stored states, calculating new values for states.
 *
 * @param network A network to apply LowLatency transformation
 * *
 */

INFERENCE_ENGINE_DEPRECATED("This transformation will be removed in 2023.1. "
                            "Use InferenceEngine::lowLatency2 instead.")
INFERENCE_ENGINE_API_CPP(void) LowLatency(InferenceEngine::CNNNetwork& network);


/**
 * @brief The transformation finds all TensorIterator/Loop layers in the network,
 * processes all back edges that describe a connection between Result and Parameter
 * of the TensorIterator/Loop bodies,and inserts ReadValue and Assign layers at the
 * input and output corresponding to this back edge.
 * Supported platforms: CPU, GNA.
 *
 * The example below describes the changes made by the transformation
 *  [] - TensorIterator body
 *  () - new layer
 *  BE - back-edge
 *
 *  before applying the transformation:
 *  -> input1[BE_1 -> Parameter -> Layers ... -> Result  -> BE_1 ]output1->
 *
 *  after applying the transformation:
 *  ->(ReadValue)-> input1[BE_1 ->Parameter->Layers ...->Result->BE_1]output1 ->(Assign)
 *                                                                      \
 *                                                                       ->...
 * After applying the transformation, the resulting network can be inferred
 * step by step, the states will store between inferences.
 * @param network A network to apply LowLatency transformation
 * @param use_const_initializer Changes the type of the initializing subgraph for ReadValue operations.
          If "true", then the transformation inserts Constant before ReadValue operation.
          If "false, then the transformation leaves existed initializing subgraph for ReadValue operation.
 * Loop operation by a given number. Does not affect TensorIterators.
 * *
 */
INFERENCE_ENGINE_API_CPP(void) lowLatency2(InferenceEngine::CNNNetwork& network,
                                           bool use_const_initializer = true);

} // namespace InferenceEngine
