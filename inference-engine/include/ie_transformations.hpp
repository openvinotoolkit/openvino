// Copyright (C) 2020-2021 Intel Corporation
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
INFERENCE_ENGINE_API_CPP(void) LowLatency(InferenceEngine::CNNNetwork& network);

/**
 * @brief The transformation finds nodes in ngraph::Function by provided `nodes_to_replace` names and
 * replaces these nodes with Constants. Constants are created inside the transformation with provided values.
 *
 * Example:
 * 1. before transformation:
 *  Parameter (shape: 3, 2) -> Split (axis: 1, num_split=2) -> AnyNode_1
 *                                                          \
 *                                                            -> AnyNode_2
 *
 * 2. transformation call, freeze Split layer with values (1, 2, 3), (4, 5, 6)
 * 3. after transformation:
 *  Const_1 (shape: 3, 1; value: (1, 2, 3)) -> AnyNode_1
 *  Const_2 (shape: 3, 1; value: (4, 5, 6)) -> AnyNode_2
 *
 * @param network A network to apply LowLatency transformation
 * @param nodes_to_replace A map contains names of nodes to replace and corresponding values for each output of the node.
 */
INFERENCE_ENGINE_API_CPP(void) FreezeNodes(InferenceEngine::CNNNetwork& network,
                                           const std::map<std::string, std::vector<std::vector<char>>>& nodes_to_replace);
} // namespace InferenceEngine
