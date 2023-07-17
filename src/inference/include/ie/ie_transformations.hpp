// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This header file defines the list of public transformations.
 *
 * @file ie_transformations.hpp
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

#include "cpp/ie_cnn_network.h"
#include "ie_api.h"

namespace InferenceEngine {

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
 */
INFERENCE_ENGINE_1_0_DEPRECATED INFERENCE_ENGINE_API_CPP(void)
    lowLatency2(InferenceEngine::CNNNetwork& network, bool use_const_initializer = true);
}  // namespace InferenceEngine
