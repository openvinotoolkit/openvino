// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>
#include <vector>

#include "ie_icnn_network.hpp"
#include "legacy/graph_tools.hpp"

namespace InferenceEngine {
namespace NetPass {

IE_SUPPRESS_DEPRECATED_START

/**
 * Try to detect LSTM Sequence pattern inside TI and convert it
 *
 * @param net network to modify
 * @return true if all Tensor iterator was converted
 */
INFERENCE_ENGINE_API_CPP(bool) CombineRNNSeq(ICNNNetwork& net);
INFERENCE_ENGINE_API_CPP(bool) CombineRNNSeq(TensorIterator::Body& net);

/**
 * Returns a vector of the topologically sorted layers from
 * the passed TI layer body.
 *
 * @param body TI body
 * @return vector of layer objects
 */
INFERENCE_ENGINE_API_CPP(std::vector<CNNLayerPtr>) TIBodySortTopologically(const TensorIterator::Body& body);

/**
 * Check if provided layer contains internal attribute like subnet/subgraph
 *
 * @param layer to check
 * @return true if layer has subnet
 */
bool HasInternalSubnet(const CNNLayerPtr &layer);

/**
 * Extract internal subnet from layer
 *
 * All internal layers are returned by reference. Any modification further subnet modification will
 * has affect on original layer state.
 *
 * @param layer to proceed
 * @return internal subnet
 */
details::CNNSubnet GetInternalSubnet(const CNNLayerPtr &layer);

/**
 * Unroll all present Tensor Iterators
 *
 * @param net network to modify
 * @return true if all Tensor iterator was unrolled successfully
 */
INFERENCE_ENGINE_API_CPP(bool) UnrollTI(ICNNNetwork& net);

/**
 * Unroll all RNN specific layers by predicate
 *
 * Will be applied to all RNNSeq and RNNCell layers
 *
 * @param net network to modify
 * @param pred predicate to mark layer to unroll
 * @return true if all RNN layers was unrolled successfully
 */
INFERENCE_ENGINE_API_CPP(bool) UnrollRNN_if(ICNNNetwork& net, std::function<bool(const RNNCellBase&)> pred);

/**
 * Construct a copy of provided subnet. Will change names by adding suffix if it was provided.
 *
 * @param subnet to copy from
 * @param suffix is optional attribute. Will be added into name of each layer/data object if provided
 * @return subnet copy. Each layer/data object is newly created. Const blob objects is inherited from
 *         original subnet.
 */
INFERENCE_ENGINE_API_CPP(TensorIterator::Body) CopyTIBody(const TensorIterator::Body& body, std::string suffix = std::string());

INFERENCE_ENGINE_API_CPP(bool) UnrollRNN_if(TensorIterator::Body& net, std::function<bool(const RNNCellBase&)> pred);

IE_SUPPRESS_DEPRECATED_END

/**
 * Precision conversion pass
 *
 * Will perform conversion of all presented tensors with specified precision including
 * const blobs and intermediate tensors. It doesn't check layer semantic. It may break
 * correctness of topology.
 *
 * It also remove redundant convert layers if they will appear.
 *
 * @param net is network to apply conversion
 * @param from precision of tensors required conversion
 * @param to resulting precision of tensors
 */
INFERENCE_ENGINE_API_CPP(void) ConvertPrecision(ICNNNetwork& net, Precision from, Precision to);

INFERENCE_ENGINE_API_CPP(void) ConvertIOPrecision(ICNNNetwork& net, Precision from, Precision to);

}  // namespace NetPass
}  // namespace InferenceEngine
