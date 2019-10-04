// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common functions for graph transformation
 * @file graph_transformer.h
 */

#pragma once

#include <map>
#include <vector>
#include <string>
#include <ie_icnn_network.hpp>
#include <details/caseless.hpp>
#include "cnn_network_impl.hpp"

namespace InferenceEngine {

/**
 * @brief TBD
 */
class INFERENCE_ENGINE_API_CLASS(ConstTransformer) {
public:
    explicit ConstTransformer(details::CNNNetworkImpl* _network);
    virtual ~ConstTransformer() = default;

    /**
     * @brief calculates const layers, combines const subgraph into a single const layers
     */
    void foldConstSubgraphs();

    /**
      * @brief folds Const Subgraphs and removes second input of Reshape-like layers (Interp, Gather, Resample, ...)
      */
    void fullTrim();

protected:
    /**
     * @brief collect all const layers with marking if it defines shape (1 - for shape, 0 - otherwise)
     */
    virtual const std::map<std::string, bool> getConstLayers(const std::vector<CNNLayerPtr>& sortedLayers);

    /**
     * @brief TBD
     */
    virtual const BlobMap
        getConstData(const std::map<std::string, bool>& constLayers, const std::vector<CNNLayerPtr>& sortedLayers);

    /**
     * @brief TBD
     */
    virtual std::vector<std::string>
    foldConstSubgraphsInternal(const std::map<std::string, bool>& constLayers, const BlobMap& constData,
                               const std::vector<CNNLayerPtr>& sortedLayers);

    /**
     * @brief TBD
     */
    virtual void trimShapeInputs(const std::vector<std::string>& constLayers);

private:
    const details::caseless_set<std::string> shapeTaking = {"Reshape", "Resample", "Interp", "Squeeze", "Unsqueeze"};
    details::CNNNetworkImpl* network;
};

}  // namespace InferenceEngine
