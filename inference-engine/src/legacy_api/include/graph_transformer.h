// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file with common functions for graph transformation
 * @file graph_transformer.h
 */

#pragma once

#include <details/caseless.hpp>
#include <ie_icnn_network.hpp>
#include <map>
#include <string>
#include <vector>

#include "cnn_network_impl.hpp"

namespace InferenceEngine {

/**
 * @brief TBD
 */
class INFERENCE_ENGINE_API_CLASS(ConstTransformer) {
public:
    explicit ConstTransformer(ICNNNetwork* _network);
    explicit ConstTransformer(details::CNNNetworkImpl* _network);
    explicit ConstTransformer(std::vector<DataPtr> &_inputs, std::vector<DataPtr> &_outputs);

    virtual ~ConstTransformer() = default;

    /**
     * @brief calculates const layers, combines const subgraph into a single const layers
     */
    void foldConstSubgraphs();

    /**
     * @brief folds Const Subgraphs and removes second input of Reshape-like layers (Interp, Gather, Resample, ...)
     */
    void fullTrim();

    /**
     * @brief move blobs from Constant layers to Convolution or FullyConnected layers attributes
     */
    void moveWeights();

protected:
    /**
     * @brief collect all const layers with marking if it defines shape (1 - for shape, 0 - otherwise)
     */
    virtual const std::map<std::string, bool> getConstLayers(const std::vector<CNNLayerPtr>& sortedLayers);

    /**
     * @brief TBD
     */
    virtual const BlobMap getConstData(const std::map<std::string, bool>& constLayers,
                                       const std::vector<CNNLayerPtr>& sortedLayers);

    /**
     * @brief TBD
     */
    virtual std::vector<CNNLayerPtr> foldConstSubgraphsInternal(const std::map<std::string, bool>& constLayers,
                                                                const BlobMap& constData,
                                                                const std::vector<CNNLayerPtr>& sortedLayers);

    /**
     * @brief TBD
     */
    virtual void trimShapeInputs(const std::vector<CNNLayerPtr>& constLayers,
                                 std::vector<CNNLayerPtr>& allLayers);

    /**
     * @brief TBD
     */
    void cleanup();

private:
    const details::caseless_set<std::string> shapeTaking = {"Reshape", "Resample", "Interp", "Squeeze", "Unsqueeze"};
    details::CNNNetworkImpl* network;
    std::vector<DataPtr> inputs;
    std::vector<DataPtr> outputs;

    /** data/layer collection to restore valida state of network if it was specified */
    std::vector<DataPtr> data_to_remove;
    std::vector<DataPtr> data_to_add;

    std::vector<CNNLayerPtr> layer_to_remove;
    std::vector<CNNLayerPtr> layer_to_add;
};

}  // namespace InferenceEngine
