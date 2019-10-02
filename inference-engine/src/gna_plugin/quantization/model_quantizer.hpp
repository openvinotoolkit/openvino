// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <utility>
#include <string>
#include "layer_transform.hpp"
#include "gna_graph_tools.hpp"
#include "details/ie_cnn_network_tools.h"
#include "layer_quantizer.hpp"
#include "scale_factor_calc.hpp"
#include "weights_converter.hpp"

namespace GNAPluginNS {

/**
 * Quantize entire cnn - network
 * @tparam T - type trait for weights and biases
 */
template<class T>
class ModelQuantizer {
 public:
    InferenceEngine::ICNNNetwork::Ptr quantize(InferenceEngine::ICNNNetwork &model, float scaleFactor) const {
        return quantize(model, [](InferenceEngine::CNNNetPtr &){}, std::vector<float>({scaleFactor}));
    }

    template <class PreQuantisationCb>
    InferenceEngine::ICNNNetwork::Ptr quantize(InferenceEngine::ICNNNetwork &model, const PreQuantisationCb &cb, float scaleFactor) const {
        return quantize(model, cb, std::vector<float>({scaleFactor}));
    }

    InferenceEngine::ICNNNetwork::Ptr quantize(InferenceEngine::ICNNNetwork &model, std::vector<float> scaleFactor) const {
        return quantize(model, [](InferenceEngine::CNNNetPtr &){}, scaleFactor);
    }

    template <class PreQuantisationCb>
    InferenceEngine::ICNNNetwork::Ptr quantize(InferenceEngine::ICNNNetwork &model, const PreQuantisationCb &cb, std::vector<float> scaleFactor) const {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            auto newLayer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
            transformLayer(newLayer, WeightsConverter());
            return newLayer;
        };
        auto copiedNet = InferenceEngine::CNNNetCopy(model, visitor);

        // TODO: probably not the best way of using dynamic cast in order to transform Precision
        // one of solution is to create not copyNet overloads, that accepts 2 functors, one for layer copy
        // and another one for net copy
        auto rawNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(copiedNet.get());
        if (rawNet != nullptr) {
            rawNet->setPrecision(T::mandatory().getNetPrecision());
        }

        // allow client code to access copied topology, to avoid copies if user would like to chain quantisation with
        // another preprocessing
        cb(copiedNet);

        if (scaleFactor.empty()) {
            THROW_GNA_EXCEPTION << "Scale factor is empty";
        }

        LayersQuantizer<T> lc(*scaleFactor.begin());
        auto sortedNewNet = InferenceEngine::details::CNNNetSortTopologically(*copiedNet.get());
        gnalog() << "Sorted layers: " << std::endl;
        for (auto &&layer : sortedNewNet) {
            gnalog() << layer->name << std::endl;
        }
        /// filling scale factors for input layers, memory layers will have scaleFactor of 1.0 by default
        InferenceEngine::InputsDataMap dm;
        copiedNet->getInputsInfo(dm);
        int scaleIndex = 0;
        for (auto &&inputData : dm) {
            auto inputLayer = inputData.second->getInputData()->getCreatorLayer().lock();
            auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            if (scaleFactor.size() <= scaleIndex) {
                THROW_GNA_EXCEPTION << "Index of scale factor element is incorrect";
            }
            quantData->_src_quant.scale = scaleFactor[scaleIndex];
            scaleIndex++;
        }

        propagateScaleFactor(sortedNewNet, T::mandatory().getWeightsPrecision().size());

        // sorted order gives possibility for propagate quantisation along depended layers
        for (auto &&layer : sortedNewNet) {
            transformLayer(layer, lc);
        }

        return copiedNet;
    }

 private :
    void propagateScaleFactor(std::vector<InferenceEngine::CNNLayerPtr> & net, int weightsBytesSize) const {
        ScaleFactorCalculator sf(net, weightsBytesSize);

        while (!sf.allLayersProcessed()) {
            for (auto &&layer : sf.getStartLayers()) {
                transformLayer(layer, sf);
                // transforming until we reached cases where output scale updated due to situation in downstream layer
                if (sf.needToRestart()) {
                    break;
                }
            }
        }
    }
};
}  // namespace GNAPluginNS
