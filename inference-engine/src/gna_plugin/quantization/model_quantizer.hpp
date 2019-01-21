// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#pragma once
#include <vector>
#include "gna_plugin_config.hpp"
#include "layer_transform.hpp"
#include "graph_tools.hpp"
#include "details/ie_cnn_network_tools.h"
#include "layer_quantizer.hpp"
#include "scale_factor_calc.hpp"

namespace GNAPluginNS {
/**
 * Quantize entire cnn - network
 * @tparam T - type trait for weights and biases
 */
template<class T>
class ModelQuantizer {
 public:
    CNNNetworkPtr quantize(InferenceEngine::ICNNNetwork &model, float scaleFactor) const {
        return quantize(model, [](InferenceEngine::CNNNetPtr &){}, scaleFactor);
    }

    template <class PreQuantisationCb>
    CNNNetworkPtr quantize(InferenceEngine::ICNNNetwork &model, const PreQuantisationCb &cb, float scaleFactor) const {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            return InferenceEngine::injectData<QuantizedLayerParams>(lp);
        };
        auto copiedNet = InferenceEngine::CNNNetCopy(model, visitor);

        // TODO: probably not the best way of using dynamic cast in order to transform Precision
        // one of solution is to create not copyNet overloads, that accepts 2 functors, one for layer copy
        // and another one for net copy
        auto rawNet = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(copiedNet.get());
        rawNet->setPrecision(T::mandatory().getNetPrecision());

        // allow client code to access copied topology, to avoid copies if user would like to chain quantisation with
        // another preprocessing
        cb(copiedNet);

        LayersQuantizer<T> lc(scaleFactor);
        auto sortedNewNet = InferenceEngine::details::CNNNetSortTopologically(*copiedNet.get());
        gnalog() << "Sorted layers: " << std::endl;
        for (auto &&layer : sortedNewNet) {
            gnalog() << layer->name << std::endl;
        }

        // weights scale is a hint, not all weightable layer preserve it in all possible precisions
        propagateScaleFactor(sortedNewNet, T::mandatory().getWeightsPrecision().size(), scaleFactor);

        // sorted order gives possibility for propagate quantisation along depended layers
        for (auto &&layer : sortedNewNet) {
            transformLayer(layer, lc);
        }

        return copiedNet;
    }

 private :
    void propagateScaleFactor(std::vector<InferenceEngine::CNNLayerPtr> & net, int weightsBytesSize, float scaleFactor) const {
        ScaleFactorCalculator sf(net, weightsBytesSize, scaleFactor);

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
