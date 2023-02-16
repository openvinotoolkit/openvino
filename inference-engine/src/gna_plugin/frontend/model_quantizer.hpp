// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>
#include <string>
#include <type_traits>

#include <legacy/layer_transform.hpp>
#include "gna_graph_tools.hpp"
#include <legacy/details/ie_cnn_network_tools.h>
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
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, float scaleFactor) const {
        return quantize(model, [](const InferenceEngine::CNNNetwork &, bool runBeforeCopy, bool lowPrecision){}, std::vector<float>({scaleFactor}));
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, const PreQuantisationCb &cb, float scaleFactor) const {
        return quantize(model, cb, std::vector<float>({scaleFactor}));
    }

    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, std::vector<float> scaleFactor) const {
        return quantize(model, [](InferenceEngine::CNNNetwork &, bool runBeforeCopy, bool lowPrecision){}, scaleFactor);
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, const PreQuantisationCb &cb, std::vector<float> scaleFactor) const {
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            auto newLayer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
            transformLayer(newLayer, WeightsConverter());
            return newLayer;
        };
        bool lowPrecision = (T::mandatory().getInputPrecision().size() == sizeof(uint8_t));
        InferenceEngine::CNNNetwork copiedNet = InferenceEngine::CNNNetCopy(model);
        cb(copiedNet, true, lowPrecision);

        copiedNet = InferenceEngine::CNNNetCopy(copiedNet, visitor);

        // allow client code to access copied topology, to avoid copies if user would like to chain quantisation with
        // another preprocessing
        cb(copiedNet, false, lowPrecision);

        if (scaleFactor.empty()) {
            THROW_GNA_EXCEPTION << "Scale factor is empty";
        }

        LayersQuantizer<T> lc(*scaleFactor.begin());
        auto sortedNewNet = InferenceEngine::details::CNNNetSortTopologically(copiedNet);
        gnalog() << "Sorted layers: " << std::endl;
        for (auto &&layer : sortedNewNet) {
            auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
            quantData->lowPrecision = lowPrecision;
            gnalog() << layer->name << std::endl;
        }
        /// filling scale factors for input layers, memory layers will have scaleFactor of 1.0 by default
        InferenceEngine::InputsDataMap dm = copiedNet.getInputsInfo();
        int scaleIndex = 0;
        for (auto &&inputData : dm) {
            auto inputLayer = getCreatorLayer(inputData.second->getInputData()).lock();
            auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            if (scaleFactor.size() <= scaleIndex) {
                THROW_GNA_EXCEPTION << "Scale factors are not set for some of the inputs";
            }
            IE_ASSERT(quantData != nullptr);
            quantData->_src_quant.SetScale(scaleFactor[scaleIndex]);
            scaleIndex++;
        }

        bool isFakeQuantize = std::is_same<T, FakeQuantI8>() || std::is_same<T, FakeQuantI16>();
        propagateScaleFactor(sortedNewNet, T::mandatory().getWeightsPrecision().size(), T::optional().getWeightsPrecision().size(),
                             T::mandatory().getInputPrecision().size(), isFakeQuantize);

        // sorted order gives possibility for propagate quantisation along depended layers
        for (auto &&layer : sortedNewNet) {
            transformLayer(layer, lc);
        }

        return copiedNet;
    }

 private :
    void propagateScaleFactor(std::vector<InferenceEngine::CNNLayerPtr> & net, int mandWeightsBytesSize,
                              int optWeightsBytesSize, int inputsBytesSize, bool fakeQuantize) const {
        ScaleFactorCalculator sf(net, mandWeightsBytesSize, optWeightsBytesSize, inputsBytesSize, fakeQuantize);

        int infiniteLoopCount = 0;
        std::vector<std::string> infiniteLoopPattern;
        std::vector<std::string> infiniteLoopHistory;
        while (!sf.allLayersProcessed() && infiniteLoopCount <= 2) {
            auto layers = sf.getStartLayers();
            infiniteLoopHistory.emplace_back(layers.front()->name);
            for (auto &&layer : layers) {
                transformLayer(layer, sf);
                // transforming until we reached cases where output scale updated due to situation in downstream layer
                if (sf.needToRestart()) {
                    infiniteLoopHistory.back() += "#" + layer->name;
                    break;
                }
            }

            // looking for infinite loop by using algorithm of compute prefix function, complexity O(N)
            std::map<int, int> prefixFunction;
            int k = infiniteLoopHistory.size();
            for (int i = infiniteLoopHistory.size() - 2; i >= 0; i--) {
                while (k < infiniteLoopHistory.size() && infiniteLoopHistory[k - 1] != infiniteLoopHistory[i]) {
                    auto iter = prefixFunction.find(k);
                    k = iter == prefixFunction.end() ? infiniteLoopHistory.size() : iter->second;
                }

                if (infiniteLoopHistory[k - 1] == infiniteLoopHistory[i]) {
                    k--;
                }

                if ((infiniteLoopHistory.size() - i) % 2 == 0 && (infiniteLoopHistory.size() - i) / 2 == infiniteLoopHistory.size() - k) {
                    infiniteLoopPattern.clear();
                    int patternLength = (infiniteLoopHistory.size() - i)/2;
                    for (int j = 0; j < patternLength; j++) {
                        infiniteLoopPattern.emplace_back(infiniteLoopHistory[infiniteLoopHistory.size() - patternLength + j]);
                    }
                    infiniteLoopHistory.clear();
                    gnalog() << "infinite loop detected\n";
                    break;
                }

                prefixFunction.emplace(i, k);
            }

            if (infiniteLoopHistory.empty()) {
                infiniteLoopCount++;
            } else {
                if (infiniteLoopCount > 0 &&
                    (infiniteLoopHistory.size()%infiniteLoopPattern.size() == 0 || sf.allLayersProcessed()) &&
                    !std::equal(infiniteLoopHistory.begin() + (infiniteLoopHistory.size() - infiniteLoopPattern.size()),
                        infiniteLoopHistory.end(), infiniteLoopPattern.begin())) {
                    infiniteLoopCount = 0;
                    infiniteLoopPattern.clear();
                    gnalog() << "infinite loop fixed\n";
                }
            }

            sf.SetInfiniteLoopCount(infiniteLoopCount);
        }

        if (infiniteLoopCount > 0) {
            std::string additionalInformation;
            for (const auto& p : infiniteLoopPattern) {
                additionalInformation += '\n' + p;
            }
            THROW_GNA_EXCEPTION << "infinite loop: " + additionalInformation;
        }
    }
};
}  // namespace GNAPluginNS
