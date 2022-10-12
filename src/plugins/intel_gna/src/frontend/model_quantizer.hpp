// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <utility>
#include <string>
#include <type_traits>

#include <legacy/details/ie_cnn_network_tools.h>
#include <legacy/layer_transform.hpp>

#include "gna_graph_tools.hpp"
#include "layer_quantizer.hpp"
#include "scale_factor_calc.hpp"
#include "weights_converter.hpp"
#include "gna_itt.hpp"
#include "descriptions/gna_desc.hpp"

namespace GNAPluginNS {

/**
 * Quantize entire cnn - network
 * @tparam T - type trait for weights and biases
 */
template<class T>
class ModelQuantizer {
 public:
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, float scaleFactor) const {
        return quantize(model, [](const InferenceEngine::CNNNetwork &, bool runBeforeCopy, bool lowPrecision){}, scaleFactor);
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, const PreQuantisationCb &cb, float scaleFactor) const {
        return quantize(model, cb, std::vector<float>({scaleFactor}));
    }

    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model, std::vector<float> scaleFactors) const {
        return quantize(model, [](InferenceEngine::CNNNetwork &, bool runBeforeCopy, bool lowPrecision){}, scaleFactors);
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model,  const PreQuantisationCb &cb, std::vector<float> scaleFactors) const {
        GNAPluginNS::GnaInputs inputs;
        InferenceEngine::InputsDataMap inputs_map = model.getInputsInfo();
        int sf_id = 0;
        for (auto &&input_data : inputs_map) {
            auto input_layer = getCreatorLayer(input_data.second->getInputData()).lock();
            if (scaleFactors.size() <= sf_id) {
                THROW_GNA_EXCEPTION << "Scale factors are not set for some of the inputs";
            }
            inputs[input_layer->name].scale_factor = scaleFactors[sf_id++];
        }

        return quantize(model, cb, inputs);
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork &model,
                                         const PreQuantisationCb &cb,
                                         const GNAPluginNS::GnaInputs &inputs) const {
        OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "ModelQuantizer::quantize");
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

        float lc_sf = GNAPluginNS::kScaleFactorDefault;
        if (inputs.empty()) {
            gnawarn() << "Inputs structure is empty, will be used default scale factor: " << lc_sf << std::endl;
        } else {
            lc_sf = inputs.Get().begin()->scale_factor;
        }

        LayersQuantizer<T> lc(lc_sf);
        auto sortedNewNet = InferenceEngine::details::CNNNetSortTopologically(copiedNet);
        gnalog() << "Sorted layers: " << std::endl;
        for (auto &&layer : sortedNewNet) {
            auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
            quantData->lowPrecision = lowPrecision;
            gnalog() << layer->name << std::endl;
        }
        /// filling scale factors for input layers, memory layers will have scaleFactor of 1.0 by default
        InferenceEngine::InputsDataMap dm = copiedNet.getInputsInfo();
        for (auto &&inputData : dm) {
            auto inputLayer = getCreatorLayer(inputData.second->getInputData()).lock();
            auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            IE_ASSERT(quantData != nullptr);
            quantData->_src_quant.SetScale(inputs.at(inputLayer->name).scale_factor);
        }

        propagateScaleFactor(sortedNewNet);

        // sorted order gives possibility for propagate quantisation along depended layers
        for (auto &&layer : sortedNewNet) {
            transformLayer(layer, lc);
        }

        return copiedNet;
    }

 private :
    void propagateScaleFactor(std::vector<InferenceEngine::CNNLayerPtr> & net) const {
        ScaleFactorCalculator<T> sf(net);

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

            // We are looking for infinite loop by using algorithm of compute prefix function, complexity O(N)
            // (a part of the Knuth–Morris–Pratt algorithm).
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

                // The pattern length is a length of a repeating string sequence (it is 2 in the example below).
                // concat_14_input_0_reshape#concat_15
                // concat_15_input_1_reshape#add_12
                // add_12#Add_16
                // Reshape_41#add_12
                // add_12#Add_16
                // Reshape_41#add_12
                //
                // In the case of pattern length is 1, an infinite loop can be found on 2 consecutive strings.
                // To avoid this, we will expect the appearance of 4 equal strings for the case pattern length is 1.
                if ((infiniteLoopHistory.size() - i) % 2 == 0 &&
                    (infiniteLoopHistory.size() - i) / 2 == infiniteLoopHistory.size() - k &&
                    ((infiniteLoopHistory.size() - i) / 2 > 1 ||
                        std::distance(infiniteLoopHistory.rbegin(),
                            std::find_if_not(infiniteLoopHistory.rbegin(), infiniteLoopHistory.rend(),
                                [&infiniteLoopHistory](const std::string& str) { return str == infiniteLoopHistory.back(); })) > 3)) {
                    gnalog() << "infiniteLoopPattern:\n";
                    for (const auto& s : infiniteLoopPattern) {
                        gnalog() << "\t " << s << '\n';
                    }
                    infiniteLoopPattern.clear();
                    int patternLength = (infiniteLoopHistory.size() - i) / 2;
                    gnalog() << "patternLength: " << patternLength << '\n';
                    for (int j = 0; j < patternLength; j++) {
                        infiniteLoopPattern.emplace_back(infiniteLoopHistory[infiniteLoopHistory.size() - patternLength + j]);
                    }
                    gnalog() << "infiniteLoopHistory:\n";
                    for (const auto& s : infiniteLoopHistory) {
                        gnalog() << "\t " << s << '\n';
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
