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
 * Quantize entire network
 */
class ModelQuantizer {
 public:
    // Below four functions are used only in unit tests
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork& model, float scale_factor) const {
        return quantize(model, [](const InferenceEngine::CNNNetwork&, bool run_before_copy, bool low_precision){}, scale_factor);
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork& model, const PreQuantisationCb& cb, float scale_factor) const {
        return quantize(model, cb, std::vector<float>({scale_factor}));
    }

    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork& model, std::vector<float> scale_factors) const {
        return quantize(model, [](InferenceEngine::CNNNetwork&, bool run_before_copy, bool low_precision){}, scale_factors);
    }

    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork& model,  const PreQuantisationCb& cb, std::vector<float> scale_factors) const {
        GNAPluginNS::GnaInputs inputs;
        InferenceEngine::InputsDataMap inputs_map = model.getInputsInfo();
        int sf_id = 0;
        for (auto &&input_data : inputs_map) {
            auto input_layer = getCreatorLayer(input_data.second->getInputData()).lock();
            if (scale_factors.size() <= sf_id) {
                THROW_GNA_EXCEPTION << "Scale factors are not set for some of the inputs";
            }
            inputs[input_layer->name].scale_factor = scale_factors[sf_id++];
        }

        return quantize(model, cb, inputs, InferenceEngine::Precision::I16, false);
    }

    // This function is acutally used by the plugin
    template <class PreQuantisationCb>
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork& model,
                                         const PreQuantisationCb& cb,
                                         const GNAPluginNS::GnaInputs& inputs,
                                         const InferenceEngine::Precision& gna_precision,
                                         const bool& inputs_int8_precision) const {
        OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "ModelQuantizer::quantize");
        auto visitor = [&](InferenceEngine::CNNLayerPtr lp) {
            auto new_layer = InferenceEngine::injectData<QuantizedLayerParams>(lp);
            transformLayer(new_layer, WeightsConverter());
            return new_layer;
        };

        // User's hints for GNA weights precision
        const bool weights_int8_precision = (gna_precision == InferenceEngine::Precision::I8) ? true : false;

        InferenceEngine::CNNNetwork copied_net = InferenceEngine::CNNNetCopy(model);
        cb(copied_net, true, inputs_int8_precision);

        copied_net = InferenceEngine::CNNNetCopy(copied_net, visitor);

        // Allow client code to access copied topology, to avoid copies if user would like to chain quantisation with
        // another preprocessing
        cb(copied_net, false, inputs_int8_precision);

        LayersQuantizer lc(GNAPluginNS::kScaleFactorDefault);
        auto sorted_new_net = InferenceEngine::details::CNNNetSortTopologically(copied_net);
        gnalog() << "Sorted layers: " << std::endl;
        for (auto &&layer : sorted_new_net) {
            auto quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
            quant_data->_inputs_int8_precision = inputs_int8_precision;
            quant_data->_weights_int8_precision = weights_int8_precision;
            gnalog() << layer->name << std::endl;
        }

        // Filling scale factors for input layers, memory layers will have scale factor of 1.0 by default
        InferenceEngine::InputsDataMap dm = copied_net.getInputsInfo();
        for (auto &&inputData : dm) {
            auto input_layer = getCreatorLayer(inputData.second->getInputData()).lock();
            auto quant_data = InferenceEngine::getInjectedData<QuantizedLayerParams>(input_layer);
            IE_ASSERT(quant_data != nullptr);
            quant_data->_src_quant.SetScale(inputs.at(input_layer->name).scale_factor);
        }

        propagateScaleFactor(sorted_new_net);

        // Sorted order gives possibility for propagate quantisation along depended layers
        for (auto &&layer : sorted_new_net) {
            transformLayer(layer, lc);
        }

        return copied_net;
    }

 private :
    void propagateScaleFactor(std::vector<InferenceEngine::CNNLayerPtr>& net) const {
        ScaleFactorCalculator sf(net);

        uint32_t inf_loop_count = 0;
        std::vector<std::string> inf_loop_pattern;
        std::vector<std::string> inf_loop_history;
        while (!sf.allLayersProcessed() && inf_loop_count <= 2) {
            auto layers = sf.getStartLayers();
            inf_loop_history.emplace_back(layers.front()->name);
            for (auto &&layer : layers) {
                transformLayer(layer, sf);
                // transforming until we reached cases where output scale updated due to situation in downstream layer
                if (sf.needToRestart()) {
                    inf_loop_history.back() += "#" + layer->name;
                    break;
                }
            }

            // We are looking for infinite loop by using algorithm of compute prefix function, complexity O(N)
            // (a part of the Knuth–Morris–Pratt algorithm).
            std::map<int, int> prefix_function;
            int32_t k = inf_loop_history.size();
            for (int32_t i = inf_loop_history.size() - 2; i >= 0; i--) {
                while (k < inf_loop_history.size() && inf_loop_history[k - 1] != inf_loop_history[i]) {
                    auto iter = prefix_function.find(k);
                    k = iter == prefix_function.end() ? inf_loop_history.size() : iter->second;
                }

                if (inf_loop_history[k - 1] == inf_loop_history[i]) {
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
                if ((inf_loop_history.size() - i) % 2 == 0 &&
                    (inf_loop_history.size() - i) / 2 == inf_loop_history.size() - k &&
                    ((inf_loop_history.size() - i) / 2 > 1 ||
                        std::distance(inf_loop_history.rbegin(),
                            std::find_if_not(inf_loop_history.rbegin(), inf_loop_history.rend(),
                                [&inf_loop_history](const std::string& str) { return str == inf_loop_history.back(); })) > 3)) {
                    gnalog() << "inf_loop_pattern:\n";
                    for (const auto& s : inf_loop_pattern) {
                        gnalog() << "\t " << s << '\n';
                    }
                    inf_loop_pattern.clear();
                    int pattern_len = (inf_loop_history.size() - i) / 2;
                    gnalog() << "pattern_len: " << pattern_len << '\n';
                    for (int j = 0; j < pattern_len; j++) {
                        inf_loop_pattern.emplace_back(inf_loop_history[inf_loop_history.size() - pattern_len + j]);
                    }
                    gnalog() << "inf_loop_history:\n";
                    for (const auto& s : inf_loop_history) {
                        gnalog() << "\t " << s << '\n';
                    }
                    inf_loop_history.clear();
                    gnalog() << "infinite loop detected\n";
                    break;
                }

                prefix_function.emplace(i, k);
            }

            if (inf_loop_history.empty()) {
                inf_loop_count++;
            } else {
                if (inf_loop_count > 0 &&
                    (inf_loop_history.size()%inf_loop_pattern.size() == 0 || sf.allLayersProcessed()) &&
                    !std::equal(inf_loop_history.begin() + (inf_loop_history.size() - inf_loop_pattern.size()),
                        inf_loop_history.end(), inf_loop_pattern.begin())) {
                    inf_loop_count = 0;
                    inf_loop_pattern.clear();
                    gnalog() << "infinite loop fixed\n";
                }
            }

            sf.SetInfiniteLoopCount(inf_loop_count);
        }

        if (inf_loop_count > 0) {
            std::string additional_info;
            for (const auto& p : inf_loop_pattern) {
                additional_info += '\n' + p;
            }
            THROW_GNA_EXCEPTION << "infinite loop: " + additional_info;
        }
    }
};
}  // namespace GNAPluginNS
