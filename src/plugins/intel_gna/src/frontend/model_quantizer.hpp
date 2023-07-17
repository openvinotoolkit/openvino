// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <legacy/details/ie_cnn_network_tools.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "descriptions/gna_desc.hpp"
#include "gna_graph_tools.hpp"
#include "gna_itt.hpp"
#include "gna_plugin_config.hpp"
#include "gna_transformations_pipeline.hpp"
#include "layer_quantizer.hpp"
#include "scale_factor_calc.hpp"
#include "weights_converter.hpp"

namespace ov {
namespace intel_gna {
namespace frontend {

/**
 * Quantize entire network
 */
class ModelQuantizer {
    ov::intel_gna::TransformationsPipeline& gna_transformer;

public:
    ModelQuantizer(ov::intel_gna::TransformationsPipeline& transformer) : gna_transformer(transformer) {}
    InferenceEngine::CNNNetwork quantize(const InferenceEngine::CNNNetwork& model, const GnaInputs& inputs) const {
        OV_ITT_SCOPED_TASK(itt::domains::GNA_LT, "ModelQuantizer::quantize");
        auto visitor = [&](InferenceEngine::CNNLayerPtr layer_ptr) {
            auto new_layer = InferenceEngine::injectData<QuantizedLayerParams>(layer_ptr);
            convert_blobs_precision(*new_layer);
            return new_layer;
        };

        InferenceEngine::CNNNetwork copied_net = InferenceEngine::CNNNetCopy(model);
        gna_transformer.apply_legacy(copied_net, true);
        copied_net = InferenceEngine::CNNNetCopy(copied_net, visitor);

        // Allow client code to access copied topology, to avoid copies if user would like to chain quantisation with
        // another preprocessing
        gna_transformer.apply_legacy(copied_net, false);

        auto sorted_new_net = InferenceEngine::details::CNNNetSortTopologically(copied_net);
        log::debug() << "Sorted layers: " << std::endl;

        // Filling scale factors for input layers, memory layers will have scale factor of 1.0 by default
        InferenceEngine::InputsDataMap dm = copied_net.getInputsInfo();
        for (auto&& inputData : dm) {
            auto input_layer = getCreatorLayer(inputData.second->getInputData()).lock();
            auto quant_data = InferenceEngine::getInjectedData<frontend::QuantizedLayerParams>(input_layer);
            IE_ASSERT(quant_data != nullptr);
            quant_data->_src_quant.SetScale(inputs.at(input_layer->name).scale_factor);
        }

        // Propagate scale factor and quantize layers
        propagateScaleFactor(sorted_new_net);
        frontend::LayerQuantizer lq(gna_transformer.config);

        for (auto&& layer : sorted_new_net) {
            lq.quantize(*layer);
        }

        return copied_net;
    }

private:
    void propagateScaleFactor(std::vector<InferenceEngine::CNNLayerPtr>& net) const {
        ScaleFactorCalculator sf(net, gna_transformer.config, gna_transformer.is_fake_quantized());
        uint32_t inf_loop_count = 0;
        std::vector<std::string> inf_loop_pattern;
        std::vector<std::string> inf_loop_history;
        while (!sf.allLayersProcessed() && inf_loop_count <= 2) {
            auto layers = sf.getStartLayers();
            inf_loop_history.emplace_back(layers.front()->name);
            for (auto&& layer : layers) {
                sf.CalculateScaleFactor(layer);
                // transforming until we reached cases where output scale updated due to situation in downstream layer
                if (sf.needToRestart()) {
                    inf_loop_history.back() += "#" + layer->name;
                    break;
                }
            }

            // We are looking for infinite loop by using algorithm of compute prefix function, complexity O(N)
            // (a part of the Knuth–Morris–Pratt algorithm).
            std::map<int, int> prefix_function;
            auto k = static_cast<int>(inf_loop_history.size());
            for (int32_t i = static_cast<int32_t>(inf_loop_history.size()) - 2; i >= 0; i--) {
                while (k < static_cast<int>(inf_loop_history.size()) &&
                       inf_loop_history[k - 1] != inf_loop_history[i]) {
                    auto iter = prefix_function.find(k);
                    k = iter == prefix_function.end() ? static_cast<int>(inf_loop_history.size()) : iter->second;
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
                                   std::find_if_not(inf_loop_history.rbegin(),
                                                    inf_loop_history.rend(),
                                                    [&inf_loop_history](const std::string& str) {
                                                        return str == inf_loop_history.back();
                                                    })) > 3)) {
                    log::debug() << "inf_loop_pattern:\n";
                    for (const auto& s : inf_loop_pattern) {
                        log::debug() << "\t " << s << '\n';
                    }
                    inf_loop_pattern.clear();
                    auto pattern_len = (static_cast<int>(inf_loop_history.size()) - i) / 2;
                    log::debug() << "pattern_len: " << pattern_len << '\n';
                    for (int j = 0; j < pattern_len; j++) {
                        inf_loop_pattern.emplace_back(inf_loop_history[inf_loop_history.size() - pattern_len + j]);
                    }
                    log::debug() << "inf_loop_history:\n";
                    for (const auto& s : inf_loop_history) {
                        log::debug() << "\t " << s << '\n';
                    }
                    inf_loop_history.clear();
                    log::debug() << "infinite loop detected\n";
                    break;
                }

                prefix_function.emplace(i, k);
            }

            if (inf_loop_history.empty()) {
                inf_loop_count++;
            } else {
                if (inf_loop_count > 0 &&
                    ((inf_loop_pattern.size() > 0 && (inf_loop_history.size() % inf_loop_pattern.size() == 0)) ||
                     sf.allLayersProcessed())) {
                    size_t history_shift = 0;
                    size_t pattern_shift = 0;

                    if (inf_loop_history.size() > inf_loop_pattern.size()) {
                        history_shift = inf_loop_history.size() - inf_loop_pattern.size();
                    } else {
                        pattern_shift = inf_loop_pattern.size() - inf_loop_history.size();
                    }

                    if (!std::equal(inf_loop_history.begin() + history_shift,
                                    inf_loop_history.end(),
                                    inf_loop_pattern.begin() + pattern_shift)) {
                        inf_loop_count = 0;
                        log::debug() << "inf_loop_pattern:\n";
                        for (const auto& s : inf_loop_pattern) {
                            log::debug() << "\t " << s << '\n';
                        }
                        inf_loop_pattern.clear();
                        log::debug() << "infinite loop fixed\n";
                    }
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

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
