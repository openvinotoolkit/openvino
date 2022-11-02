// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include "gna_data_types.hpp"
#include "gna_plugin_config.hpp"
#include "layers/gna_layer_info.hpp"
#include "quantized_layer_params.hpp"

using namespace GNAPluginNS;

namespace ov {
namespace intel_gna {
namespace frontend {

enum class QuantizedDataType { input, output, weights, bias };

/**
 * @brief Returns a scale factor for specific layer data
 * @param layer Layer to be quantized
 * @param data_type Type of data to be quantized
 * @return scale factor
 */
float GetScaleFactor(InferenceEngine::CNNLayerPtr layer, QuantizedDataType data_type);

/**
* @brief Calculates a scale factor from FakeQuantize statistics according to the formula:
* scale factor = max representable value / max absolute input value
* @param levels Number of integer quants
* @param minValue Minimum value to be quantized
* @param maxValue Maximum value to be quantized
*/
float CalculateScaleFactorFromStats(size_t levels, float minValue, float maxValue);

struct ScaleFactorUpdateResult {
    InferenceEngine::CNNLayer* restartLayer = nullptr;
    ScaleFactorUpdateResult() = default;
    explicit ScaleFactorUpdateResult(InferenceEngine::CNNLayer* restartlayer) : restartLayer(restartlayer) {}
    operator bool() {
        return restartLayer == nullptr;
    }
};

/**
 * @brief scale factor calculator will calculate only output scale factors for the layer
 * if scale factor propagation is not possible, it will fall indicate a restart condition
 */
class ScaleFactorCalculator {
    using Cnt = std::vector<InferenceEngine::CNNLayerPtr>;
    Cnt net;
    const Config& gna_config;
    const bool fake_quantized;
    mutable Cnt::const_iterator idx;
    mutable bool needRestart = false;
    int infiniteLoopCount = 0;

    std::vector<double> getPWLSlopes(const GNAPluginNS::LayerInfo& info) const;
    static float selectBestOutputScaleFactors(float inScale,
                                              std::vector<float> outScales,
                                              const std::vector<double>& slopes);
    static float selectBestWeightsScaleFactors(float inScale,
                                               float outScale,
                                               std::vector<float> weightsScales,
                                               const std::vector<double>& slopes);
    static std::vector<float> generateScaleFactors(float startRange, float endRange, size_t numScaleFactors);
    static double calculateWeightsReducerFromDstStats(QuantizationParams dst_quant);
    static bool requantizeInput(InferenceEngine::CNNLayerPtr input,
                                float newOutputScale,
                                ScaleFactorUpdateResult& result,
                                int infiniteLoopCount);
    float adjustScaleFactor(float sf,
                            InferenceEngine::CNNLayer const* cnnLayer,
                            GNAPluginNS::LayerInfo const& layer,
                            QuantizedLayerParams* quantizedParams) const;
    float getActivationScale(InferenceEngine::CNNLayer const* cnnLayer,
                             GNAPluginNS::LayerInfo const& layer,
                             int inputsSize,
                             const bool fakeQuantize) const;
    bool ScaleFactorPerLayerCNN(InferenceEngine::CNNLayer* cnnLayer,
                                ScaleFactorUpdateResult& result,
                                int infiniteLoopCount,
                                const GNAPluginNS::Config& gna_config) const;
    bool ScaleFactorPerLayerConcat(InferenceEngine::ConcatLayer* concatLayer,
                                   ScaleFactorUpdateResult& result,
                                   int infiniteLoopCount,
                                   const Config& gna_config) const;
    bool ScaleFactorPerLayerEltwise(InferenceEngine::EltwiseLayer* eltwiseLayer,
                                    ScaleFactorUpdateResult& result,
                                    int infiniteLoopCount,
                                    const Config& gna_config) const;
    bool ScaleFactorPerLayerGemm(InferenceEngine::GemmLayer* gemmLayer,
                                 ScaleFactorUpdateResult& result,
                                 int infiniteLoopCount,
                                 const Config& gna_config) const;
    bool ScaleFactorPerLayerWeightable(InferenceEngine::WeightableLayer* wl,
                                       ScaleFactorUpdateResult& result,
                                       int infiniteLoopCount,
                                       const Config& gna_config) const;

 public:
    ScaleFactorCalculator(Cnt& net, const Config& gna_config, const bool fake_quantized)
        : net(net),
          gna_config(gna_config),
          fake_quantized(fake_quantized) {
        idx = std::begin(this->net);
    }
    bool needToRestart() const {
        return needRestart;
    }
    bool allLayersProcessed() const {
        return idx == std::end(net);
    }
    std::vector<InferenceEngine::CNNLayerPtr> getStartLayers() const {
        return std::vector<InferenceEngine::CNNLayerPtr>(idx, std::end(net));
    }
    void SetInfiniteLoopCount(int infiniteLoopCount) {
        this->infiniteLoopCount = infiniteLoopCount;
    }
    bool CalculateScaleFactor(InferenceEngine::CNNLayerPtr layer_ptr) const {
        ScaleFactorUpdateResult result;
        needRestart = false;
        auto layer_info = LayerInfo(layer_ptr);

        // check  layer_info.isConvolution() || layer_info.isScaleShift()
        if (layer_info.isConcat()) {
            if (!ScaleFactorPerLayerConcat(dynamic_cast<InferenceEngine::ConcatLayer*>(layer_ptr.get()),
                                            result,
                                            infiniteLoopCount,
                                            gna_config)) {
                return false;
            }
        } else if (layer_info.isEltwise()) {
            if (!ScaleFactorPerLayerEltwise(dynamic_cast<InferenceEngine::EltwiseLayer*>(layer_ptr.get()),
                                            result,
                                            infiniteLoopCount,
                                            gna_config)) {
                return false;
            }
        } else if (layer_info.isGemm()) {
            if (!ScaleFactorPerLayerGemm(dynamic_cast<InferenceEngine::GemmLayer*>(layer_ptr.get()),
                                         result,
                                         infiniteLoopCount,
                                         gna_config)) {
                return false;
            }
        } else if (layer_info.isWeightable()) {
            if (!ScaleFactorPerLayerWeightable(dynamic_cast<InferenceEngine::WeightableLayer*>(layer_ptr.get()),
                                               result,
                                               infiniteLoopCount,
                                               gna_config)) {
                return false;
            }
        } else {
            if (!ScaleFactorPerLayerCNN(layer_ptr.get(), result, infiniteLoopCount, gna_config)) {
                return false;
            }
        }

        if (result) {
            idx++;
            return true;
        }

        idx = std::find_if(net.begin(), net.end(), [&](InferenceEngine::CNNLayerPtr cnnLayer) {
            if (!result) {
                return result.restartLayer == cnnLayer.get();
            }
            return layer_ptr == cnnLayer;
        });

        if (idx != net.end()) {
            idx++;
        }

        needRestart = true;
        return true;
    }
};

}  // namespace frontend
}  // namespace intel_gna
}  // namespace ov
