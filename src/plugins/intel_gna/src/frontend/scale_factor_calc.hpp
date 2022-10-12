// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <string>
#include <tuple>

#include <legacy/ie_layers.h>
#include "gna_upstream_iterator.hpp"
#include "layers/gna_layer_info.hpp"
#include "layers/gna_convolution_layer.hpp"
#include "gna_plugin_log.hpp"
#include "gna_slope_scale.h"
#include "runtime/pwl.h"
#include "gna_data_types.hpp"
#include "round_float_define.hpp"

namespace GNAPluginNS {

template<typename QUANT_DESC>
class ScaleFactorCalculator;

namespace frontend {
static const float MIN_SEARCH_WEIGHTS_VAL = 1.0f;
static const float MAX_SEARCH_WEIGHTS_VAL = 1024.0f;

struct ScaleFactorUpdateResult {
    InferenceEngine::CNNLayer *restartLayer = nullptr;
    ScaleFactorUpdateResult() = default;
    explicit ScaleFactorUpdateResult(InferenceEngine::CNNLayer * restartlayer) : restartLayer(restartlayer) {
    }
    operator bool() {
        return restartLayer == nullptr;
    }
};

/**
* @brief Calculates a scale factor from FakeQuantize statistics according to the formula:
* scale factor = max representable value / max absolute input value
* @param levels Number of integer quants
* @param minValue Minimum value to be quantized
* @param maxValue Maximum value to be quantized
*/
inline float CalculateScaleFactorFromStats(size_t levels, float minValue, float maxValue) {
    return maxValue == minValue ? 1.0f : (levels - 1) / (maxValue - minValue);
}

/**
 * @brief Compares two float values and returns if they are equal
 * @param p1 First float value
 * @param p2 Second float value
 * @return Returns true if two float values are equal
 */
static bool fp32eq(float p1, float p2, float accuracy = 0.00001f) {
    return (std::abs(p1 - p2) <= accuracy * std::min(std::abs(p1), std::abs(p2)));
}

/**
 * @brief Calculates PWL slopes for specified function in a given input range
 * @param info Layer information
 * @return Array of slopes for a function
 */
static std::vector<double> getPWLSlopes(const LayerInfo& info) {
    if (info.isIdentity() || info.isFakeQuantize() || info.isRelu() || info.isClamp() || info.isAbs()) {
        return { 1.0f };
    }

    return {};
}

/**
 * @brief Finds the best output activation scale factor that allows to get the most precise PWL slope
 * @param inScale Input activation layer scale factor
 * @param outScales Array of output activation scale factors
 * @param slopes Array of slopes for a given function
 * @return Best output activation scale factor
 */
static float selectBestOutputScaleFactors(float inScale, std::vector<float> outScales, const std::vector<double>& slopes) {
    std::vector<float> scaleErrors;
    for (size_t i = 0; i < outScales.size(); ++i) {
        auto outScale = outScales[i];

        auto sd = 0.0f;
        for (size_t j = 0; j < slopes.size(); ++j) {
            auto s = gna_slope(slopes[j], inScale, outScale);
            auto slope = FLOAT_TO_INT16(s.slope * s.slope_scale);
            if (slope < std::numeric_limits<int16_t>::min() || slope > std::numeric_limits<int16_t>::max()) {
                sd += std::numeric_limits<int8_t>::max();
                continue;
            }

            auto testSlope = static_cast<double>(slope) / s.slope_scale * inScale / outScale;
            if (fp32eq(static_cast<float>(testSlope), static_cast<float>(slopes[j]), 1.0E-6f)) {
                return outScale;
            }

            sd += pow(testSlope - slopes[j], 2.0);
        }

        sd /= slopes.size();
        sd = sqrtf(sd);
        scaleErrors.push_back(sd);
    }

    size_t minIndex = 0;
    auto minError = scaleErrors[0];
    for (size_t i = 1; i < scaleErrors.size(); ++i) {
        if (scaleErrors[i] < minError) {
            minError = scaleErrors[i];
            minIndex = i;
        }
    }

    return outScales[minIndex];
}

/**
 * @brief Finds the weights scale factor that allows to get the most precise PWL slope
 * @param inScale Input weightable layer scale factor
 * @param outScale Output activation scale factor
 * @param weightsScales Array of weights scales to check
 * @return Best weights scale factor
 */
static float selectBestWeightsScaleFactors(float inScale, float outScale, std::vector<float> weightsScales,
    const std::vector<double>& slopes) {
    std::vector<float> scaleErrors;
    for (size_t i = 0; i < weightsScales.size(); ++i) {
        auto weightScale = weightsScales[i];

        auto sd = 0.0;
        for (size_t j = 0; j < slopes.size(); ++j) {
            auto s = gna_slope(slopes[j], inScale * weightScale, outScale);
            auto slope = static_cast<uint32_t>(s.slope * s.slope_scale);
            if (slope < static_cast<uint32_t>(std::numeric_limits<int16_t>::min()) &&
                slope > static_cast<uint32_t>(std::numeric_limits<int16_t>::max())) {
                sd += std::numeric_limits<int8_t>::max();
                continue;
            }

            auto testSlope = static_cast<double>(slope) / s.slope_scale * (inScale * weightScale) / outScale;
            if (fp32eq(static_cast<float>(testSlope), static_cast<float>(slopes[j]))) {
                return outScale;
            }
            sd += pow(testSlope - slopes[j], 2.0);
        }

        sd /= slopes.size();
        sd = sqrtf(sd);
        scaleErrors.push_back(sd);
    }

    size_t minIndex = 0;
    auto minError = scaleErrors[0];
    for (size_t i = 1; i < scaleErrors.size(); ++i) {
        if (scaleErrors[i] < minError) {
            minError = scaleErrors[i];
            minIndex = i;
        }
    }

    return weightsScales[minIndex];
}

/**
 * @brief Generates specified number of scale factors in a given range.
 * @param startRange First scale factor
 * @param endRange Last scale factor
 * @param numIterations number of scale factors to generate
 * @return Array of scale factors
 */
static std::vector<float> generateScaleFactors(float startRange, float endRange, size_t numScaleFactors) {
    if (!numScaleFactors) {
        return { startRange, endRange };
    }

    auto scaleFactors = std::vector<float>{};
    auto domain = endRange - startRange;
    auto step = domain / numScaleFactors;
    for (size_t i = 0; i <= numScaleFactors; ++i) {
        auto scale = startRange + step * i;
        if (!std::isnan(scale)) {
            scaleFactors.push_back(scale);
        }
    }

    return scaleFactors;
}

/**
 * @brief Calculates weights reducer using statistics from the destination quantization statistics.
 * @param dst_quant destination quantization statistics
 * @return Weights reducer
 */
static double calculateWeightsReducerFromDstStats(Quantization dst_quant) {
    auto maxAbsVal = std::max(std::abs(dst_quant.GetMinValues().front()),
        std::abs(dst_quant.GetMaxValues().front()));

    auto maxIntVal = static_cast<int64_t>(maxAbsVal * dst_quant.GetScale() + 0.5f);
    double weightsReducer = static_cast<double>(maxIntVal) / std::numeric_limits<int32_t>::max();
    weightsReducer = std::max(1.0, weightsReducer);
    return weightsReducer;
}

/**
 * @brief Tries to re-quantize an input to reach the desired output scale factor value.
 * This function searches for layers above which output scale factors can be changed:
 * - activations,
 * - constants,
 * - weightable layers (output scale factor is modified by modification of weights scale factor).
 * @param input input to be re-quantized
 * @param newOutputScale the desired output scale factor value
 * @param result information about the restarted layer
 * @return true if the input can be re-quantized
 */
static bool requantizeInput(InferenceEngine::CNNLayerPtr input, float newOutputScale, ScaleFactorUpdateResult &result, int infiniteLoopCount) {
    auto layer = input;
    if (!layer || LayerInfo(layer).isInput() || LayerInfo(layer).isMemory() || LayerInfo(layer).isCopy()) {
        return false;
    }

    size_t prevInputIdx = 0;
    auto info = LayerInfo(layer);
    auto quantDataForInputLayer = InferenceEngine::getInjectedData<QuantizedLayerParams>(*layer);
    if (quantDataForInputLayer->_dst_quant.IsStatsSet()) {
        auto levels = LayerInfo(layer).has32BOutput() ? (std::numeric_limits<uint32_t>::max() + 1ul) :
            (std::numeric_limits<uint16_t>::max() + 1ul);
        auto maxSF = CalculateScaleFactorFromStats(levels, quantDataForInputLayer->_dst_quant.GetMinValues().front(),
            quantDataForInputLayer->_dst_quant.GetMaxValues().front());
        if (newOutputScale > maxSF) {
            gnalog() << layer->name << ": Scale factor " << newOutputScale << " is too large. The maximum scale factor: "
                << maxSF << " levels=" << levels << " min=" << quantDataForInputLayer->_dst_quant.GetMinValues().front()
                << " max=" << quantDataForInputLayer->_dst_quant.GetMaxValues().front() << "\n";
            return false;
        }
    }
    if (info.isActivation() || info.isConst()) {
        gnawarn() << "[WARNING] requantize " << layer->name
                    << ". Layer new output scale: " << newOutputScale
                    << ", was " << quantDataForInputLayer->_dst_quant.GetScale() << std::endl;
        quantDataForInputLayer->_dst_quant.SetScale(newOutputScale);
        result = ScaleFactorUpdateResult(layer.get());
        return true;
    }

    if (info.isWeightableIdentity() && !fp32eq(quantDataForInputLayer->_weights_quant.GetScale(), 1.0f)) {
        auto reducer = std::max(1.0f, quantDataForInputLayer->_dst_quant.GetScale() / newOutputScale);
        auto newWeightsScale = std::max(1.0f, quantDataForInputLayer->_weights_quant.GetScale() / reducer);
        quantDataForInputLayer->_weights_quant.SetScale(static_cast<int32_t>(newWeightsScale));
        quantDataForInputLayer->_dst_quant.SetScale(quantDataForInputLayer->_weights_quant.GetScale() *
            quantDataForInputLayer->_src_quant.GetScale());

        result = ScaleFactorUpdateResult(layer.get());
        return true;
    }

    if (info.isFullyConnected() || info.isConvolution() || info.isPower()) {
        quantDataForInputLayer->_dst_quant.SetScale(newOutputScale);
        quantDataForInputLayer->_weights_quant.SetScale(newOutputScale / quantDataForInputLayer->_src_quant.GetScale());
        result = ScaleFactorUpdateResult(layer.get());
        return true;
    }

    if (LayerInfo(layer).isEltwiseSum() && LayerInfo(input).has32BOutput()) {
        // re-quantize bias branch for Eltwise layer
         for (uint8_t ix = 0; ix < 2; ++ix) {
            if (LayerInfo(InferenceEngine::CNNNetPrevLayer(layer, ix)).has32BOutput()) {
                prevInputIdx = ix;
                break;
            }
        }
        auto prevLayer = InferenceEngine::CNNNetPrevLayer(layer, prevInputIdx);
        auto prevQuantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
        newOutputScale *= prevQuantData->_dst_quant.GetScale() / quantDataForInputLayer->_dst_quant.GetScale();
    }

    if (LayerInfo(layer).isEltwiseMul()) {
        for (uint8_t ix = 0; ix < 2; ++ix) {
            auto restartedPrevLayer = InferenceEngine::CNNNetPrevLayer(layer, ix);
            auto otherPrevLayer = InferenceEngine::CNNNetPrevLayer(layer, !ix);
            if (infiniteLoopCount % 2 == 1) {
                gnawarn() << "[WARNING] going into the loop: swap inputs order for Eltwise Multiply" << std::endl;
                std::swap(restartedPrevLayer, otherPrevLayer);
            }
            auto otherPrevQuantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*otherPrevLayer);
            auto newScale = newOutputScale / otherPrevQuantData->_dst_quant.GetScale();
            if (requantizeInput(restartedPrevLayer, newScale, result, infiniteLoopCount)) {
                return true;
            }
        }
        return false;
    }

    auto prevLayer = InferenceEngine::CNNNetHasPrevLayer(layer.get(), prevInputIdx) ?
        InferenceEngine::CNNNetPrevLayer(layer, prevInputIdx) : nullptr;
    return requantizeInput(prevLayer, newOutputScale, result, infiniteLoopCount);
}

/**
 * @brief calculates output scale factor per layer
 * @tparam T
 */
template<typename T, typename QUANT_DESC>
class ScaleFactorPerLayer {
 public:
    /**
     * @brief calculates weights scale factor to fit dynamic range into target bitsize,
     * also calculates output scale factor for the given layer
     * @param cnnLayer
     * @param result
     * @return
     */
    bool operator()(T cnnLayer, ScaleFactorUpdateResult &result, int infiniteLoopCount) {
        return false;
    }
};

template<typename QUANT_DESC>
class ScaleFactorPerLayer<InferenceEngine::CNNLayer*, QUANT_DESC> {
 private :
    const float activation_scale_factor = 2048.f;
    const float low_prec_activation_scale_factor = 4.f;
    const float identity_scale_factor = 2049.0f;
    const float max_activation_scale_factor = 4096.0f;
    const float k = 5;
    const float k_identity = 6;
    const double pow_domain = 16;

 protected :
    /**
     * @brief Adjust output scale factor to get the most precise PWL slope.
     * NOTE: Currently it is only implemented for identity, clamp, relu and FQ layers.
     *       For all other layers, it does not improve accuracy.
     * @param sf Scale factor to be adjusted
     * @param layer Layer information
     * @param quantizedParams Quantization parameters
     * @return the adjusted scale factor
     */
    float adjustScaleFactor(float sf, InferenceEngine::CNNLayer const* cnnLayer,
                            GNAPluginNS::LayerInfo const& layer,
                            QuantizedLayerParams* quantizedParams) {
        auto get_rank = [](uint32_t value) {
            uint8_t rank = 0;
            while (value >= 1) {
                ++rank;
                value /= 10;
            }
            return rank;
        };
        auto pow_10 = [](uint8_t degree) {
            uint32_t value = 1;
            for (uint8_t i = 0; i < degree; ++i) {
                value *= 10;
            }
            return value;
        };

        auto slopes = getPWLSlopes(layer);
        if (!slopes.empty()) {
            auto div = 10;
            auto startRange = sf > 1.0f ? static_cast<uint32_t>(sf) : sf;
            auto endRange = startRange - startRange / div;
            endRange = endRange > 1.0f ? static_cast<uint32_t>(endRange) : endRange;
            uint32_t steps = 10000;
            uint32_t rangeSize = static_cast<uint32_t>(startRange - endRange);
            if (rangeSize >= 1) {
                steps *= rangeSize / pow_10(get_rank(rangeSize) - 1);
            }

            auto scaleFactors = generateScaleFactors(startRange, endRange, steps);
            auto newScaleFactor = selectBestOutputScaleFactors(quantizedParams->_src_quant.GetScale(), scaleFactors, slopes);
            if (!fp32eq(sf, newScaleFactor) && !fp32eq(newScaleFactor, 0.0f) && !std::isinf(newScaleFactor)) {
                gnalog() << "[INFO] Adjusting scale factor for " << cnnLayer->name
                    << " from: " << sf << " to: " << newScaleFactor << "\n";
                sf = newScaleFactor;
            }
        }
        return sf;
    }

    float getActivationScale(InferenceEngine::CNNLayer const* cnnLayer,
                             GNAPluginNS::LayerInfo const& layer,
                             int inputsSize,
                             const bool fakeQuantize) {
        auto quantizedParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);

        // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
        // set the initial value
        float result = (inputsSize == 2 ? activation_scale_factor : low_prec_activation_scale_factor);
        if (layer.isIdentity()) {
// #define accurate_identity_scale_factor
#ifdef accurate_identity_scale_factor
            // searching more accurate scale factor for identity
            const double min_range = 1024.0;
            const double max_range = 2049.0;

            // gna encoded scale factor - the max this is the more accurate PWL approximation
            double optimalK = 0.0f;
            result = min_range;

            for (int slope_scale_index = 1; slope_scale_index != 5; slope_scale_index ++) {
                auto slope_scale = static_cast<double>(static_cast<uint64_t>(1) << (8 * slope_scale_index));
                auto mink = min_range * slope_scale / quantizedParams->_src_quant.GetScale();
                auto maxk = max_range * slope_scale / quantizedParams->_src_quant.GetScale();

                if (mink < std::numeric_limits<int16_t>::max()) {
                    auto localMaxK = std::min(static_cast<double>(std::numeric_limits<int16_t>::max()), maxk);
                    if (localMaxK > optimalK) {
                        result = localMaxK / slope_scale *  quantizedParams->_src_quant.GetScale();
                        optimalK = localMaxK;
                    }
                }
            }
#else
            // GNA scale factor encoding might poor represent target slop scale, we are probing 2 values
            auto s = gna_slope(1.0, quantizedParams->_src_quant.GetScale(), identity_scale_factor);
            auto scale_default = s.slope * s.slope_scale;
            // probing one more quite good approximation for identity
            s = gna_slope(1.0, quantizedParams->_src_quant.GetScale(), identity_scale_factor / 2);
            auto scale_extra = s.slope * s.slope_scale;
            result = fabs(scale_extra) > fabs(scale_default) ?  identity_scale_factor / 2 : identity_scale_factor;

#endif
        } else if (layer.isRelu()) {
            // if activation is one from relu family, we need to apply heuristic to avoid activation output overflow
            auto limit = (inputsSize == 1 ? std::numeric_limits<int8_t>::max() : std::numeric_limits<int32_t>::max()) - 1;

            if (static_cast<uint64_t>(result * quantizedParams->_src_quant.GetScale()) > limit) {
                    result *= 0.5;
            }
        } else if (layer.isPower()) {
            double exponent = 0;
            double scale = 0;
            double offset = 0;
            auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer const*>(cnnLayer);
            if (!powerLayer) {
                std::shared_ptr<ov::intel_gna::op::Pwl> pwl_node;
                if (!cnnLayer->getNode() ||
                    !(pwl_node = std::dynamic_pointer_cast<ov::intel_gna::op::Pwl>(cnnLayer->getNode()))) {
                    IE_THROW() << "Incorrect Power Layer pointer \n";
                } else {
                    auto powerIE = std::dynamic_pointer_cast<ngraph::op::PowerIE>(pwl_node->get_base_node());
                    if (!powerIE) {
                        IE_THROW() << "Incorrect Power Layer pointer \n";
                    } else {
                        exponent = powerIE->power;
                        scale = powerIE->scale;
                        offset = powerIE->shift;
                    }
                }
            } else {
                exponent = powerLayer->power;
                scale = powerLayer->scale;
                offset = powerLayer->offset;
            }

            auto input_min_value = static_cast<double>(std::numeric_limits<int32_t>::min());
            auto input_max_value = static_cast<double>(std::numeric_limits<int32_t>::max());
            auto output_max_value = static_cast<double>((inputsSize == 2) ? std::numeric_limits<int16_t>::max() : std::numeric_limits<int8_t>::max());

            auto x_min = fp32eq(fmod(exponent, 1.0), 0) ? input_min_value / quantizedParams->_src_quant.GetScale() : 0.0;
            x_min = std::max(x_min, -pow_domain);

            auto x_max = input_max_value / quantizedParams->_src_quant.GetScale();
            x_max = std::min(x_max, pow_domain);

            auto val1 = pow(x_min * scale + offset, exponent);
            auto val2 = pow(x_max * scale + offset, exponent);

            auto abs_val = std::max(std::abs(val1), std::abs(val2));
            auto scale_val = output_max_value / abs_val;

            if (!std::isinf(scale_val)) {
                result = static_cast<float>(scale_val);
            }
        }

        // Identity layer is inserted by GNA passes and requires statistics to correctly set output
        // scale factor. POT does not produce any statistics for this layer as it does not exist
        // in the source IR.
        if (fakeQuantize && !quantizedParams->_dst_quant.IsScaleSet() && layer.isIdentity()) {
            auto prevLayer = CNNNetPrevLayer(cnnLayer);
            while (prevLayer != nullptr) {
                auto prevQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
                if (prevQuantParams->_dst_quant.IsStatsSet()) {
                    quantizedParams->_dst_quant.CopyStats(prevQuantParams->_dst_quant);
                    quantizedParams->_src_quant.CopyStats(prevQuantParams->_dst_quant);
                    break;
                }

                // Take the input statistics only if layer does not modify input values.
                if (prevQuantParams->_src_quant.IsStatsSet() &&
                    (LayerInfo(prevLayer).isNonFunctional() || LayerInfo(prevLayer).isMemory() ||
                    LayerInfo(prevLayer).isConst() || LayerInfo(prevLayer).isInput())) {
                    quantizedParams->_dst_quant.CopyStats(prevQuantParams->_src_quant);
                    quantizedParams->_src_quant.CopyStats(prevQuantParams->_src_quant);
                    break;
                }

                // Stop searching for statistics if previous layer does not modify input values.
                if ((LayerInfo(prevLayer).isWeightable() && !LayerInfo(prevLayer).isWeightableIdentity())
                    || LayerInfo(prevLayer).isEltwise() || LayerInfo(prevLayer).isActivation()) {
                    break;
                }

                if (!CNNNetHasPrevLayer(prevLayer.get())) {
                    break;
                }

                prevLayer = CNNNetPrevLayer(prevLayer);
            }

            // If did not find statistics by searching previous layers, check if a next layer has
            // statistics set.
            if (!quantizedParams->_dst_quant.IsStatsSet()) {
                auto donotSkip = [](InferenceEngine::CNNLayerPtr) {
                    return false;
                };

                auto nextLayers = CNNNetGetAllNextLayersSkipCertain(cnnLayer, -1, donotSkip);
                for (auto &l : nextLayers) {
                    auto nextQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*l);
                    if (nextQuantParams->_src_quant.IsStatsSet()) {
                        quantizedParams->_dst_quant.CopyStats(nextQuantParams->_src_quant);
                        quantizedParams->_src_quant.CopyStats(nextQuantParams->_src_quant);
                        break;
                    }

                    // Take output statistics only if a next layer does not modify input values
                    if (nextQuantParams->_dst_quant.IsStatsSet() &&
                        (LayerInfo(l).isNonFunctional() || LayerInfo(l).isMemory())) {
                        quantizedParams->_dst_quant.CopyStats(nextQuantParams->_dst_quant);
                        quantizedParams->_src_quant.CopyStats(nextQuantParams->_dst_quant);
                        break;
                    }
                }
            }
        }

        // Adjust output scale factor based on statistics (if present) in the following steps:
        // 1. calculate scale factor based on output min and max values
        // 2. (temporary W/A) clamp scale factor to maximum activation scale factor
        // 3. search previous layers if there was already scale factor set
        // 4. adjust output scale factor to get the most precise PWL slope
        if (quantizedParams->_dst_quant.IsStatsSet()) {
            auto minOutValue = quantizedParams->_dst_quant.GetMinValues().front();
            auto maxOutValue = quantizedParams->_dst_quant.GetMaxValues().front();
            auto absMax = std::max(std::abs(minOutValue), std::abs(maxOutValue));

            auto levels = std::min(quantizedParams->_dst_quant.GetLevels(), static_cast<size_t>(std::numeric_limits<uint16_t>::max()) + 1);
            result = CalculateScaleFactorFromStats(levels, minOutValue, maxOutValue);
            if (std::isinf(result) || fp32eq(absMax, 0.0f)) {
                result = max_activation_scale_factor;
            }

            // TODO: remove clamping maximum scale factor
            result = result > max_activation_scale_factor ? max_activation_scale_factor : result;
            if (!layer.isIdentity() && !layer.isFakeQuantize() && !layer.isRelu() && !layer.isClamp()) {
                result = result > activation_scale_factor ? activation_scale_factor : result;
            }

            // Take input scale factor from previous layer if previous layer does not modify
            // input values
            bool usePrevScaleFactor = false;
            auto skipNonFunctional = [](InferenceEngine::CNNLayerPtr l) {
                return LayerInfo(l).isNonFunctional();
            };

            auto prevLayer = CNNNetHasPrevLayer(cnnLayer, 0) ? CNNNetPrevLayerSkipCertain(cnnLayer, 0, skipNonFunctional) : nullptr;
            if (prevLayer != nullptr &&
                (layer.isIdentity() || layer.isFakeQuantize()) && LayerInfo(prevLayer).isWeightableIdentity()) {
                auto prevLayerQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
                auto prevLayer2 = CNNNetHasPrevLayer(prevLayer.get(), 0) ? CNNNetPrevLayerSkipCertain(prevLayer, 0, skipNonFunctional) : nullptr;
                if (!fp32eq(prevLayerQuant->_src_quant.GetScale(), 1.0f) &&
                    prevLayerQuant->_src_quant.IsStatsSet() &&
                    (prevLayer2 == nullptr || LayerInfo(prevLayer2).has8BOr16BOutput())) {
                    result = prevLayerQuant->_src_quant.GetScale();
                    usePrevScaleFactor = true;
                }
            }

            if (!usePrevScaleFactor) {
                result = adjustScaleFactor(result, cnnLayer, layer, quantizedParams);
            }
        }

        /* Destination max value for Relu is equal to its source max value,
         * so we can control overflows if we know only destination statistics
         */
        if (!quantizedParams->_dst_quant.IsStatsSet() && quantizedParams->_src_quant.IsStatsSet() && layer.isRelu()) {
            auto maxInValue = quantizedParams->_src_quant.GetMaxValues().front();
            auto limit = (inputsSize == 1 ? std::numeric_limits<int8_t>::max() : std::numeric_limits<int16_t>::max()) - 1;
            if (result * maxInValue > limit) {
                result = limit / maxInValue;
            }
        }

        return result;
    }

 public :
    bool operator()(InferenceEngine::CNNLayer *cnnLayer, ScaleFactorUpdateResult &result, int infiniteLoopCount) {
        if ( !cnnLayer ) {
            IE_THROW() << "Incorrect Layer pointer \n";
        }

        int inputsSize = ScaleFactorCalculator<QUANT_DESC>::GetInputsBytesSize();
        bool fakeQuantize = ScaleFactorCalculator<QUANT_DESC>::IsFakeQuantize();

        LayerInfo layerInfo(*cnnLayer);
        // TODO: current approach set input scale factor for true input layer(s) equals to provided factor,
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);

        if (layerInfo.isMemory()) {
            if (CNNNetHasPrevLayer(cnnLayer) && quant->_dst_quant.IsStatsSet() && !quant->_dst_quant.IsScaleSet()) {
                auto minOutValue = quant->_dst_quant.GetMinValues().front();
                auto maxOutValue = quant->_dst_quant.GetMaxValues().front();
                auto scale = CalculateScaleFactorFromStats(quant->_dst_quant.GetLevels(), minOutValue, maxOutValue);
                quant->_dst_quant.SetScale(scale);
                quant->_src_quant = quant->_dst_quant;
            }

            if (CNNNetHasPrevLayer(cnnLayer)) {
                auto prevLayer = CNNNetPrevLayer(cnnLayer);
                auto inputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
                // locating corresponding memory layers with same ID
                for (auto&& input : CNNNetGetAllInputLayers(cnnLayer)) {
                    LayerInfo ll(input);
                    if (!ll.isMemory() ||
                        !InferenceEngine::details::CaselessEq<std::string>()(input->params["id"], cnnLayer->params["id"])) {
                        continue;
                    }

                    auto quantSibling = InferenceEngine::getInjectedData<QuantizedLayerParams>(input);

                    // after restarting from memory input - quant is fine
                    if (fp32eq(quantSibling->_dst_quant.GetScale(), inputQuant->_dst_quant.GetScale())) {
                        quant->_src_quant.SetScale(inputQuant->_dst_quant.GetScale());
                        quant->_dst_quant.SetScale(inputQuant->_dst_quant.GetScale());
                        return true;
                    }

                    if ((!fakeQuantize && quantSibling->_dst_quant.IsScaleSet()) ||
                        (fakeQuantize && quantSibling->_dst_quant.IsScaleSet() && !fp32eq(quantSibling->_dst_quant.GetScale(), 1.0) &&
                        quantSibling->_dst_quant.GetScale() < inputQuant->_dst_quant.GetScale()) ||
                        quantSibling->_dst_quant.IsScaleSet() && infiniteLoopCount > 0) {
                        // means we already restarted propagation input memory layer
                        // need to search for requantiseable layer prior memory output layer
                        InferenceEngine::CNNLayerPtr restartedLayer;

                        gnalog() << "Memory layer :" << input->name << " scale factor: " << quantSibling->_dst_quant.GetScale()
                            << " doesn't match its outputs counterpart: " << cnnLayer->name << " scale factor: " << inputQuant->_dst_quant.GetScale() << "\n";
                        gnalog() << "[UFS] searching for quantizeable input layer for: " << cnnLayer->name << "\n";

                        CNNNetDFS(InferenceEngine::CNNLayerPtr(cnnLayer, [](InferenceEngine::CNNLayer*) {}),
                            [&restartedLayer, cnnLayer](InferenceEngine::CNNLayerPtr layer) {
                                gnalog() << "[UFS] from : " << cnnLayer->name << " reached: " << layer->name;
                                // found that direct input to concat is a indirect parent of align filter - so no link required
                                auto info = LayerInfo(layer);
                                if (!info.isWeightable() && !info.isActivation()) {
                                    gnalog() << "... skipped\n";
                                    return;
                                }
                                restartedLayer = layer;
                                gnalog() << "... OK,  need requantize\n";
                            }, true, [&restartedLayer, &cnnLayer](InferenceEngine::CNNLayer* from) {
                                // aborting UFS once found suitable layer
                                return make_upstream_order(restartedLayer == nullptr ? from : nullptr);
                            });

                        if (restartedLayer == nullptr) {
                            THROW_GNA_EXCEPTION << "cannot requantize input to " << cnnLayer->name;
                        }

                        auto quantDataForMemoryOutput = InferenceEngine::getInjectedData<QuantizedLayerParams>(*restartedLayer);

                        auto restarLayerInfo = LayerInfo(restartedLayer);
                        if (restarLayerInfo.isActivation()) {
                            // requantize activation by just changing it's output scale factor
                            quantDataForMemoryOutput->_dst_quant.SetScale(quantSibling->_dst_quant.GetScale());
                        } else {
                            THROW_GNA_EXCEPTION << "quantization error : input scale factor ( " << inputQuant->_dst_quant.GetScale() << ") "
                                << " for " << cnnLayer->name << ", that is child of " << prevLayer->name << " doesnt match : "
                                << activation_scale_factor;
                        }

                        result = ScaleFactorUpdateResult(restartedLayer.get());
                        return true;
                    }

                    gnawarn() << "[INFO] quantization : input scale factor (" << inputQuant->_dst_quant.GetScale() << ")"
                        << " for " << cnnLayer->name << ", that is child of " << prevLayer->name << " doesnt match : "
                        << activation_scale_factor << ", restarting from corresponding memory: " << input->name << std::endl;

                    // try updating memory input layer scale factor and restart from it
                    quantSibling->_src_quant.SetScale(inputQuant->_dst_quant.GetScale());
                    quantSibling->_dst_quant.SetScale(inputQuant->_dst_quant.GetScale());
                    result = ScaleFactorUpdateResult(input.get());
                    return true;
                }
            }
            return true;
        }

        if (cnnLayer->type == "Const") {
            if (quant->_dst_quant.IsScaleSet()) {
                quant->_src_quant = quant->_dst_quant;
                return true;
            }

            auto max_val = std::numeric_limits<float>::min();
            auto min_val = std::numeric_limits<float>::max();
            if (quant->_dst_quant.IsStatsSet()) {
                min_val = quant->_dst_quant.GetMinValues().front();
                max_val = quant->_dst_quant.GetMaxValues().front();
            } else {
                auto blob = cnnLayer->blobs["custom"];
                auto blob_precision = blob->getTensorDesc().getPrecision();

                if (blob_precision != InferenceEngine::Precision::FP32 && blob_precision != InferenceEngine::Precision::FP16) {
                    quant->_dst_quant.SetScale(1.0f);
                    return true;
                }

                if (blob_precision == InferenceEngine::Precision::FP16) {
                    blob = make_fp32_blob(blob);
                }

                auto flt_buf = blob->buffer().as<float*>();
                auto size = blob->size();

                for (int i = 0; i < size; i++) {
                    auto val = flt_buf[i];
                    if (val > max_val) max_val = val;
                    if (val < min_val) min_val = val;
                }
            }

            auto levels = 0;
            if (fakeQuantize) {
                levels = (inputsSize == 2) ? MAX_VAL_2B_FEAT : MAX_VAL_1B_FEAT;
            } else {
                levels = (inputsSize == 2) ? std::numeric_limits<int16_t>::max() : std::numeric_limits<int8_t>::max();
            }

            auto abs_val = std::max(std::abs(max_val), std::abs(min_val));
            auto scale_val = static_cast<float>(levels) / abs_val;
            //TODO: use FQ formula for scale factor calculation

            if (std::isinf(scale_val) || fp32eq(abs_val, 0.0f)) {
                quant->_dst_quant.SetScale(fakeQuantize ? levels : 1.0f);
            } else {
                quant->_dst_quant.SetScale(scale_val);
            }
            quant->_src_quant.SetScale(quant->_dst_quant.GetScale());

            return true;
        }

        if (!CNNNetHasPrevLayer(cnnLayer)) {
            quant->_dst_quant = quant->_src_quant;
            return true;
        }

        // by default layer is pass thru its scale factor
        auto inputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(CNNNetPrevLayer(cnnLayer));
        if (!inputQuant) {
            THROW_GNA_EXCEPTION << "layer: " << CNNNetPrevLayer(cnnLayer)->name << "not quantized";
        }

        if (layerInfo.isPower() && !layerInfo.isActivation()) {
            auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);
            auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer const*>(cnnLayer);
            if (!powerLayer) {
                IE_THROW() << "Incorrect Power Layer pointer \n";
            }

            auto powerScale = std::abs(powerLayer->scale);
            if (fp32eq(powerScale, 0.0f)) {
                powerScale = 1.0f;
            }
            auto weightsScaleFactor = MAX_VAL_2B_WEIGHT / powerScale;
            quant->_src_quant.SetScale(inputQuant->_dst_quant.GetScale());
            quant->_weights_quant.SetScale(weightsScaleFactor);
            quant->_dst_quant.SetScale(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale());
            return true;
        } else if (layerInfo.isActivation()) {
            // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
            // set the initial value
            if (!quant->_dst_quant.IsScaleSet() || fp32eq(quant->_dst_quant.GetScale(), 1.0f) ||
                !fp32eq(quant->_src_quant.GetScale(), inputQuant->_dst_quant.GetScale())) {
                quant->_src_quant.SetScale(inputQuant->_dst_quant.GetScale());
                auto scale = getActivationScale(cnnLayer, layerInfo, inputsSize, fakeQuantize);
                quant->_dst_quant.SetScale(scale);
            }
            return true;
        } else if (layerInfo.isCropAffined()) {
            auto weightsScaleFactor = 1;
            quant->_weights_quant.SetScale(weightsScaleFactor);
            quant->_src_quant.SetScale(inputQuant->_dst_quant.GetScale());
            quant->_dst_quant.SetScale(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale());
            return true;
        }
        quant->_src_quant.SetScale(inputQuant->_dst_quant.GetScale());
        quant->_dst_quant.SetScale(inputQuant->_dst_quant.GetScale());

        return true;
    }
};

template<typename QUANT_DESC>
class ScaleFactorPerLayer<InferenceEngine::EltwiseLayer*, QUANT_DESC> {
 public:
    bool operator()(InferenceEngine::EltwiseLayer* eltwiseLayer, ScaleFactorUpdateResult &result, int infiniteLoopCount) {
        if ( !eltwiseLayer ) {
            THROW_GNA_EXCEPTION << "Incorrect Eltwise Layer pointer \n";
        }
        int inputsSize = ScaleFactorCalculator<QUANT_DESC>::GetInputsBytesSize();
        bool fakeQuantize = ScaleFactorCalculator<QUANT_DESC>::IsFakeQuantize();
        bool lowPrecision = (inputsSize == sizeof(int8_t));

        auto in0 = InferenceEngine::CNNNetPrevLayer(eltwiseLayer, 0);
        auto in1 = InferenceEngine::CNNNetPrevLayer(eltwiseLayer, 1);

        auto quantParams0 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0);
        auto quantParams1 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in1);

        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*eltwiseLayer);

        switch (eltwiseLayer->_operation) {
            case InferenceEngine::EltwiseLayer::Prod: {
                quantData->_weights_quant.SetScale(quantParams1->_dst_quant.GetScale());
                quantData->_dst_quant.SetScale(
                            quantParams0->_dst_quant.GetScale() * quantParams1->_dst_quant.GetScale());
                break;
            }
            case InferenceEngine::EltwiseLayer::Sub:
            case InferenceEngine::EltwiseLayer::Sum: {
                // detect which input will be used as biases
                auto eltwiseFunctionalPrev =
                    LayerInfo(in0).isNonFunctional() && CNNNetHasPrevLayer(in0.get(), 0) ?
                    CNNNetPrevLayerSkipCertain(in0, 0, [](InferenceEngine::CNNLayerPtr l) {
                    return LayerInfo(l).isNonFunctional();
                        }) : in0;

                if (LayerInfo(in0).has32BOutput() ||
                    (LayerInfo(in0).isNonFunctional() && LayerInfo(eltwiseFunctionalPrev).has32BOutput())) {
                    std::swap(in0, in1);
                    std::swap(quantParams0, quantParams1);
                }

                auto prevLayer = in1;
                while (LayerInfo(prevLayer).isNonFunctional() && CNNNetHasPrevLayer(prevLayer.get(), 0)) {
                    prevLayer = CNNNetPrevLayer(prevLayer);
                }

                // this path might result in significant data loss
                quantData->_bias_quant.SetScale(quantParams1->_dst_quant.GetScale() / quantParams0->_dst_quant.GetScale());
                auto weightsScale = quantParams1->_dst_quant.GetScale() / quantParams0->_dst_quant.GetScale();

                // If a previous layer is a layer where freely weights scale factor can be selected,
                // try to find the scale factor that will allow to use integer as weights scale factor for eltwise
                // operation.
                // If the weights scale factor for eltwise sum/sub is not integer, it will cause accuracy degradation.
                if (fakeQuantize) {
                    auto prevLayerIn1 = CNNNetPrevLayer(in1);
                    if (LayerInfo(in1).isWeightableIdentity() &&
                        (prevLayerIn1 == nullptr || LayerInfo(prevLayerIn1).has8BOr16BOutput())) {
                        auto bestWeightsScale = 0.0f;
                        auto bestError = static_cast<float>(std::numeric_limits<int16_t>::max());
                        auto scaleIn0Dst = quantParams0->_dst_quant.GetScale();
                        auto scaleIn1Src = quantParams1->_src_quant.GetScale();
                        for (size_t i = MAX_VAL_2B_FEAT; i > 0; --i) {
                            auto scaleIn1Dst = i * scaleIn1Src;
                            auto eltwiseWeightsScale = scaleIn1Dst / scaleIn0Dst;
                            if (eltwiseWeightsScale < 1.0 || eltwiseWeightsScale > std::numeric_limits<int16_t>::max() - 1) {
                                continue;
                            }

                            auto error = std::abs(eltwiseWeightsScale - static_cast<int16_t>(eltwiseWeightsScale));
                            if (error < bestError) {
                                bestError = error;
                                bestWeightsScale = i;
                            }

                            if (fp32eq(error, 0.0f)) {
                                break;
                            }
                        }

                        if (bestWeightsScale > 0.0f && !fp32eq(bestWeightsScale, quantParams1->_weights_quant.GetScale())) {
                            quantParams1->_weights_quant.SetScale(bestWeightsScale);
                            quantParams1->_dst_quant.SetScale(quantParams1->_weights_quant.GetScale() * quantParams1->_src_quant.GetScale());
                            result = ScaleFactorUpdateResult(in1.get());
                            return true;
                        }
                    }
                }
                quantData->_weights_quant.SetScale(weightsScale);
                quantData->_dst_quant.SetScale(quantParams1->_dst_quant.GetScale());

                // eltwise will work in int16 or int8 if low precision inputs are used
                auto maxValue = lowPrecision ? std::numeric_limits<int8_t>::max() : std::numeric_limits<int16_t>::max();
                if (quantData->_weights_quant.GetScale() > maxValue &&
                    !fp32eq(quantData->_weights_quant.GetScale(), maxValue)) {
                    float newOutputScale = quantParams0->_dst_quant.GetScale() * maxValue;
                    gnalog() << "[INFO] weights saturated for " << eltwiseLayer->name << ", try to requiantize input " << in1->name << std::endl;
                    if (requantizeInput(in1, newOutputScale, result, infiniteLoopCount)) {
                        return true;
                    }
                    // we unable to rescale the input - results might be bad
                    gnawarn() << "[INFO] weights saturated for " << eltwiseLayer->name << "\n";
                }

                if (!quantData->_dst_quant.IsStatsSet()) {
                    return true;
                }

                auto weightsReducer = calculateWeightsReducerFromDstStats(quantData->_dst_quant);
                if (!fp32eq(weightsReducer, 1.0f)) {
                    float newOutputScale = quantParams1->_dst_quant.GetScale() / weightsReducer;
                    if (requantizeInput(in1, newOutputScale, result, infiniteLoopCount)) {
                        return true;
                    }
                    THROW_GNA_EXCEPTION << "Unable to quantize " << eltwiseLayer->name;
                }
            }
            break;
            default : THROW_GNA_EXCEPTION << "Unsupported Eltwise layer for quantisation: " << eltwiseLayer->_operation;
        }
        return true;
    }
};

template<typename QUANT_DESC>
class ScaleFactorPerLayer<InferenceEngine::ConcatLayer*, QUANT_DESC> {
 public:
    bool operator()(InferenceEngine::ConcatLayer* concatLayer, ScaleFactorUpdateResult &result, int infiniteLoopCount) {
        if ( !concatLayer ) {
            THROW_GNA_EXCEPTION << "Incorrect Concat Layer pointer \n";
        }

        bool fakeQuantize = ScaleFactorCalculator<QUANT_DESC>::IsFakeQuantize();

        if (concatLayer->insData.size() < 2) {
            THROW_GNA_EXCEPTION << "Concat layer has unsupported number of incoming layers.";
        }

        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*concatLayer);
        std::vector<InferenceEngine::CNNLayerPtr> inputLayers;
        for (auto input_idx = 0; input_idx != concatLayer->insData.size(); input_idx++) {
            auto notChangeScaleFactors = [](InferenceEngine::CNNLayerPtr layer) {
                return LayerInfo(layer).isNonFunctional() || LayerInfo(layer).isSplit() || LayerInfo(layer).isCopy();
            };
            auto prev_layer = CNNNetPrevLayerSkipCertain(concatLayer, input_idx, notChangeScaleFactors);
            inputLayers.push_back(prev_layer);
        }

        // if all inputs have same quant value - trivial propagation
        auto in0 = inputLayers.front();
        auto quantParams0 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0);
        auto scaleFactor = quantParams0->_dst_quant.GetScale();
        auto scaleFactorCheck = [scaleFactor](InferenceEngine::CNNLayerPtr& inputLayer) {
            auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            return fp32eq(quantParams->_dst_quant.GetScale(), scaleFactor);
        };

        if (std::find_if_not(inputLayers.begin() + 1, inputLayers.end(), scaleFactorCheck) == inputLayers.end()) {
            quantData->_dst_quant.SetScale(quantParams0->_dst_quant.GetScale());
            quantData->_src_quant.SetScale(quantParams0->_dst_quant.GetScale());
            return true;
        }

        // verify if all "Input"-type inputs have the same quant value
        auto inputLayerCheck = [](InferenceEngine::CNNLayerPtr& inputLayer) {
            auto info = LayerInfo(inputLayer);
            return info.isInput();
        };

        GNAPluginNS::QuantizedLayerParams* sourceQuantParams = nullptr;
        auto sourceLayerIt = std::find_if(inputLayers.begin(), inputLayers.end(), inputLayerCheck);
        if (sourceLayerIt != inputLayers.end()) {
            auto quantParamsFirst = InferenceEngine::getInjectedData<QuantizedLayerParams>(*sourceLayerIt);
            auto nextInputIt = sourceLayerIt + 1;
            while ((nextInputIt = std::find_if(nextInputIt, inputLayers.end(), inputLayerCheck)) != inputLayers.end()) {
                auto quantParamsSecond = InferenceEngine::getInjectedData<QuantizedLayerParams>(*nextInputIt);
                if (!fp32eq(quantParamsSecond->_dst_quant.GetScale(), quantParamsFirst->_dst_quant.GetScale())) {
                    THROW_GNA_EXCEPTION << "Two Input layers " << (*sourceLayerIt)->name
                        << " and " << (*nextInputIt)->name << " have different scales in concat!!! \n";
                }
                ++nextInputIt;
            }
        }

        // find a source quant value
        // - 1st candidate - input layer
        // - 2nd candidate - non-activation layer with non-1 scale factor
        if (sourceLayerIt == inputLayers.end()) {
            if (infiniteLoopCount % 2 == 1) {
                std::reverse(inputLayers.begin(), inputLayers.end());
            }

            if (fakeQuantize) {
                sourceLayerIt = inputLayers.begin();
                auto quantParamsFirst = InferenceEngine::getInjectedData<QuantizedLayerParams>(*inputLayers.begin());
                auto minScaleFactor = quantParamsFirst->_dst_quant.GetScale();
                for (auto it = inputLayers.begin(); it != inputLayers.end(); ++it) {
                    auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*it);
                    if ((quantParams->_dst_quant.GetScale() < minScaleFactor &&
                         !fp32eq(quantParams->_dst_quant.GetScale(), 1.0f)) ||
                        fp32eq(minScaleFactor, 1.0f)) {
                        minScaleFactor = quantParams->_dst_quant.GetScale();
                        sourceLayerIt = it;
                    }
                }
            } else {
                if (infiniteLoopCount % 4 == 2 || infiniteLoopCount % 4 == 3) {
                    auto sourceLayerCheck = [](InferenceEngine::CNNLayerPtr& inputLayer) {
                        auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
                        LayerInfo info(inputLayer);
                        return !info.isActivation() && !fp32eq(quantParams->_dst_quant.GetScale(), 1.0f);
                    };
                    sourceLayerIt = std::find_if(inputLayers.begin(), inputLayers.end(), sourceLayerCheck);
                }

                if (sourceLayerIt == inputLayers.end()) {
                    auto nonDefaultScaleFactor = [](InferenceEngine::CNNLayerPtr& inputLayer) {
                        auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
                        return !fp32eq(quantParams->_dst_quant.GetScale(), 1.0f);
                    };

                    sourceLayerIt = std::find_if(inputLayers.begin(), inputLayers.end(), nonDefaultScaleFactor);
                }
            }
        }

        std::set<size_t> concatIdxToUpdate;
        if (sourceLayerIt != inputLayers.end()) {
            auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*sourceLayerIt);
            auto scaleFactor = quantParams->_dst_quant.GetScale();
            sourceQuantParams = quantParams;

            for (auto it = inputLayers.begin(); it != inputLayers.end(); ++it) {
                auto quantParamsIn = InferenceEngine::getInjectedData<QuantizedLayerParams>(*it);
                if (fp32eq(quantParamsIn->_dst_quant.GetScale(), scaleFactor)) {
                    continue;
                }

                if (fakeQuantize) {
                    concatIdxToUpdate.insert(std::distance(inputLayers.begin(), it));
                    quantParamsIn->_dst_quant.SetScale(quantParams->_dst_quant.GetScale());
                } else {
                    // possible case when some of the concat inputs are free to select scale ex: const->concat<-affine
                    if (!fp32eq(quantParamsIn->_dst_quant.GetScale(), 1.0f) && !LayerInfo(*it).isActivation()) {
                        concatIdxToUpdate.insert(std::distance(inputLayers.begin(), it));
                    }

                    quantParamsIn->_dst_quant.SetScale(quantParams->_dst_quant.GetScale());
                }
            }
        }

        auto updatedScaleFactor = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0)->_dst_quant.GetScale();
        auto equalScaleFactor = [updatedScaleFactor](InferenceEngine::CNNLayerPtr& inputLayer) {
            auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            return fp32eq(quantParams->_dst_quant.GetScale(), updatedScaleFactor);
        };

        auto layerIt = std::find_if_not(inputLayers.begin() + 1, inputLayers.end(), equalScaleFactor);
        if (layerIt != inputLayers.end()) {
            THROW_GNA_EXCEPTION << "layers entered into concat have different scale factors. Layer name: " << concatLayer->name;
        }

        if (sourceQuantParams == nullptr) {
            THROW_GNA_EXCEPTION << "Source quantization parameters have not been initialized";
        }

        quantData->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
        quantData->_src_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());

        if (layerIt == inputLayers.end() && concatIdxToUpdate.empty()) {
            return true;
        }

        for (auto& layerIdToUpdate : concatIdxToUpdate) {
            auto destinationQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*concatLayer);
            destinationQuantParams->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());

            InferenceEngine::CNNLayerPtr restartedLayer;
            // making a link activation possible without extra layer if first input to concat not a parent / indirect parent of second input
            // using ufs - upper first search
            gnalog() << "[UFS] searching for quantizeable layer prior: " << concatLayer->name << ", via " << layerIdToUpdate << "\n";

            CNNNetDFS(InferenceEngine::CNNLayerPtr(concatLayer, [](InferenceEngine::CNNLayer*) {}),
                [&restartedLayer, concatLayer](InferenceEngine::CNNLayerPtr layer) {
                    gnalog() << "[UFS] from : " << concatLayer->name << " reached: " << layer->name;
                    // found that direct input to concat is a indirect parent of align filter - so no link required
                    auto info = LayerInfo(layer);
                    if (!info.isWeightable() && !info.isActivation() && !info.isConst() && !info.isMemory()) {
                        gnalog() << "... skipped\n";
                        return;
                    }
                    restartedLayer = layer;
                    gnalog() << "... OK,  need requantize\n";
                }, true, [&restartedLayer, &concatLayer, &layerIdToUpdate](InferenceEngine::CNNLayer* from) {
                    // aborting UFS once found functional layer, and using only specified input of concat
                    return make_upstream_order(restartedLayer == nullptr ? from : nullptr,
                        from == concatLayer ? layerIdToUpdate : -1);
                });

            if (restartedLayer == nullptr) {
                THROW_GNA_EXCEPTION << "cannot requantize " << layerIdToUpdate << " input to concat: " << concatLayer->name;
            }
            auto quantDataForConCatInput = InferenceEngine::getInjectedData<QuantizedLayerParams>(*restartedLayer);

            auto restarLayerInfo = LayerInfo(restartedLayer);
            if (restarLayerInfo.isActivation()) {
                // requantize activation by just changing it's output scale factor
                auto newScaleFactor = sourceQuantParams->_dst_quant.GetScale();
                auto skipNonFunctional = [](InferenceEngine::CNNLayerPtr l) {
                    return LayerInfo(l).isNonFunctional();
                };

                auto prevLayer = CNNNetPrevLayerSkipCertain(restartedLayer, 0, skipNonFunctional);
                auto prevLayer2 = prevLayer != nullptr ? CNNNetPrevLayerSkipCertain(prevLayer, 0, skipNonFunctional) : nullptr;

                if (fakeQuantize && prevLayer != nullptr &&
                    (LayerInfo(prevLayer).isConcatAlignFilter() || LayerInfo(prevLayer).isSyntheticScaleShift()) &&
                    (prevLayer2 == nullptr || LayerInfo(prevLayer2).has8BOr16BOutput())) {
                    auto weightsScales = generateScaleFactors(MIN_SEARCH_WEIGHTS_VAL, MAX_SEARCH_WEIGHTS_VAL,
                        MAX_SEARCH_WEIGHTS_VAL - MIN_SEARCH_WEIGHTS_VAL);

                    auto prevLayerQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
                    auto bestWeightsScale = 1.0f;
                    auto slopes = getPWLSlopes(restarLayerInfo);
                    if (!slopes.empty() && !fp32eq(prevLayerQuant->_src_quant.GetScale(), newScaleFactor)) {
                        bestWeightsScale = selectBestWeightsScaleFactors(prevLayerQuant->_src_quant.GetScale(),
                            newScaleFactor, weightsScales, { 1.0f });
                    }
                    if (!slopes.empty() && !fp32eq(bestWeightsScale, prevLayerQuant->_weights_quant.GetScale())) {
                        gnalog() << "[INFO][Concat] Optimizing weights scale factor for '" << prevLayer->name << "' layer. Change from "
                            << prevLayerQuant->_weights_quant.GetScale() << " to " << bestWeightsScale << "\n";

                        prevLayerQuant->_weights_quant.SetScale(bestWeightsScale);
                        prevLayerQuant->_dst_quant.SetScale(prevLayerQuant->_weights_quant.GetScale() * prevLayerQuant->_src_quant.GetScale());
                        result = ScaleFactorUpdateResult(prevLayer.get());
                        return true;
                    }
                }

                quantDataForConCatInput->_dst_quant.SetScale(newScaleFactor);
            } else if (restarLayerInfo.isConst() || restarLayerInfo.isMemory()) {
                gnalog() << "... warning " << restartedLayer->type << " layer will be requantized\n";
                quantDataForConCatInput->_src_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
                quantDataForConCatInput->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
            } else {
                THROW_GNA_EXCEPTION << "cannot requantize '" << restartedLayer->name << "' input to concat: " << concatLayer->name;
            }
            result = ScaleFactorUpdateResult(restartedLayer.get());
        }

        return true;
    }
};

template<typename QUANT_DESC>
class ScaleFactorPerLayer<InferenceEngine::WeightableLayer*, QUANT_DESC> {
 private:
    std::vector<std::tuple<uint16_t const, float const, float const>> thresholds = {
        // tuple values: scale factor threshold, scale factor reduction factor for I16 precision, for I8 precision
        std::make_tuple(30, 0.50f, 0.50f),     // entry check value
        std::make_tuple(100, 0.50f, 0.50f),    // if below this threshold, then use this factor
        std::make_tuple(150, 0.45f, 0.45f),
        std::make_tuple(200, 0.40f, 0.40f),
        std::make_tuple(200, 0.35f, 0.35f)     // max level -> if above, then use this factor
    };

 public:
    bool operator()(InferenceEngine::WeightableLayer *wl, ScaleFactorUpdateResult &result, int infiniteLoopCount) {
        if ( !wl ) {
            THROW_GNA_EXCEPTION << "Incorrect Weightable Layer pointer  \n";
        } else if (!wl->_weights) {
            THROW_GNA_EXCEPTION << "Incorrect weight value for " << wl->name << ":" << wl->type << "\n";
        }

        int inputsSize = ScaleFactorCalculator<QUANT_DESC>::GetInputsBytesSize();
        bool fakeQuantize = ScaleFactorCalculator<QUANT_DESC>::IsFakeQuantize();
        auto prevLayer = CNNNetPrevLayer(wl);
        auto quantDataForInputLayer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(*InferenceEngine::CNNNetPrevLayer(wl).get());

        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*wl);
        quant->_src_quant = quantDataForInputLayer->_dst_quant;
        if (quant->_weights_quant.IsStatsSet() && !quant->_weights_quant.IsScaleSet()) {
            auto getScale = [&quant](size_t i) {
                return CalculateScaleFactorFromStats(quant->_weights_quant.GetLevels(),
                    quant->_weights_quant.GetMinValues(false)[i], quant->_weights_quant.GetMaxValues(false)[i]);
            };

            float min_channel_scale = getScale(0);
            for (uint32_t i = 1; i < quant->_weights_quant.GetMinValues().size(); i++) {
                min_channel_scale = std::min(min_channel_scale, getScale(i));
            }

            auto multiplier = 1.0f;
            if (quant->_weights_quant.GetLevels() <= std::numeric_limits<uint8_t>::max()) {
                // GNA supports additional multiplier for only 8bit weights.
                // The multipler is used to extend dynamic range.
                multiplier = MAX_OUT_MULTIPLIER;
            }

            // Common weights scale calculation
            quant->_weights_quant.SetScale(min_channel_scale * multiplier);
        }

        // TODO: pass 8 bits somehow
        int weightsSize = ScaleFactorCalculator<QUANT_DESC>::GetMandatoryWeightsBytesSize(wl);
        if (!quant->_weights_quant.IsScaleSet()) {
            size_t scaleRange = 0;
            if (weightsSize == 2) {
                scaleRange = MAX_VAL_2B_WEIGHT;
            } else if (weightsSize == 1) {
                scaleRange = MAX_VAL_1B_WEIGHT;
            } else {
                THROW_GNA_EXCEPTION << "Unsupported weights size of: " << weightsSize;
            }
            quant->_weights_quant.SetScale(
                ScaleFactorForQuantization(wl->_weights->buffer().as<float *>(), scaleRange, wl->_weights->size()));
            if (quant->_weights_quant.GetScale() == -1.0f || (fakeQuantize && LayerInfo(wl).isConcatAlignFilter())) {
                quant->_weights_quant.SetScale(1.0f);
            }

            if (wl->_biases) {
                // for now the only case of INT8 bias we support comes with INT8 inputs and weights as well
                if (inputsSize == 1 && weightsSize == 1) {
                    quant->_bias_quant.SetScale(ScaleFactorForQuantization(wl->_biases->buffer().as<float*>(),
                        MAX_VAL_1B_BIAS,
                        wl->_biases->size()));
                } else {
                    quant->_bias_quant.SetScale(ScaleFactorForQuantization(wl->_biases->buffer().as<float*>(),
                        MAX_VAL_4B_BIAS,
                        wl->_biases->size()));
                }
                if (quant->_bias_quant.GetScale() != -1.0f) {
                    // for low precision we don't change bias scale factor based on source and weights scale factors
                    // in order not to loose too much precision
                    if (inputsSize != 1 || weightsSize != 1) {
                        quant->_bias_quant.SetScale(
                            std::min(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale(), quant->_bias_quant.GetScale()));
                    }
                    quant->_weights_quant.SetScale(quant->_bias_quant.GetScale() / quant->_src_quant.GetScale());
                }
            }

            // use the MAX_OUT_MULTIPLIER only for int8_t weigths with compound bias (for now handled here only with int16_t inputs)
            // it gives the possibility to exetend the output dynamic range
            if (weightsSize == 1 && inputsSize == 2) {
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * MAX_OUT_MULTIPLIER);
            }

            double weights_reducer = 1.0;
            auto conv = dynamic_cast<InferenceEngine::ConvolutionLayer *>(wl);
            if (conv && !LayerInfo(conv).isConvolutionFilter()) {
                const auto inDepth = GetDataDimSize(conv->insData.front().lock(), InferenceEngine::DataDimName::C);
                weights_reducer = GNAConvolutionLayer::getWeightsReducer(*conv);
                weights_reducer *= MAX_VAL_2B_FEAT * scaleRange * inDepth / std::numeric_limits<int32_t>::max();
                weights_reducer = std::max(1.0, weights_reducer);
            }
            quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() / weights_reducer);
        }
        double tmp_dst_quant_scale = quant->_weights_quant.GetScale() * quant->_src_quant.GetScale();
        if (weightsSize == 1) {
            auto itt = thresholds.begin();
            auto limit = std::numeric_limits<int32_t>::max();

            if (inputsSize == 1) {
                limit = std::numeric_limits<int8_t>::max();
            }

            if (static_cast<uint64_t>(tmp_dst_quant_scale * quant->_src_quant.GetScale()) >
                static_cast<uint64_t>(limit - 1) * std::get<0>(*itt)) {
                gnawarn() << "Output scale for " << wl->name
                    << " too large and are being reduced. Else saturations likely will happen \n";
                // reduce weight scale according experimental heuristic
                while ((itt + 1) != thresholds.end() && quant->_dst_quant.GetScale() * quant->_src_quant.GetScale() /
                    static_cast<float>(limit) >= std::get<0>(*(++itt))) {}
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * (inputsSize == 2 ? std::get<1>(*itt) : std::get<2>(*itt)));
            }
        }

        quant->_dst_quant.SetScale(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale());
        if (quant->_dst_quant.IsStatsSet()) {
            // Adjust weights scale factor if output values exceed int32 maximum value

            if (wl->_biases && !quant->_bias_quant.IsScaleSet()) {
                auto minMax = FindMinMaxValues(wl->_biases->buffer().as<float*>(), wl->_biases->size());
                quant->_bias_quant.SetMinValues({ minMax.first });
                quant->_bias_quant.SetMaxValues({ minMax.second });

                auto biasScale = ScaleFactorForQuantization(wl->_biases->buffer().as<float*>(), MAX_VAL_4B_BIAS, wl->_biases->size());
                quant->_bias_quant.SetScale(biasScale);
                if (quant->_bias_quant.GetScale() != -1.0f && quant->_bias_quant.GetScale() < quant->_dst_quant.GetScale()) {
                    quant->_weights_quant.SetScale(quant->_bias_quant.GetScale() / quant->_src_quant.GetScale());
                    quant->_dst_quant.SetScale(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale());
                }
            }

            auto weightsReducer = calculateWeightsReducerFromDstStats(quant->_dst_quant);
            if (!fp32eq(weightsReducer, 1.0f)) {
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() / weightsReducer);
            }

            if (fp32eq(quant->_weights_quant.GetScale(), 0.0f) || std::isinf(quant->_weights_quant.GetScale())) {
                quant->_weights_quant.SetScale(1.0f);
            }

            quant->_dst_quant.SetScale(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale());
        }

        return true;
    }
};

template<typename QUANT_DESC>
class ScaleFactorPerLayer<InferenceEngine::ScaleShiftLayer*, QUANT_DESC> :
    public ScaleFactorPerLayer<InferenceEngine::WeightableLayer*, QUANT_DESC> {
};

template<typename QUANT_DESC>
class ScaleFactorPerLayer<InferenceEngine::ConvolutionLayer*, QUANT_DESC> :
    public ScaleFactorPerLayer<InferenceEngine::WeightableLayer*, QUANT_DESC> {
};

template<typename QUANT_DESC>
class ScaleFactorPerLayer<InferenceEngine::GemmLayer*, QUANT_DESC> {
public:
    bool operator() (InferenceEngine::GemmLayer* gemmLayer, ScaleFactorUpdateResult &result, int infiniteLoopCount) {
        if ( !gemmLayer ) {
            THROW_GNA_EXCEPTION << "Incorrect Gemm Layer pointer \n";
        }
        auto in0 = InferenceEngine::CNNNetPrevLayer(gemmLayer, 0);
        auto in1 = InferenceEngine::CNNNetPrevLayer(gemmLayer, 1);

        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*gemmLayer);
        auto quantParams1 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in1);
        auto quantParams0 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0);
        quantData->_src_quant.SetScale(quantParams0->_dst_quant.GetScale());
        quantData->_weights_quant.SetScale(quantParams1->_dst_quant.GetScale());
        quantData->_dst_quant.SetScale(
                quantData->_src_quant.GetScale() * quantData->_weights_quant.GetScale());

        if (!quantData->_dst_quant.IsStatsSet()) {
            return true;
        }

        // Adjust weights scale factor if output values exceed int32 maximum value
        auto weightsReducer = calculateWeightsReducerFromDstStats(quantData->_dst_quant);
        if (LayerInfo(in0).isConst()) {
            if (!fp32eq(weightsReducer, 1.0f)) {
                quantParams0->_dst_quant.SetScale(quantData->_src_quant.GetScale() / weightsReducer);
                quantData->_src_quant.SetScale(quantData->_src_quant.GetScale() / weightsReducer);
            }
            if (fp32eq(quantData->_src_quant.GetScale(), 0.0f) || std::isinf(quantData->_src_quant.GetScale())) {
                quantParams0->_dst_quant.SetScale(1.0f);
                quantData->_src_quant.SetScale(1.0f);
            }

            quantData->_dst_quant.SetScale(quantData->_weights_quant.GetScale() * quantData->_src_quant.GetScale());
        } else {
            if (!fp32eq(weightsReducer, 1.0f)) {
                for (int i = 0; i < 2; ++i) {
                    auto input = InferenceEngine::CNNNetPrevLayer(gemmLayer, i);
                    auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(input);
                    float newOutputScale = quantParams->_dst_quant.GetScale() / weightsReducer;
                    if (requantizeInput(input, newOutputScale, result, infiniteLoopCount)) {
                        return true;
                    }
                }
                THROW_GNA_EXCEPTION << "Unable to quantize " << gemmLayer->name;
            }
        }
        return true;
    }
};

}  // namespace frontend

/**
 * @brief scale factor calculator will calculate only output scale factors for the layer
 * if scale factor propagation not possible, it will fall indicate a restart condition
 */
template<typename QUANT_DESC>
class ScaleFactorCalculator {
    using Cnt = std::vector<InferenceEngine::CNNLayerPtr>;
    Cnt  net;
    mutable Cnt::const_iterator idx;
    mutable bool needRestart = false;
    int infiniteLoopCount = 0;

 public:
    ScaleFactorCalculator(Cnt &net) : net(net) {
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
    template<class T>
    bool operator()(T ptr) const {
        needRestart = false;
        frontend::ScaleFactorUpdateResult result;
        if (!frontend::ScaleFactorPerLayer<T, QUANT_DESC>()(ptr, result, infiniteLoopCount)) {
            return false;
        }
        if (result) {
            idx++;
            return true;
        }

        idx = std::find_if(net.begin(), net.end(), [&](InferenceEngine::CNNLayerPtr cnnLayer) {
            if (!result) {
                return result.restartLayer == cnnLayer.get();
            }
            return ptr == cnnLayer.get();
        });
        if (idx != net.end()) {
            idx++;
        }
        needRestart = true;
        return true;
    }

    template<class T>
    static int GetMandatoryWeightsBytesSize(T ptr) {
        auto info = LayerInfo(ptr);
        if (info.isConvolution() || info.isScaleShift()) {
            return GetOptionalWeightsBytesSize();
        }

        if (IsFakeQuantize()) {
            auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*ptr);
            if (quantData->_weights_quant.IsStatsSet()) {
                if (quantData->_weights_quant.GetLevels() <= std::numeric_limits<uint8_t>::max()) {
                    return frontend::FakeQuantI8().getWeightsPrecision().size();
                } else {
                    return frontend::FakeQuantI16().getWeightsPrecision().size();
                }
            } else {
                if (!info.isSynthetic()) {
                    gnawarn() << "The layer (" << ptr->name << ") has not quantization statistics\n";
                }

                return GetOptionalWeightsBytesSize();
            }
        }

        return QUANT_DESC::mandatory().getWeightsPrecision().size();
    }

    static int GetOptionalWeightsBytesSize() {
        return QUANT_DESC::optional().getWeightsPrecision().size();
    }

    static int GetInputsBytesSize() {
        return QUANT_DESC::mandatory().getInputPrecision().size();
    }

    static bool IsFakeQuantize() {
        return std::is_same<QUANT_DESC, FakeQuant>();
    }
}; // class ScaleFactorCalculator

}  // namespace GNAPluginNS
