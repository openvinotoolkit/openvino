// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <string>
#include <map>

#include <legacy/ie_layers.h>
#include "gna_upstream_iterator.hpp"
#include "layers/gna_layer_info.hpp"
#include "gna_plugin_log.hpp"
#include "gna_slope_scale.h"

namespace GNAPluginNS {
namespace frontend {
struct ScaleFactorUpdateResult {
    InferenceEngine::CNNLayer *restartLayer = nullptr;
    ScaleFactorUpdateResult() = default;
    explicit ScaleFactorUpdateResult(InferenceEngine::CNNLayer * restartlayer) : restartLayer(restartlayer) {
    }
    operator bool() {
        return restartLayer == nullptr;
    }
};

static const float DEFAULT_ACTIVATION_SCALE_FACTOR = 2048.0f;
static const float ID_SCALE_FACTOR = 2049.0f;

/**
* @brief Function to compare two floating points with epsilon
* @param p1 First floating point value to compare
* @param p2 Second floating point value to compare
* @return Returns true if two floating points are equal
*/
static bool fp32eq(float p1, float p2) {
    return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
}

/**
* @brief Function generates scale factors within given range
* @param startRange Start scale factor
* @param startRange End scale factor
* @param numScaleFactors Number of scale factors to generate
* @return Returns list of scale factors
*/
static std::vector<float> generateScaleFactors(float startRange, float endRange, size_t numScaleFactors) {
    auto scaleFactors = std::vector<float>{};
    auto domain = endRange - startRange;
    auto step = domain / numScaleFactors;
    for (size_t i = 0; i <= numScaleFactors; ++i) {
        auto scale = floorf(startRange + step * i + 0.5f);
        if (!std::isnan(scale)) {
            scaleFactors.push_back(scale);
        }
    }

    return scaleFactors;
}

/**
* @brief Function verifies if GNA PWL slope is accurate for given set of scale factors
* @param inScale Input scale factor
* @param outScale Output scale factor
* @param maxError Maximum % of error
* @param slopes List of GNA PWL slopes
* @return Returns true if the encoded slope is within maximum error
*/
static float checkOutputScaleFactors(float inScale, float outScale, float maxError, const std::vector<double>& slopes) {
    auto sd = 0.0;
    for (size_t j = 0; j < slopes.size(); ++j) {
        auto s = gna_slope(slopes[j], inScale, outScale);
        auto slope = static_cast<uint32_t>(s.slope * s.slope_scale);
        if (slope < std::numeric_limits<int16_t>::min() && slope > std::numeric_limits<int16_t>::max()) {
            sd += std::numeric_limits<int8_t>::max();
            continue;
        }

        auto testSlope = static_cast<double>(slope) / s.slope_scale * inScale / outScale;
        if (std::abs(testSlope - slopes[j]) > std::abs(maxError * slopes[j])) {
            return false;
        }
    }

    return true;
}

/**
* @brief Function finds the weights scale factor that gives the most precise GNA PWL slope
* @param inScale Input scale factor
* @param outScale Output scale factor
* @param weightsScales List of weights to verify
* @param slopes List of GNA PWL slopes
* @return Returns weights scale factor with the smallest error
*/
static float selectBestWeightsScaleFactors(float inScale, float outScale, std::vector<float> weightsScales, const std::vector<double>& slopes) {
    std::vector<float> scaleErrors;
    for (size_t i = 0; i < weightsScales.size(); ++i) {
        auto weightScale = weightsScales[i];

        auto sd = 0.0;
        for (size_t j = 0; j < slopes.size(); ++j) {
            auto s = gna_slope(slopes[j], inScale * weightScale, outScale);
            auto slope = static_cast<uint32_t>(s.slope * s.slope_scale);
            if (slope < std::numeric_limits<int16_t>::min() && slope > std::numeric_limits<int16_t>::max()) {
                sd += std::numeric_limits<int8_t>::max();
                continue;
            }

            auto testSlope = static_cast<double>(slope) / s.slope_scale * (inScale * weightScale) / outScale;
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
 * @brief calculates output scale factor per layer
 * @tparam T
 */
template<class T>
class ScaleFactorPerLayer {
 public:
    /**
     * @brief calculates weights scale factor for fit dynamic range into target bitsize,
     * also calculates output scale factor for the given layer
     * @param cnnLayer
     * @param weightsSize
     * @param result
     * @return
     */
    bool operator()(T cnnLayer, int weightsSize, ScaleFactorUpdateResult &result) {
        return false;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::CNNLayer *> {
 private :
    const float k = 5;
    const float k_identity = 6;
    const double pow_domain = 16;

 protected :
    float getActivationScale(InferenceEngine::CNNLayer const* cnnLayer,
                             GNAPluginNS::LayerInfo const& layer) {
        auto quantizedParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);

        // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
        // set the initial value
        float result = DEFAULT_ACTIVATION_SCALE_FACTOR;
        if (layer.isIdentity()) {
            auto prevLayer = CNNNetPrevLayer(cnnLayer);
            bool noChange = false;
            while (prevLayer != nullptr && (
                LayerInfo(prevLayer).isWeightable() ||
                LayerInfo(prevLayer).isEltwiseMul() ||
                LayerInfo(prevLayer).isCrop() ||
                LayerInfo(prevLayer).isNonFunctional())) {
                if (LayerInfo(prevLayer).isConcatAlignFilter() ||
                    prevLayer->name.find("SyntheticScaleShift") != std::string::npos ||
                    LayerInfo(prevLayer).isCrop()) {
                    noChange = true;
                }
                prevLayer = CNNNetPrevLayer(prevLayer);
            }

            if (prevLayer != nullptr && (LayerInfo(prevLayer).has16BOutput())) {
                auto prevQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
                result = fp32eq(prevQuantParams->_dst_quant.GetScale(), 1.0f)? DEFAULT_ACTIVATION_SCALE_FACTOR : prevQuantParams->_dst_quant.GetScale();
            }

            std::vector<float> scaleFactors;

            if (noChange) {
                scaleFactors.push_back(result);
            }
            scaleFactors.push_back(ID_SCALE_FACTOR);
            scaleFactors.push_back(DEFAULT_ACTIVATION_SCALE_FACTOR);
            scaleFactors.push_back(result);
            scaleFactors.push_back(DEFAULT_ACTIVATION_SCALE_FACTOR / 2);
            scaleFactors.push_back(ID_SCALE_FACTOR / 2);

            for (auto& scale : scaleFactors) {
                if (checkOutputScaleFactors(quantizedParams->_src_quant.GetScale(), scale, 0.001f, { 1.0f })) {
                    result = scale;
                    break;
                }
            }
        } else if (layer.isRelu() &&
                static_cast<uint64_t>(DEFAULT_ACTIVATION_SCALE_FACTOR * quantizedParams->_src_quant.GetScale())
                                                            > std::numeric_limits<int32_t>::max()-1) {
            // if activation is one from relu family, we need to apply heuristic to avoid activation output overflow
            result = (DEFAULT_ACTIVATION_SCALE_FACTOR * 0.5);
        } else if (layer.isPower()) {
            auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer const*>(cnnLayer);
            if (!powerLayer) {
                THROW_IE_EXCEPTION << "Incorrect Power Layer pointer \n";
            }

            auto input_min_value = static_cast<double>(std::numeric_limits<int32_t>::min());
            auto input_max_value = static_cast<double>(std::numeric_limits<int32_t>::max());
            auto output_max_value = static_cast<double>(std::numeric_limits<int16_t>::max());

            auto x_min = fp32eq(fmod(powerLayer->power, 1.0), 0) ? input_min_value / quantizedParams->_src_quant.GetScale() : 0.0;
            x_min = std::max(x_min, -pow_domain);

            auto x_max = input_max_value / quantizedParams->_src_quant.GetScale();
            x_max = std::min(x_max, pow_domain);

            auto val1 = pow(x_min * powerLayer->scale + powerLayer->offset, powerLayer->power);
            auto val2 = pow(x_max * powerLayer->scale + powerLayer->offset, powerLayer->power);

            auto abs_val = std::max(std::abs(val1), std::abs(val2));
            auto scale_val = output_max_value / abs_val;

            if (!std::isinf(scale_val)) {
                result = scale_val;
            }
        }

        //search for previous activation layer
        if (!quantizedParams->_dst_quant.GetMaxValues().empty()) {
            auto min_value = quantizedParams->_dst_quant.GetMinValues().front();
            auto max_value = quantizedParams->_dst_quant.GetMaxValues().front();
            auto newScaleFactor = (quantizedParams->_dst_quant.GetLevels() - 1) / (max_value - min_value);
            result = newScaleFactor < result ? newScaleFactor : result;
        }

        return result;
    }

 public :
    bool operator()(InferenceEngine::CNNLayer *cnnLayer, int weightsSize, ScaleFactorUpdateResult &result) {
        if ( !cnnLayer ) {
            THROW_IE_EXCEPTION << "Incorrect Convolutional Layer pointer \n";
        }
        LayerInfo layerInfo(*cnnLayer);
        // TODO: current approach set input scale factor for true input layer(s) equals to provided factor,
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);

        if (InferenceEngine::details::CaselessEq<std::string>()(cnnLayer->type, "Memory")) {
            if (!CNNNetHasPrevLayer(cnnLayer) && quant->_dst_quant.IsScaleSet()) {
                quant->_src_quant = quant->_dst_quant;
            }

            if (CNNNetHasPrevLayer(cnnLayer)) {
                auto prevLayer = CNNNetPrevLayer(cnnLayer);
                auto prevInfo = LayerInfo(prevLayer);
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

                    if (quantSibling->_dst_quant.IsScaleSet()) {
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
                                << DEFAULT_ACTIVATION_SCALE_FACTOR;
                        }

                        result = ScaleFactorUpdateResult(restartedLayer.get());
                        return true;
                    }

                    gnawarn() << "[INFO] quantization : input scale factor (" << inputQuant->_dst_quant.GetScale() << ")"
                        << " for " << cnnLayer->name << ", that is child of " << prevLayer->name << " doesnt match : "
                        << DEFAULT_ACTIVATION_SCALE_FACTOR << ", restarting from corresponding memory: " << input->name << std::endl;

                    // try updating memory input layer scale factor and restart from it
                    quantSibling->_src_quant = quantSibling->_dst_quant = inputQuant->_dst_quant;
                    result = ScaleFactorUpdateResult(input.get());
                    return true;
                }
            }
            return true;
        }

        if (cnnLayer->type == "Const") {
            if (quant->_dst_quant.IsScaleSet()) {
                quant->_src_quant = quant->_dst_quant;
                return ScaleFactorUpdateResult();
            }

            auto blob = cnnLayer->blobs["custom"];
            auto blob_precision = blob->getTensorDesc().getPrecision();

            if (blob_precision != InferenceEngine::Precision::FP32 && blob_precision != InferenceEngine::Precision::FP16) {
                quant->_dst_quant.SetScale(1.0f);
                return true;
            }

            if (blob_precision == InferenceEngine::Precision::FP16) {
                blob = make_fp32_blob(blob);
            }

            auto max_val = std::numeric_limits<float>::min();
            auto min_val = std::numeric_limits<float>::max();

            auto flt_buf = blob->buffer().as<float*>();
            auto size = blob->size();

            for (int i=0; i < size; i++) {
                auto val = flt_buf[i];
                if (val > max_val) max_val = val;
                if (val < min_val) min_val = val;
            }

            auto abs_val = std::max(std::abs(max_val), std::abs(min_val));
            auto scale_val = static_cast<float>(MAX_VAL_2B_FEAT) / abs_val;

            // TODO: Investigate what should be the scale in such cases (31910)
            if (std::isinf(scale_val)) {
                quant->_dst_quant.SetScale(quant->_src_quant.GetScale());
            } else {
                quant->_dst_quant.SetScale(scale_val);
            }

            return ScaleFactorUpdateResult();
        }

        if (!CNNNetHasPrevLayer(cnnLayer)) {
            quant->_dst_quant = quant->_src_quant;
            return ScaleFactorUpdateResult();
        }

        // by default layer is pass thru its scale factor
        auto inputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(CNNNetPrevLayer(cnnLayer));
        if (!inputQuant) {
            THROW_GNA_EXCEPTION << "layer: " << CNNNetPrevLayer(cnnLayer)->name << "not quantized";
        }

        quant->_src_quant = inputQuant->_dst_quant;

        if (layerInfo.isPower() && !layerInfo.isActivation()) {
            auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer*>(cnnLayer);
            if (!powerLayer) {
                THROW_GNA_EXCEPTION << "Invalid power layer pointer";
            }

            auto scale = std::abs(powerLayer->scale);
            if (!fp32eq(scale, 1.0f)) {
                quant->_weights_quant.SetScale(MAX_VAL_2B_FEAT / scale);
            }

            quant->_dst_quant.SetScale(quant->_src_quant.GetScale() * quant->_weights_quant.GetScale());
            return true;
        }

        if (layerInfo.isActivation()) {
            // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
            // set the initial value
            if (!quant->_dst_quant.IsScaleSet()) {
                auto scale = getActivationScale(cnnLayer, layerInfo);
                quant->_dst_quant.SetScale(scale);
            }
            return true;
        }
        quant->_dst_quant = inputQuant->_dst_quant;

        return true;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::EltwiseLayer*> {
 public:
    bool operator()(InferenceEngine::EltwiseLayer* eltwiseLayer, int weightsSize, ScaleFactorUpdateResult &result) {
        if ( !eltwiseLayer ) {
            THROW_GNA_EXCEPTION << "Incorrect Eltwise Layer pointer \n";
        }
        auto in0 = InferenceEngine::CNNNetPrevLayer(eltwiseLayer, 0);
        auto in1 = InferenceEngine::CNNNetPrevLayer(eltwiseLayer, 1);

        auto quantParams0 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0);
        auto quantParams1 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in1);

        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*eltwiseLayer);

        switch (eltwiseLayer->_operation) {
            case InferenceEngine::EltwiseLayer::Prod: {
                quantData->_weights_quant = quantParams1->_dst_quant;
                quantData->_dst_quant.SetScale(quantParams0->_dst_quant.GetScale() * quantParams1->_dst_quant.GetScale());
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
                    (LayerInfo(in0).isNonFunctional() && (LayerInfo(eltwiseFunctionalPrev).has32BOutput()))) {
                    std::swap(in0, in1);
                    std::swap(quantParams0, quantParams1);
                }

                // this path might result in significant data loss
                quantData->_bias_quant.SetScale(quantParams1->_dst_quant.GetScale() / quantParams0->_dst_quant.GetScale());
                quantData->_weights_quant.SetScale(quantParams1->_dst_quant.GetScale() / quantParams0->_dst_quant.GetScale());
                quantData->_dst_quant.SetScale(quantParams1->_dst_quant.GetScale());

                // eltwise will always work in int16
                auto maxValue = std::numeric_limits<int16_t>::max() - 1;
                if (quantData->_weights_quant.GetScale() > maxValue + 1) {
                    // rescaling it's activation input
                    // iterating thru previous layers of eltwise
                    for (uint8_t i = 0; i < 2; ++i) {
                        InferenceEngine::CNNLayerPtr in = InferenceEngine::CNNNetPrevLayer(eltwiseLayer, i);
                        // trick to get opposite index (for 0 -> 1 for 1 -> 0) by inversing i.
                        auto quantParams =
                                InferenceEngine::getInjectedData<QuantizedLayerParams>(InferenceEngine::CNNNetPrevLayer(eltwiseLayer, !i));

                        for (; InferenceEngine::CNNNetHasPrevLayer(in.get()); in = CNNNetPrevLayer(in)) {
                            auto info = LayerInfo(in);
                            // we skipping only split layers so far, also need to work on memory layers
                            // this case for input from port 0
                            if (info.isSplit() || info.isSlice()) {
                                continue;
                            } else if (info.has16BOutput() && info.isActivation()) {
                                auto newOutputScale = quantParams->_dst_quant.GetScale() / maxValue;
                                if (newOutputScale > static_cast<float>(std::numeric_limits<int16_t>::max()) / 2) {
                                    break;
                                }
                                auto quantDataForActivation = InferenceEngine::getInjectedData<QuantizedLayerParams>(*in);
                                gnawarn() << "[WARNING] saturated weights for " << eltwiseLayer->name
                                         << ". Layer new output scale: " << in->name << ", output_scale=" << newOutputScale
                                         << ", was " << quantDataForActivation->_dst_quant.GetScale() <<"\n" << std::flush;
                                quantDataForActivation->_dst_quant.SetScale(newOutputScale);
                                result = ScaleFactorUpdateResult(in.get());
                                return true;
                            } else if (info.has16BOutput()) {
                                break;
                            }

                            // if we are here it means that we are in the port 1
                            if (info.isFullyConnected() || info.isConvolution()) {
                                auto quantDataForInputLayer = InferenceEngine::getInjectedData<QuantizedLayerParams>(*in);
                                auto newOutputScale = quantParams->_dst_quant.GetScale() * maxValue;
                                auto newWeightScale = newOutputScale / quantDataForInputLayer->_src_quant.GetScale();
                                quantDataForInputLayer->_dst_quant.SetScale(newOutputScale);
                                quantDataForInputLayer->_weights_quant.SetScale(newWeightScale);
                                result = ScaleFactorUpdateResult(in.get());
                                return true;
                            }
                        }
                    }
                    // we unable to rescale the input - results might be bad
                    gnawarn() << "[INFO] weights saturated for " << eltwiseLayer->name << "\n";
                }
                break;
            }
            default : THROW_GNA_EXCEPTION << "Unsupported Eltwise layer for quantisation: " << eltwiseLayer->_operation;
        }
        return true;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::ConcatLayer*> {
 public:
    bool operator()(InferenceEngine::ConcatLayer* concatLayer, int weightsSize, ScaleFactorUpdateResult &result) {
        if ( !concatLayer ) {
            THROW_GNA_EXCEPTION << "Incorrect Concat Layer pointer \n";
        }

        if (concatLayer->insData.size() < 2) {
            THROW_GNA_EXCEPTION << "Concat layer has unsupported number of incoming layers.";
        }

        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*concatLayer);
        std::vector<InferenceEngine::CNNLayerPtr> inputLayers;
        for (auto input_idx = 0; input_idx != concatLayer->insData.size(); input_idx++) {
            auto prev_layer = InferenceEngine::CNNNetPrevLayer(concatLayer, input_idx);
            // FlattenConcat inserts reshape between concat and its inputs, which results in taking wrong layers as inputs for scale factor calulation
            if (prev_layer->type == "reshape" && prev_layer->insData.size() == 1 && prev_layer->outData.size() == 1) {
                prev_layer = InferenceEngine::CNNNetPrevLayer(prev_layer, 0);
            }
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

        // find the lowest scale factor (the biggest value) and use it as source to
        // requantize all other layers
        auto sourceLayerIt = inputLayers.begin();
        auto minScaleFactor = std::numeric_limits<float>::max();
        for (size_t i = 0; i < inputLayers.size(); ++i) {
            auto inputQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*inputLayers[i]);
            auto inputScale = inputQuantParams->_dst_quant.GetScale();
            if (inputQuantParams->_dst_quant.IsScaleSet() && !fp32eq(inputScale, 1.0f) && minScaleFactor > inputScale) {
                minScaleFactor = inputScale;
                sourceLayerIt = inputLayers.begin() + i;
            }
        }

        if (sourceLayerIt == inputLayers.end()) {
            sourceLayerIt = inputLayers.begin();
        }

        std::set<size_t> concatIdxToUpdate;
        auto sourceQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*sourceLayerIt);
        auto sourceScaleFactor = sourceQuantParams->_dst_quant.GetScale();

        for (size_t i = 0; i < inputLayers.size(); ++i) {
            auto quantParamsIn = InferenceEngine::getInjectedData<QuantizedLayerParams>(*inputLayers[i]);
            if (fp32eq(quantParamsIn->_dst_quant.GetScale(), sourceScaleFactor)) {
                continue;
            }

            concatIdxToUpdate.insert(i);
            quantParamsIn->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
        }

        auto updatedScaleFactor = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0)->_dst_quant.GetScale();
        auto equalScaleFactor = [updatedScaleFactor](InferenceEngine::CNNLayerPtr& inputLayer) {
            auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            return fp32eq(quantParams->_dst_quant.GetScale(), updatedScaleFactor);
        };

        auto layerIt = std::find_if_not(inputLayers.begin() + 1, inputLayers.end(), equalScaleFactor);
        if (layerIt != inputLayers.end()) {
            THROW_GNA_EXCEPTION << "layers entered into concat have different scale factors" << concatLayer->name;
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
                    if (!info.isActivation() && !info.isConst()) {
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
                THROW_GNA_EXCEPTION << "cannot requantize '" << layerIdToUpdate << "' ('" << inputLayers[layerIdToUpdate]->name
                    << "') input to concat: '" << concatLayer->name << "'";
            }
            auto quantDataForConCatInput = InferenceEngine::getInjectedData<QuantizedLayerParams>(*restartedLayer);

            auto restarLayerInfo = LayerInfo(restartedLayer);
            if (restarLayerInfo.isActivation()) {
                auto prevLayer = CNNNetPrevLayer(restartedLayer);
                while (prevLayer != nullptr &&
                    (LayerInfo(prevLayer).isNonFunctional() || LayerInfo(prevLayer).isPooling())) {
                    prevLayer = CNNNetPrevLayer(prevLayer);
                }

                if ((restarLayerInfo.isIdentity() || restarLayerInfo.isRelu()) &&
                    LayerInfo(prevLayer).isWeightable()) {
                    auto prevQuantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
                    auto weightsScaleFactor = prevQuantData->_weights_quant.GetScale();
                    auto weightsScaleFactors = std::vector<float>{};
                    if (LayerInfo(prevLayer).isConcatAlignFilter() || prevLayer->name.find("SyntheticScaleShift") != std::string::npos) {
                        weightsScaleFactors = generateScaleFactors(1.0f, DEFAULT_ACTIVATION_SCALE_FACTOR, DEFAULT_ACTIVATION_SCALE_FACTOR);
                    } else if (prevQuantData->_weights_quant.GetScale() > 1.0f) {
                        auto scaleFactors = generateScaleFactors(weightsScaleFactor, weightsScaleFactor / 2, weightsScaleFactor / 2);
                        weightsScaleFactors.insert(weightsScaleFactors.end(), weightsScaleFactor);
                        weightsScaleFactors.insert(weightsScaleFactors.end(), scaleFactors.begin(), scaleFactors.end());
                        weightsScaleFactors.insert(weightsScaleFactors.end(), weightsScaleFactor / 2);
                    }

                    if (!weightsScaleFactors.empty()) {
                        auto selWeightsScaleFactor = selectBestWeightsScaleFactors(prevQuantData->_src_quant.GetScale(),
                            sourceQuantParams->_dst_quant.GetScale(), weightsScaleFactors, {1.0f});

                        if (!fp32eq(weightsScaleFactor, selWeightsScaleFactor)) {
                            gnalog() << "[INFO] Changing weights scale factor for " << prevLayer ->name  << " from "
                                << prevQuantData->_weights_quant.GetScale() << " to " << selWeightsScaleFactor << std::endl;

                            prevQuantData->_weights_quant.SetScale(selWeightsScaleFactor);
                            prevQuantData->_dst_quant.SetScale(prevQuantData->_src_quant.GetScale() *
                                prevQuantData->_weights_quant.GetScale());
                            result = ScaleFactorUpdateResult(prevLayer.get());
                            return true;
                        }
                    }
                }

                // requantize activation by just changing it's output scale factor
                quantDataForConCatInput->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
            } else if (restarLayerInfo.isConst()) {
                gnalog() << "... warning const layer will be requantized\n";
                quantDataForConCatInput->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
            } else {
                THROW_GNA_EXCEPTION << "cannot requantize '" << layerIdToUpdate << "' ('" << inputLayers[layerIdToUpdate]->name
                    << "') input to concat: '" << concatLayer->name << "'";
            }
            result = ScaleFactorUpdateResult(restartedLayer.get());
        }

        return true;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::WeightableLayer*> {
 private:
    float const _scale_reduction_50 = 0.50;
    float const _scale_reduction_45 = 0.45;
    float const _scale_reduction_40 = 0.40;
    float const _scale_reduction_35 = 0.35;

    uint16_t const _scale_change_req_threshold = 30;
    uint16_t const _scale_change_threshold_100 = 100;
    uint16_t const _scale_change_threshold_150 = 150;
    uint16_t const _scale_change_threshold_200 = 200;

 public:
    bool operator()(InferenceEngine::WeightableLayer *wl, int weightsSize, ScaleFactorUpdateResult &result) {
        if ( !wl ) {
            THROW_GNA_EXCEPTION << "Incorrect Weightable Layer pointer  \n";
        } else if (!wl->_weights) {
            THROW_GNA_EXCEPTION << "Incorrect weight value for " << wl->name << ":" << wl->type << "\n";
        }

        auto prevLayer = CNNNetPrevLayer(wl);
        auto quantDataForInputLayer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(*InferenceEngine::CNNNetPrevLayer(wl).get());

        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*wl);
        quant->_src_quant = quantDataForInputLayer->_dst_quant;
        // TODO: pass 8 bits somehow
        if (quant->_weights_quant.GetScale() == 1.0f) {
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
            if (quant->_weights_quant.GetScale() == -1.0f) {
                quant->_weights_quant.SetScale(1.0f);
            }

            if (wl->_biases) {
                quant->_bias_quant.SetScale(ScaleFactorForQuantization(wl->_biases->buffer().as<float *>(),
                                                                      MAX_VAL_4B_BIAS,
                                                                      wl->_biases->size()));
                if (quant->_bias_quant.GetScale() != -1.0f) {
                    quant->_bias_quant.SetScale(
                        std::min(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale(), quant->_bias_quant.GetScale()));
                    quant->_weights_quant.SetScale(quant->_bias_quant.GetScale() / quant->_src_quant.GetScale());
                }
            }

            // TODO: findout why ???
            if (weightsSize == 1) {
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * MAX_OUT_MULTIPLIER);
            }

            double weights_reducer = 1.0;
            auto conv = dynamic_cast<InferenceEngine::ConvolutionLayer *>(wl);
            if (conv) {
                auto dims = conv->insData.front().lock()->getDims();

                weights_reducer = MAX_VAL_2B_FEAT * scaleRange * dims[1] / std::numeric_limits<int32_t>::max();
                weights_reducer = std::max(1.0, weights_reducer);
            }
            quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() / weights_reducer);
        }

        double tmp_dst_quant_scale = quant->_weights_quant.GetScale() * quant->_src_quant.GetScale();
        if (weightsSize == 1 &&
            static_cast<uint64_t>(tmp_dst_quant_scale * quant->_src_quant.GetScale()) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - 1) * _scale_change_req_threshold) {
            gnawarn() << "Output scale for " << wl->name
                << " too large and are being reduced. Else saturations likely will happen \n";
            // reduce weight scale according experimental heuristic
            if (quant->_dst_quant.GetScale() * quant->_src_quant.GetScale() /
                static_cast<float>(std::numeric_limits<int32_t>::max()) < _scale_change_threshold_100) {
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * _scale_reduction_50);
            } else if (quant->_dst_quant.GetScale() * quant->_src_quant.GetScale() /
                static_cast<float>(std::numeric_limits<int32_t>::max()) < _scale_change_threshold_150) {
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * _scale_reduction_45);
            } else if (quant->_dst_quant.GetScale() * quant->_src_quant.GetScale() /
                static_cast<float>(std::numeric_limits<int32_t>::max()) < _scale_change_threshold_200) {
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * _scale_reduction_40);
            } else {
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * _scale_reduction_35);
            }
        }

        quant->_dst_quant.SetScale(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale());
        return true;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::ScaleShiftLayer*> : public ScaleFactorPerLayer<InferenceEngine::WeightableLayer*> {
 public:
    bool operator()(InferenceEngine::WeightableLayer *wl, int weightsSize, ScaleFactorUpdateResult &result) {
        return ScaleFactorPerLayer<InferenceEngine::WeightableLayer*>::operator()(wl, 2, result);
    }
};

/**
 * GNA convolutions cannot be quantized in int8, remove when library starts support that
 */
template<>
class ScaleFactorPerLayer<InferenceEngine::ConvolutionLayer*> : public ScaleFactorPerLayer<InferenceEngine::ScaleShiftLayer*> {
};


}  // namespace frontend

/**
 * @brief scale factor calculator will calculate only output scale factors for the layer
 * if scale factor propagation not possible, it will fall indicate a restart condition
 */
class ScaleFactorCalculator {
    using Cnt = std::vector<InferenceEngine::CNNLayerPtr>;
    Cnt  net;
    mutable Cnt::const_iterator idx;
    mutable bool needRestart = false;
    int weightsBytesSize;

 public:
    ScaleFactorCalculator(Cnt &net, int weightsBytesSize)
            : net(net), weightsBytesSize(weightsBytesSize) {
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
    template<class T>
    bool operator()(T ptr) const {
        needRestart = false;
        frontend::ScaleFactorUpdateResult result;
        if (!frontend::ScaleFactorPerLayer<T>()(ptr, weightsBytesSize, result)) {
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
};

}  // namespace GNAPluginNS
