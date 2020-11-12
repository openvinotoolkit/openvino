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
#include <random>

#include <legacy/ie_layers.h>
#include "gna_upstream_iterator.hpp"
#include "layers/gna_layer_info.hpp"
#include "gna_plugin_log.hpp"
#include "gna_slope_scale.h"
#include "runtime/pwl.h"

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
    const float activation_scale_factor = 2048.f;
    const float identity_scale_factor = 2049.0f;
    const float k = 5;
    const float k_identity = 6;
    const double pow_domain = 16;
    const float MAX_VALUE = MAX_VAL_2B_FEAT;
    const float MEMORY_INIT_MAX_VALUE = 16.0f;

    float getWeightScaleRange(int32_t weightsSize) {
        auto scaleRange = 0.0f;
        switch (weightsSize) {
        case 1:
            return MAX_VAL_1B_WEIGHT;
            break;
        case 2:
            return MAX_VAL_2B_WEIGHT;
            break;
        default:
            THROW_GNA_EXCEPTION << "Unsupported weights size of: " << weightsSize;
            break;
        }
    };

 protected :
    static bool fp32eq(float p1, float p2) {
        return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
    }

    float getActivationScale(InferenceEngine::CNNLayer const* cnnLayer,
                             GNAPluginNS::LayerInfo const& layer) {
        auto quantizedParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);

        // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
        // set the initial value
        if (quantizedParams->_src_quant.GetMinValues().size() != 1 ||
            quantizedParams->_src_quant.GetMaxValues().size() != 1) {
            THROW_GNA_EXCEPTION << "Invalid number of min and max input values. min size: "
                << quantizedParams->_src_quant.GetMinValues().size()
                << ", max size: " << quantizedParams->_src_quant.GetMaxValues().size();
        }

        auto inputMinValue = quantizedParams->_src_quant.GetMinValues().front();
        auto inputMaxValue = quantizedParams->_src_quant.GetMaxValues().front();

        if (quantizedParams->_dst_quant.GetMaxValues().empty()) {
            auto outputMinValue = inputMinValue;
            auto outputMaxValue = inputMaxValue;

            if (layer.isPower()) {
                auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer const*>(cnnLayer);
                if (!powerLayer) {
                    THROW_IE_EXCEPTION << "Incorrect Power Layer pointer \n";
                }

                // Special case: when power exponent is not integral number, it means that
                // minimum number must be equal or greater than 0.
                if (!fp32eq(fmod(powerLayer->power, 1.0), 0)) {
                    quantizedParams->_src_quant.SetMinValues({ 0.0f });
                    inputMinValue = quantizedParams->_src_quant.GetMinValues().front();
                }

                outputMinValue = pow(inputMinValue * powerLayer->scale + powerLayer->offset, powerLayer->power);
                outputMaxValue = pow(inputMaxValue * powerLayer->scale + powerLayer->offset, powerLayer->power);
            } else if (layer.isSigmoid()) {
                outputMinValue = sigmoid(inputMinValue);
                outputMaxValue = sigmoid(inputMaxValue);
            } else if (layer.isTanh()) {
                outputMinValue = tanh(inputMinValue);
                outputMaxValue = tanh(inputMaxValue);
            } else if (layer.isExp()) {
                outputMinValue = exp(inputMinValue);
                outputMaxValue = exp(inputMaxValue);
            } else if (layer.isLog()) {
                outputMinValue = log(inputMinValue);
                outputMaxValue = log(inputMaxValue);
            } else if (layer.isNegLog()) {
                outputMinValue = neglog(inputMinValue);
                outputMaxValue = neglog(inputMaxValue);
            } else if (layer.isNegHalfLog()) {
                outputMinValue = neghalflog(inputMinValue);
                outputMaxValue = neghalflog(inputMaxValue);
            } else if (layer.isClamp()) {
                outputMinValue = std::max(static_cast<float>(KALDI_LSTM_CLIP_LOWER), inputMinValue);
                outputMaxValue = std::min(static_cast<float>(KALDI_LSTM_CLIP_UPPER), inputMaxValue);
            } else if (layer.isAbs()) {
                outputMinValue = std::abs(inputMinValue);
                outputMaxValue = std::abs(inputMaxValue);
            } else if (layer.isSoftSign()) {
                outputMinValue = softsign(inputMinValue);
                outputMaxValue = softsign(inputMaxValue);
            } else if (layer.isSign()) {
                outputMinValue = sign(inputMinValue);
                outputMaxValue = sign(inputMaxValue);
            } else if (layer.isIdentity()) {
                outputMinValue = inputMinValue;
                outputMaxValue = inputMaxValue;
            } else if (layer.isRelu()) {
                outputMinValue = relu(inputMinValue);
                outputMaxValue = relu(inputMaxValue);
            } else if (layer.isLeakyRelu()) {
                outputMinValue = leaky_relu(inputMinValue);
                outputMaxValue = leaky_relu(inputMaxValue);
            } else {
                THROW_IE_EXCEPTION << "Unknown activation function\n";
            }

            // Special case:
            // When power exponent is 0. In such case min and max are equal to 1.
            // If they are the same, the scale factor cannot be calculated correctly.
            outputMinValue = fp32eq(outputMinValue, outputMaxValue) ? 0.0f : outputMinValue;

            auto maxOutValue = std::max(std::abs(outputMinValue), std::abs(outputMaxValue));

            quantizedParams->_dst_quant.SetLevels(2 * MAX_VALUE);
            quantizedParams->_dst_quant.SetMinValues({ -maxOutValue });
            quantizedParams->_dst_quant.SetMaxValues({ maxOutValue });
        }

        float result = activation_scale_factor;
        if (!quantizedParams->_dst_quant.GetMaxValues().empty()) {
            auto minValue = quantizedParams->_dst_quant.GetMinValues().front();
            auto maxValue = quantizedParams->_dst_quant.GetMaxValues().front();
            result = (quantizedParams->_dst_quant.GetLevels() - 1) / (maxValue - minValue);
        }

        //result = result > activation_scale_factor ? activation_scale_factor : result;

        /*
        if (layer.isIdentity()) {
            // #define accurate_identity_scale_factor
#ifdef accurate_identity_scale_factor
            // searching more accurate scale factor for identity
            const double min_range = 1024.0;
            const double max_range = 2049.0;

            // gna encoded scale factor - the max this is the more accurate PWL approximation
            double optimalK = 0.0f;
            result = min_range;

            for (int slope_scale_index = 1; slope_scale_index != 5; slope_scale_index++) {
                auto slope_scale = static_cast<double>(static_cast<uint64_t>(1) << (8 * slope_scale_index));
                auto mink = min_range * slope_scale / quantizedParams->_src_quant.GetScale();
                auto maxk = max_range * slope_scale / quantizedParams->_src_quant.GetScale();

                if (mink < std::numeric_limits<int16_t>::max()) {
                    auto localMaxK = std::min(static_cast<double>(std::numeric_limits<int16_t>::max()), maxk);
                    if (localMaxK > optimalK) {
                        result = localMaxK / slope_scale * quantizedParams->_src_quant.GetScale();
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
            result = fabs(scale_extra) > fabs(scale_default) ? identity_scale_factor / 2 : identity_scale_factor;

#endif
        }
        else if (layer.isRelu() &&
            static_cast<uint64_t>(activation_scale_factor * quantizedParams->_src_quant.GetScale())
                                                            > std::numeric_limits<int32_t>::max() - 1) {
            // if activation is one from relu family, we need to apply heuristic to avoid activation output overflow
            result = (activation_scale_factor * 0.5);
        }*/

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

            if (!CNNNetHasPrevLayer(cnnLayer) && !quant->_dst_quant.IsScaleSet()) {
                quant->_dst_quant.SetMinValues({ -MEMORY_INIT_MAX_VALUE });
                quant->_dst_quant.SetMaxValues({ MEMORY_INIT_MAX_VALUE });
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
                        quant->_src_quant = inputQuant->_dst_quant;
                        quant->_dst_quant = inputQuant->_dst_quant;
                        return true;
                    }

                    if (quantSibling->_src_quant.IsScaleSet() &&
                        !fp32eq(quantSibling->_src_quant.GetScale(), quantSibling->_dst_quant.GetScale())) {
                        THROW_GNA_EXCEPTION << "Inconsistent scale factor in " << cnnLayer->name << " layer. "
                            << "Input: " << quantSibling->_src_quant.GetScale()
                            << ", output: " << quantSibling->_dst_quant.GetScale();
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
                            if (quantDataForMemoryOutput->_dst_quant.IsScaleSet()
                                && quantDataForMemoryOutput->_dst_quant.GetScale() < quantSibling->_dst_quant.GetScale()) {
                                gnalog() << "[WARNING] Possible output saturation. "
                                    << restartedLayer->name << " layer scale factor change from "
                                    << quantDataForMemoryOutput->_dst_quant.GetScale() << " to: "
                                    << quantSibling->_dst_quant.GetScale() << "\n";
                            }
                            gnalog() << "[WARNING] " << restartedLayer->name << " layer will be requantized. Output scale factor change from "
                                << quantDataForMemoryOutput->_dst_quant.GetScale() << " to "
                                << quantSibling->_dst_quant.GetScale() << "\n";

                            quantDataForMemoryOutput->_dst_quant = quantSibling->_dst_quant;
                        } else {
                            THROW_GNA_EXCEPTION << "quantization error : input scale factor ( " << inputQuant->_dst_quant.GetScale() <<") "
                                  << " for " << cnnLayer->name << ", that is child of " << prevLayer->name <<" doesnt match : "
                                  << activation_scale_factor;
                        }

                        result = ScaleFactorUpdateResult(restartedLayer.get());
                        return true;
                    }

                    gnawarn() << "[INFO] quantization : input scale factor (" << inputQuant->_dst_quant.GetScale() << ")"
                        << " for " << cnnLayer->name << ", that is child of " << prevLayer->name << " doesnt match : "
                        << quantSibling->_dst_quant.GetScale() << ", restarting from corresponding memory: " << input->name << std::endl;

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

            if (blob_precision == InferenceEngine::Precision::I16) {
                auto max_val = std::numeric_limits<int16_t>::min();
                auto min_val = std::numeric_limits<int16_t>::max();

                auto buf = blob->buffer().as<int16_t*>();
                auto size = blob->size();
                for (size_t i = 0; i < size; i++) {
                    auto val = buf[i];
                    if (val > max_val) max_val = val;
                    if (val < min_val) min_val = val;
                }

                quant->_dst_quant.SetMinValues({ static_cast<float>(min_val) });
                quant->_dst_quant.SetMaxValues({ static_cast<float>(max_val) });
            }

            if (blob_precision == InferenceEngine::Precision::U8) {
                auto max_val = std::numeric_limits<int8_t>::min();
                auto min_val = std::numeric_limits<int8_t>::max();

                auto buf = blob->buffer().as<int8_t*>();
                auto size = blob->size();
                for (size_t i = 0; i < size; i++) {
                    auto val = buf[i];
                    if (val > max_val) max_val = val;
                    if (val < min_val) min_val = val;
                }

                quant->_dst_quant.SetMinValues({ static_cast<float>(min_val) });
                quant->_dst_quant.SetMaxValues({ static_cast<float>(max_val) });
            }

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

            for (size_t i = 0; i < size; i++) {
                auto val = flt_buf[i];
                if (val > max_val) max_val = val;
                if (val < min_val) min_val = val;
            }

            auto abs_val = std::max(std::abs(max_val), std::abs(min_val));
            //auto scale_val = static_cast<float>(MAX_VALUE) / abs_val;
            if (quant->_dst_quant.GetMinValues().empty()) {
                quant->_dst_quant.SetMinValues({ -abs_val });
                quant->_dst_quant.SetMaxValues({ abs_val });
            }

            auto scale_val = static_cast<float>(std::numeric_limits<int16_t>::max()) / abs_val;

            // TODO: Investigate what should be the scale in such cases (31910)
            if (std::isinf(scale_val)) {
                quant->_dst_quant.SetScale(quant->_src_quant.GetScale());
            } else {
                quant->_dst_quant.SetScale(scale_val);
            }

            return ScaleFactorUpdateResult();
        }

        if (!CNNNetHasPrevLayer(cnnLayer)) {
            if (quant->_src_quant.GetMinValues().empty()) {
                auto maxValue = MAX_VALUE / quant->_src_quant.GetScale();
                quant->_src_quant.SetMinValues({ -maxValue });
                quant->_src_quant.SetMaxValues({ maxValue });
                quant->_dst_quant = quant->_src_quant;
            }
            return ScaleFactorUpdateResult();
        }

        // by default layer is pass thru its scale factor
        auto inputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(CNNNetPrevLayer(cnnLayer));
        if (!inputQuant) {
            THROW_GNA_EXCEPTION << "layer: " << CNNNetPrevLayer(cnnLayer)->name << "not quantized";
        }

        quant->_src_quant = inputQuant->_dst_quant;
        // power layer creates weights dynamically and it needs to be taken into account
        // during scale factor calculation
        if (!layerInfo.isActivation() && layerInfo.isPower()) {
            auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer const*>(cnnLayer);
            if (!powerLayer) {
                THROW_IE_EXCEPTION << "Incorrect Power Layer pointer \n";
            }

            auto powerScale = std::abs(powerLayer->scale);
            auto powerOffset = std::abs(powerLayer->offset);

            auto weightScaleRange = getWeightScaleRange(weightsSize);
            auto weightScaleFactor = weightScaleRange / powerScale;
            quant->_weights_quant.SetScale(weightScaleFactor);
            quant->_dst_quant.SetScale(quant->_src_quant.GetScale() * weightScaleFactor);

            auto maxOutputValue = quant->_src_quant.GetMaxValues().front() * powerScale + powerOffset;
            quant->_dst_quant.SetMinValues({ -maxOutputValue });
            quant->_dst_quant.SetMaxValues({ maxOutputValue });
            return true;
        }

        if (layerInfo.isActivation()) {
            // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
            // set the initial value
            auto scale = getActivationScale(cnnLayer, layerInfo);
            quant->_dst_quant.SetScale(scale);
            return true;
        }
        quant->_dst_quant = inputQuant->_dst_quant;

        return true;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::EltwiseLayer*> {
private:
    static bool fp32eq(float p1, float p2) {
        return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
    }

public:
    bool operator()(InferenceEngine::EltwiseLayer* eltwiseLayer, int weightsSize, ScaleFactorUpdateResult &result) {
        if ( !eltwiseLayer ) {
            THROW_GNA_EXCEPTION << "Incorrect Eltwise Layer pointer \n";
        }
        auto in0 = InferenceEngine::CNNNetPrevLayer(eltwiseLayer, 0);
        auto in1 = InferenceEngine::CNNNetPrevLayer(eltwiseLayer, 1);

        auto quantParams0 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0);
        auto quantParams1 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in1);

        if (quantParams0->_dst_quant.GetMinValues().size() != 1 ||
            quantParams0->_dst_quant.GetMaxValues().size() != 1) {
            THROW_GNA_EXCEPTION << "Invalid number of min and max input values for " << in0->name << " layer."
                << " Min: " << quantParams0->_dst_quant.GetMinValues().size()
                << ", max: " << quantParams0->_dst_quant.GetMaxValues().size();
        }

        if (quantParams1->_dst_quant.GetMinValues().size() != 1 ||
            quantParams1->_dst_quant.GetMaxValues().size() != 1) {
            THROW_GNA_EXCEPTION << "Invalid number of min and max input values for " << in1->name << " layer."
                << " Min: " << quantParams1->_dst_quant.GetMinValues().size()
                << ", max: " << quantParams1->_dst_quant.GetMaxValues().size();
        }

        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*eltwiseLayer);

        switch (eltwiseLayer->_operation) {
            case InferenceEngine::EltwiseLayer::Prod: {
                quantData->_weights_quant = quantParams1->_dst_quant;
                quantData->_dst_quant.SetScale(quantParams0->_dst_quant.GetScale() * quantParams1->_dst_quant.GetScale());
                auto maxInputValue = quantParams0->_dst_quant.GetMaxValues().front() * quantParams1->_dst_quant.GetMaxValues().front();
                quantData->_dst_quant.SetMinValues({ -maxInputValue });
                quantData->_dst_quant.SetMaxValues({ maxInputValue });
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

                auto maxInputValue = quantParams0->_dst_quant.GetMaxValues().front() + quantParams1->_dst_quant.GetMaxValues().front();
                quantData->_dst_quant.SetMinValues({ -maxInputValue });
                quantData->_dst_quant.SetMaxValues({ maxInputValue });

                // eltwise will always work in int16
                double reducer = quantData->_weights_quant.GetScale() / std::numeric_limits<int16_t>::max();
                reducer = std::max(1.0, reducer);

                if (!fp32eq(reducer, 1.0f) && reducer > 1.0) {
                    gnawarn() << "[WARNING] Possible saturation of output results for " << eltwiseLayer->name << " layer."
                        << " Attempting to reduce scale factors for input layers.\n";

                    auto compare = [](InferenceEngine::CNNLayer* first, InferenceEngine::CNNLayer* second) -> bool {
                        auto quantDataFirst = InferenceEngine::getInjectedData<QuantizedLayerParams>(*first);
                        auto quantDataSecond = InferenceEngine::getInjectedData<QuantizedLayerParams>(*second);

                        auto scaleFactorFirst = quantDataFirst->_dst_quant.GetScale();
                        auto infoFirst = LayerInfo(first);
                        if (infoFirst.isWeightable() ||
                            (infoFirst && !infoFirst.isActivation())) {
                            auto weightsScale = quantDataFirst->_weights_quant.GetScale();
                            auto srcScale = quantDataFirst->_src_quant.GetScale();
                            scaleFactorFirst = weightsScale > srcScale ? weightsScale : srcScale;
                        }

                        auto scaleFactorSecond = quantDataSecond->_dst_quant.GetScale();
                        auto infoSecond = LayerInfo(second);
                        if (infoSecond.isWeightable() |
                            (infoSecond.isPower() && !infoSecond.isActivation())) {
                            auto weightsScale = quantDataSecond->_weights_quant.GetScale();
                            auto srcScale = quantDataSecond->_src_quant.GetScale();
                            scaleFactorSecond = weightsScale > srcScale ? weightsScale : srcScale;
                        }

                        return scaleFactorFirst > scaleFactorSecond;
                    };

                    std::deque<InferenceEngine::CNNLayer*> layersToCheck;
                    layersToCheck.push_back(in1.get());

                    auto reductionFactor = reducer; //(reducer > 2.0) ? (reducer / 2.0) : reducer;
                    while (!layersToCheck.empty()) {
                        auto layer = *layersToCheck.begin();
                        auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*layer);
                        auto info = LayerInfo(*layer);

                        if (info.isWeightable() ||
                            (info.isPower() && !info.isActivation())) {
                            if (quantParams->_src_quant.GetScale() < quantParams->_weights_quant.GetScale()) {
                                auto newWeightScale = quantParams->_weights_quant.GetScale() / reductionFactor;
                                gnalog() << "Output scale for " << layer->name << " is reduced. "
                                    << "Layer new weight scale: " << newWeightScale
                                    << ", old weight scale: " << quantParams->_weights_quant.GetScale()
                                    << "\n";
                                quantParams->_weights_quant.SetScale(newWeightScale);
                                quantParams->_dst_quant.SetScale(quantParams->_weights_quant.GetScale() * quantParams->_src_quant.GetScale());
                                result = ScaleFactorUpdateResult(layer);
                            }
                            return true;
                        } else if (info.isActivation()) {
                            auto newOutputScale = quantParams->_dst_quant.GetScale() / reductionFactor;
                            gnalog() << "Output scale for " << layer->name << " is reduced. "
                                << "Layer new output scale: " << newOutputScale
                                << ", old output scale: " << quantParams->_dst_quant.GetScale()
                                << "\n";
                            quantParams->_dst_quant.SetScale(newOutputScale);
                            result = ScaleFactorUpdateResult(layer);
                            return true;
                        } else if (info.isConst()) {
                            auto newOutputScale = quantParams->_dst_quant.GetScale() / reductionFactor;
                            gnalog() << "Output scale for " << layer->name << " is reduced. "
                                << "Layer new input and output scale: " << newOutputScale
                                << ", old new input and output scale: " << quantParams->_dst_quant.GetScale()
                                << "\n";
                            quantParams->_src_quant.SetScale(newOutputScale);
                            quantParams->_dst_quant.SetScale(newOutputScale);
                            result = ScaleFactorUpdateResult(layer);
                            return true;
                        }
                        else if (info.isMemory() || info.isInput()) {
                            layersToCheck.erase(layersToCheck.begin());
                            continue;
                        }

                        for (auto input_idx = 0; input_idx != layer->insData.size(); input_idx++) {
                            auto prevLayer = InferenceEngine::CNNNetPrevLayer(layer, input_idx);
                            layersToCheck.push_back(prevLayer.get());
                        }

                        layersToCheck.erase(layersToCheck.begin());
                        std::sort(layersToCheck.begin(), layersToCheck.end(), compare);
                    }

                    THROW_GNA_EXCEPTION << "Scale factor for " << eltwiseLayer->name << " layer cannot be reduced\n";
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
    bool operator()(InferenceEngine::ConcatLayer* concatLayer, int weightsSize, ScaleFactorUpdateResult& result) {
        if (!concatLayer) {
            THROW_GNA_EXCEPTION << "Incorrect Concat Layer pointer \n";
        }

        if (concatLayer->insData.size() < 2) {
            THROW_GNA_EXCEPTION << "Concat layer has unsupported number of incoming layers.";
        }

        auto fp32eq = [](float p1, float p2) -> bool {
            return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
        };

        auto concatQuantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*concatLayer);
        std::vector<InferenceEngine::CNNLayerPtr> inputLayers;
        for (auto input_idx = 0; input_idx != concatLayer->insData.size(); input_idx++) {
            inputLayers.push_back(InferenceEngine::CNNNetPrevLayer(concatLayer, input_idx));
        }

        // if all inputs have same quant value - trivial propagation
        auto in0 = inputLayers.front();
        auto quantParams0 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0);
        auto scaleFactor = quantParams0->_dst_quant.GetScale();
        auto scaleFactorCheck = [scaleFactor, &fp32eq](InferenceEngine::CNNLayerPtr& inputLayer) {
            auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayer);
            return fp32eq(quantParams->_dst_quant.GetScale(), scaleFactor);
        };

        if (std::find_if_not(inputLayers.begin() + 1, inputLayers.end(), scaleFactorCheck) == inputLayers.end()) {
            concatQuantData->_dst_quant = quantParams0->_dst_quant;
            concatQuantData->_src_quant = quantParams0->_dst_quant;
            return true;
        }

        // Scale factors for input layers to concat differ.
        // Continue only if all inputs have the same scale factor values.
        auto inputLayerCheck = [](InferenceEngine::CNNLayerPtr& inputLayer) -> bool {
            auto prevLayer = inputLayer;
            while (prevLayer) {
                auto info = LayerInfo(prevLayer);
                if (info.isInput()) {
                    return true;
                }

                if (!info.isNonFunctional()) {
                    return false;
                }

                prevLayer = CNNNetPrevLayer(prevLayer);
            }

            
            return false;
        };

        GNAPluginNS::QuantizedLayerParams* sourceQuantParams = nullptr;
        auto firstInputIt = std::find_if(inputLayers.begin(), inputLayers.end(), inputLayerCheck);
        if (firstInputIt != inputLayers.end()) {
            auto quantParamsFirst = InferenceEngine::getInjectedData<QuantizedLayerParams>(*firstInputIt);
            auto nextInputIt = firstInputIt + 1;
            while ((nextInputIt = std::find_if(nextInputIt, inputLayers.end(), inputLayerCheck)) != inputLayers.end()) {
                auto quantParamsSecond = InferenceEngine::getInjectedData<QuantizedLayerParams>(*nextInputIt);
                if (!fp32eq(quantParamsSecond->_dst_quant.GetScale(), quantParamsFirst->_dst_quant.GetScale())) {
                    THROW_GNA_EXCEPTION << "Two Input layers " << (*firstInputIt)->name
                        << " and " << (*nextInputIt)->name << " have different scales in concat!!! \n";
                }
            }
        }


        // find a source quant value
        // - 1st candidate - input layer
        // - 2nd candidate - non-input layer with minimum scale factor
        if (firstInputIt != inputLayers.end()) {
            sourceQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*firstInputIt);
        } else {
            sourceQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayers.front());
            for (auto& layer : inputLayers) {
                auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(layer);
                if (sourceQuantParams->_dst_quant.GetScale() > quantParams->_dst_quant.GetScale()) sourceQuantParams = quantParams;
            }
        }

        std::set<size_t> concatIdxToUpdate;
        auto maxInputValue = 0.0f;
        for (int i = 0; i < inputLayers.size(); i++) {
            auto quantParamsIn = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayers[i]);

            // Find maximum input value to concat layer. It does not have to be the lowest scale factor.
            if (quantParamsIn->_dst_quant.GetMinValues().size() != 1 ||
                quantParamsIn->_dst_quant.GetMaxValues().size() != 1) {
                THROW_GNA_EXCEPTION << "Incorrect number of maximum values for " << inputLayers[i]->name << " layer. "
                    << "Min: " << quantParamsIn->_src_quant.GetMinValues().size()
                    << ", max: " << quantParamsIn->_src_quant.GetMaxValues().size();
            }

            maxInputValue = std::max(maxInputValue, std::abs(quantParamsIn->_dst_quant.GetMinValues().front()));
            maxInputValue = std::max(maxInputValue, std::abs(quantParamsIn->_dst_quant.GetMaxValues().front()));

            // Skip updating layer if the new scale factor and old one are the equal
            if (fp32eq(quantParamsIn->_dst_quant.GetScale(), sourceQuantParams->_dst_quant.GetScale())) continue;

            // Scale factor for an input layar has to be updated
            concatIdxToUpdate.insert(i);
        }

        concatQuantData->_dst_quant = sourceQuantParams->_dst_quant;
        concatQuantData->_dst_quant.SetMinValues({ -maxInputValue });
        concatQuantData->_dst_quant.SetMaxValues({ maxInputValue });

        concatQuantData->_src_quant = concatQuantData->_dst_quant;

        if (concatIdxToUpdate.empty()) {
            return true;
        }

        for (auto& layerIdToUpdate : concatIdxToUpdate) {
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
                THROW_GNA_EXCEPTION << "cannot requantize " << inputLayers[layerIdToUpdate]->name << " input to concat: " << concatLayer->name;
            }
            auto quantDataForConCatInput = InferenceEngine::getInjectedData<QuantizedLayerParams>(*restartedLayer);

            auto checkIfScaleFactorIsSet = [](InferenceEngine::CNNLayerPtr& layer) -> bool {
                auto prevLayer = layer;
                while (prevLayer) {
                    auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
                    if (!quantParams->_src_quant.IsScaleSet()) {
                        return false;
                    }
                    GNAPluginNS::LayerInfo info(prevLayer);
                    if (info.isInput() || info.isConst() || info.isMemory()) {
                        break;
                    }

                    prevLayer = CNNNetPrevLayer(prevLayer);
                }

                return true;
            };

            auto restarLayerInfo = LayerInfo(restartedLayer);
            if (restarLayerInfo.isActivation()) {
                // requantize activation by just changing it's output scale factor
                if (checkIfScaleFactorIsSet(restartedLayer) &&
                    quantDataForConCatInput->_dst_quant.GetScale() < sourceQuantParams->_dst_quant.GetScale()) {
                    gnalog() << "[WARNING] Possible output saturation. "
                        << restartedLayer->name << " layer scale factor change from "
                        << quantDataForConCatInput->_dst_quant.GetScale() << " to: "
                        << sourceQuantParams->_dst_quant.GetScale() << "\n";
                }

                gnalog() << "[WARNING] " << restartedLayer->name << " layer will be requantized. Output scale factor change from "
                    << quantDataForConCatInput->_dst_quant.GetScale() << " to "
                    << sourceQuantParams->_dst_quant.GetScale() << "\n";
                quantDataForConCatInput->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
            } else if (restarLayerInfo.isMemory()) {
                quantDataForConCatInput->_src_quant = sourceQuantParams->_dst_quant;
                quantDataForConCatInput->_dst_quant = sourceQuantParams->_dst_quant;

                // search for output layer that corresponds to input memory layer to requanitize it
                InferenceEngine::CNNLayerSet outputLayers;
                std::unordered_set<InferenceEngine::CNNLayer*> allLayers;

                InferenceEngine::CNNLayerPtr layerPtr(restartedLayer.get(), [](InferenceEngine::CNNLayer*) {});

                InferenceEngine::details::UnorderedDFS(
                    allLayers, layerPtr,
                    [&](InferenceEngine::CNNLayerPtr layer) {
                        if (layer->outData.empty()) {
                            outputLayers.insert(layer);
                        }
                    },
                    false);

                for (auto&& output : outputLayers) {
                    LayerInfo ll(output);
                    if (!ll.isMemory() ||
                        !InferenceEngine::details::CaselessEq<std::string>()(output->params["id"], restartedLayer->params["id"])) {
                        continue;
                    }

                    auto quantSibling = InferenceEngine::getInjectedData<QuantizedLayerParams>(output);

                    gnalog() << "Memory layer :" << output->name << " scale factor: " << quantSibling->_dst_quant.GetScale()
                        << " doesn't match its input counterpart: " << restartedLayer->name << " scale factor: "
                        << quantDataForConCatInput->_src_quant.GetScale() << "\n";
                    gnalog() << "[UFS] searching for quantizeable input layer for: " << output->name << "\n";

                    InferenceEngine::CNNLayerPtr memoryLayerInput;

                    CNNNetDFS(InferenceEngine::CNNLayerPtr(output.get(), [](InferenceEngine::CNNLayer*) {}),
                        [&memoryLayerInput, output, concatLayer](InferenceEngine::CNNLayerPtr layer) {
                            gnalog() << "[UFS] from : " << output->name << " reached: " << layer->name;
                            // found that direct input to concat is a indirect parent of align filter - so no link required
                            auto info = LayerInfo(layer);
                            if (!info.isActivation() && (layer.get() != concatLayer)) {
                                gnalog() << "... skipped\n";
                                return;
                            }

                            memoryLayerInput = layer;
                            gnalog() << "... OK,  need requantize\n";
                        }, true, [&memoryLayerInput, &output](InferenceEngine::CNNLayer* from) {
                            // aborting UFS once found suitable layer
                            return make_upstream_order(memoryLayerInput == nullptr ? from : nullptr);
                        });

                    if (memoryLayerInput == nullptr) {
                        THROW_GNA_EXCEPTION << "cannot requantize input to " << output->name;
                    }

                    auto memoryLayerInputInfo = LayerInfo(memoryLayerInput);
                    if (memoryLayerInput.get() == concatLayer) {
                        // found a concat layer that triggered requantization
                    } else if (memoryLayerInputInfo.isActivation()) {
                        auto memoryLayerInputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*memoryLayerInput);

                        // requantize activation by just changing it's output scale factor
                        if (memoryLayerInputQuant->_dst_quant.IsScaleSet()
                            && memoryLayerInputQuant->_dst_quant.GetScale() < sourceQuantParams->_dst_quant.GetScale()) {
                            gnalog() << "[WARNING] Possible output saturation. "
                                << memoryLayerInput->name << " layer scale factor change from "
                                << memoryLayerInputQuant->_dst_quant.GetScale() << " to: "
                                << sourceQuantParams->_dst_quant.GetScale() << "\n";
                        }
                        gnalog() << "[WARNING] " << memoryLayerInput->name << " layer will be requantized. Output scale factor change from "
                            << memoryLayerInputQuant->_dst_quant.GetScale() << " to "
                            << sourceQuantParams->_dst_quant.GetScale() << "\n";

                        memoryLayerInputQuant->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
                    } else {
                        THROW_GNA_EXCEPTION << "quantization error: cannot requantize input to " << output->name << " layer";
                    }
                }
            }
            if (restarLayerInfo.isConst()) {
                gnalog() << "[WARNING] Const layer will be requantized. Scale factor change from "
                    << quantDataForConCatInput->_src_quant.GetScale() << " to "
                    << sourceQuantParams->_dst_quant.GetScale() << "\n";
                quantDataForConCatInput->_src_quant = sourceQuantParams->_dst_quant;
                quantDataForConCatInput->_dst_quant = sourceQuantParams->_dst_quant;
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

    static bool fp32eq(float p1, float p2) {
        return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
    }

    std::pair<float, float> findMinMaxValues(const std::vector<float>& A) {
        auto minValue = std::numeric_limits<float>::max();
        auto maxValue = std::numeric_limits<float>::min();
        for (auto val : A) {
            minValue = std::min(minValue, val);
            maxValue = std::max(maxValue, val);
        }

        return { minValue, maxValue };
    }

    std::vector<std::size_t> getFromIRDimsOrderNCHW(InferenceEngine::Layout layout) {
        std::vector<std::size_t> order;
        switch (layout) {
        case InferenceEngine::Layout::NHWC:
            order = { 4, 1, 3, 2 };
            break;
        case InferenceEngine::Layout::NCHW:
        default:
            order = { 4, 3, 2, 1 };
            break;
        }
        return order;
    }

    auto transposeMatrix(float* ptr_matrix, size_t num_rows, size_t num_cols) {
        size_t element_size = sizeof(float);
        std::vector<float> temp_buffer(num_rows * num_cols);
        for (size_t i = 0; i < num_rows; i++) {
            for (size_t j = 0; j < num_cols; j++) {
                ie_memcpy(&temp_buffer.front() + (j * num_rows + i),
                    temp_buffer.size() - (i * num_cols + j),
                    ptr_matrix + (i * num_cols + j),
                    element_size);
            }
        }
        return temp_buffer;
    }

    auto simulateAffine(const std::vector<float>& weights, const std::vector<float>& input, const std::vector<float>& biases,
        size_t numRowsA, size_t numColA, float scaleFactorA,
        size_t numRowsB, size_t numColB, float scaleFactorB) {
        if (numColA != numRowsB) {
            THROW_GNA_EXCEPTION << "Invalid number of rows in weights";
        }

        const size_t numRowsC = numRowsA;
        const size_t numColC = numColB;
        std::vector<float> result(numRowsC * numColC, 0.0f);
        for (size_t rowC = 0; rowC < numRowsC; ++rowC) {
            for (size_t colC = 0; colC < numColC; ++colC) {
                auto& sum = result[rowC * numColC + colC];
                sum = biases.empty() ? 0.0f : biases[rowC * numColC + colC];
                for (size_t colA = 0; colA < numColA; ++colA) {
                    int32_t valA = weights[rowC * numColA + colA] * scaleFactorA;
                    int32_t valB = input[colA * numColB + colC] * scaleFactorB;
                    sum += valA * valB / (scaleFactorA * scaleFactorB);
                }
            }
        }

        return result;
    }

    auto convertNHWCtoNCHW(std::vector<float>& input,
        size_t width, size_t height, size_t channelNum, size_t batchSize) {
        std::vector<float> tmp(input.size());
        for (int n = 0; n < batchSize; n++) {
            for (int c = 0; c < channelNum; c++) {
                for (int h = 0; h < height; h++) {
                    for (int w = 0; w < width; w++) {
                        size_t dstIdx = n * channelNum * width * height +
                            c * width * height +
                            h * width +
                            w;
                        size_t srcIdx = n * channelNum * width * height +
                            h * width * channelNum +
                            w * channelNum +
                            c;
                        tmp[dstIdx] = input[srcIdx];
                    }
                }
            }
        }

        return tmp;
    }

    auto simulateConvolution(const std::vector<float>& src_buffer,
        const std::vector<float>& kernels_data,
        const std::vector<float>& bias_data,
        size_t strideX, size_t strideY, size_t strideZ,
        size_t paddingX, size_t paddingY, size_t paddingZ,
        size_t inputWidth, size_t inputHeight, size_t inputChannels, float inputScaleFactor,
        size_t outputWidth, size_t outputHeight, size_t outputChannels,
        size_t kernelWidth, size_t kernelHeight, float kernelScaleFactor) {
        std::vector<float> dst_data(outputWidth * outputHeight * outputChannels, 0.0f);

        if (inputHeight != 1 || outputHeight != 1) {
            THROW_GNA_EXCEPTION << "2D convolutions are not supported yet";
        }

        for (uint32_t oc = 0; oc < outputChannels; ++oc) {
            for (uint32_t oh = 0; oh < outputHeight; ++oh) {
                for (uint32_t ow = 0; ow < outputWidth; ++ow) {
                    size_t oidx = oc * outputHeight * outputWidth + oh * outputWidth + ow;
                    if (!bias_data.empty()) {
                        dst_data[oidx] = static_cast<int32_t>(bias_data[oc] * inputScaleFactor * kernelScaleFactor) /
                            (inputScaleFactor * kernelScaleFactor);
                    }

                    for (size_t ic = 0; ic < inputChannels; ++ic) {
                        for (size_t kh = 0; kh < kernelHeight; ++kh) {
                            for (size_t kw = 0; kw < kernelWidth; ++kw) {
                                int32_t iw = ow * strideX - paddingX + kw;
                                int32_t ih = oh * strideY - paddingY + kh;
                                if (iw < 0 || iw >= inputWidth ||
                                    ih < 0 || ih >= inputHeight)
                                    continue;

                                size_t inputIdx = ic * inputWidth * inputHeight +
                                    ih * inputWidth +
                                    iw;
                                size_t kernelIdx = oc * kernelWidth * kernelHeight * inputChannels +
                                    ic * kernelWidth * kernelHeight +
                                    kh * kernelWidth +
                                    kw;

                                int32_t inputVal = src_buffer[inputIdx] * inputScaleFactor;
                                int32_t kernelVal = kernels_data[kernelIdx] * kernelScaleFactor;
                                dst_data[oidx] += static_cast<double>(inputVal * kernelVal) /
                                    (inputScaleFactor * kernelScaleFactor);

                            }
                        }
                    }
                }
            }
        }

        return dst_data;
    }

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
        if (fp32eq(quant->_weights_quant.GetScale(), 1.0f)) {
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
            quant->_weights_quant.SetLevels(scaleRange);
            quant->_weights_quant.SetMinValues({ -scaleRange / quant->_weights_quant.GetScale() });
            quant->_weights_quant.SetMaxValues({ scaleRange / quant->_weights_quant.GetScale() });
            if (quant->_weights_quant.GetScale() == -1.0f) {
                quant->_weights_quant.SetScale(1.0f);
                quant->_weights_quant.SetMinValues({ 0.0f });
                quant->_weights_quant.SetMaxValues({ 1.0f } );
            }

            if (wl->_biases) {
                quant->_bias_quant.SetScale(ScaleFactorForQuantization(wl->_biases->buffer().as<float *>(),
                                                                      MAX_VAL_4B_BIAS,
                                                                      wl->_biases->size()));
                if (quant->_bias_quant.GetScale() != -1.0f) {
                    quant->_weights_quant.SetScale(MAX_VAL_4B_BIAS);
                    quant->_weights_quant.SetMinValues({ -MAX_VAL_4B_BIAS / quant->_bias_quant.GetScale() });
                    quant->_weights_quant.SetMaxValues({ MAX_VAL_4B_BIAS / quant->_bias_quant.GetScale() });
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

        if (quant->_dst_quant.GetMaxValues().empty()) {
            if (quant->_src_quant.GetMinValues().size() != 1 ||
                quant->_src_quant.GetMaxValues().size() != 1) {
                THROW_GNA_EXCEPTION << "Incorrect number of maximum values for " << wl->name << " layer. "
                    << "Min: " << quant->_src_quant.GetMinValues().size()
                    << ", max: " << quant->_src_quant.GetMaxValues().size();
            }

            if (wl->_weights->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
                THROW_GNA_EXCEPTION << "Invalid weights precision " << wl->name << " layer. Expected: FP32, got: "
                    << wl->_weights->getTensorDesc().getPrecision();
            }

            const int seed = 0;
            std::mt19937 gen(static_cast<float>(seed));
            auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
                std::vector<float> res;

                std::normal_distribution<float> dist(min, max);
                for (int i = 0; i < vec_len; i++)
                    res.emplace_back(static_cast<float>(dist(gen)));

                return res;
            };

            auto inputMinValue = quantDataForInputLayer->_dst_quant.GetMinValues().front();
            auto inputMaxValue = quantDataForInputLayer->_dst_quant.GetMaxValues().front();
            auto minOutputValue = inputMinValue * quant->_weights_quant.GetMaxValues().front();
            auto maxOutputValue = std::abs(inputMaxValue) * std::abs(quant->_weights_quant.GetMaxValues().front());
            if (!quant->_bias_quant.GetMaxValues().empty()) {
                maxOutputValue += std::abs(quant->_bias_quant.GetMaxValues().front());
            }

            if (LayerInfo(wl).isFullyConnected()) {
                //calculate a maximum value from layer by simulating GNA affine layer
                auto inputs = wl->insData.begin()->lock();
                auto outputs = *wl->outData.begin();
                auto num_rows_in = FROM_IR_DIM(inputs, 1);
                auto num_columns_in = FROM_IR_DIM(inputs, 2);
                auto num_rows_out = FROM_IR_DIM(outputs, 1);

                //generate random input
                std::vector<float> input = generateFloatNumbers(num_rows_in * num_columns_in, -inputMaxValue, inputMaxValue);

                //get weights and convert them to float
                auto weights = std::vector<float>(wl->_weights->size());
                memcpy(&weights[0], wl->_weights->buffer().as<float*>(), weights.size());

                //get biases and convert them to float
                auto biases = std::vector<float>();
                if (wl->_biases) {
                    biases.resize(wl->_biases->size());
                    memcpy(&biases[0], wl->_biases->buffer().as<float*>(), wl->_biases->size());
                }

                //simulate GNA affine layer
                auto result = simulateAffine(weights, input, biases,
                    num_rows_out, num_rows_in, quant->_weights_quant.GetScale(),
                    num_rows_in, num_columns_in, quant->_src_quant.GetScale());

                //find minimum and maximum values
                auto outputValue = findMinMaxValues(result);
                minOutputValue = outputValue.first;
                maxOutputValue = outputValue.second;
            } else if (LayerInfo(wl).isConvolution()) {
                //calculate a maxium value from layer by simulating GNA convolution layer
                auto convLayer = dynamic_cast<InferenceEngine::ConvolutionLayer*>(wl);
                if (!convLayer) {
                    THROW_GNA_EXCEPTION << "Incorrect Convolution Layer pointer  \n";
                }

                auto inputs = wl->insData.begin()->lock();
                auto outputs = *wl->outData.begin();
                if (inputs->getLayout() != InferenceEngine::Layout::NHWC &&
                    inputs->getLayout() != InferenceEngine::Layout::NCHW &&
                    inputs->getLayout() != InferenceEngine::Layout::NC) {
                    THROW_GNA_EXCEPTION << convLayer->name << " layer with layout "
                        << inputs->getLayout() << " isn't currently supported on GNA";
                }
                if (inputs->getLayout() != outputs->getLayout()) {
                    THROW_GNA_EXCEPTION << convLayer->name << " layer I/O layout mismatch: "
                        << inputs->getLayout() << " vs " << outputs->getLayout();
                }

                auto in_order = getFromIRDimsOrderNCHW(inputs->getLayout());
                auto in_batch = FROM_IR_DIM(inputs, in_order[0]);
                auto in_channels = FROM_IR_DIM(inputs, in_order[1]);
                auto in_height = FROM_IR_DIM(inputs, in_order[2]);
                auto in_width = FROM_IR_DIM(inputs, in_order[3]);

                auto out_order = getFromIRDimsOrderNCHW(outputs->getLayout());
                auto out_batch = FROM_IR_DIM(outputs, out_order[0]);
                auto out_channels = FROM_IR_DIM(outputs, out_order[1]);
                auto out_height = FROM_IR_DIM(outputs, out_order[2]);
                auto out_width = FROM_IR_DIM(outputs, out_order[3]);

                if (in_batch != 1 || out_batch != 1) {
                    //If this check is removed, the convolution calculations needs to be corrected.
                    THROW_GNA_EXCEPTION << convLayer->name << " layer with batch size not equals 1 is not supported";
                }
                if ((in_channels > 1) && (in_height > 1) && (in_width > 1)) {
                    THROW_GNA_EXCEPTION << convLayer->name << " layer with 3D input is not supported on GNA";
                }

                std::vector<float> convertedKernels;
                for (uint32_t k = 0; k < convLayer->_out_depth; k++) {
                    //float* ptr_filt_current
                    //    = convLayer->_weights->cbuffer().as<float*>() +
                    //    k * in_channels * convLayer->_kernel[InferenceEngine::eDIMS_AXIS::X_AXIS];
                                    //auto convertedKernels = transposeMatrix(ptr_filt_current, in_channels,
                    //    convLayer->_kernel[InferenceEngine::eDIMS_AXIS::X_AXIS]);

                    auto kernel = std::vector<float>(convLayer->_kernel_x * in_channels);
                    memcpy(&kernel[0], convLayer->_weights->cbuffer().as<float*>() + k * kernel.size(), kernel.size() * sizeof(float));

                    //if (inputs->getLayout() == InferenceEngine::Layout::NHWC) {
                    //    kernel = convertNHWCtoNCHW(kernel, convLayer->_kernel_x, 1, in_channels, 1);
                    //}
                    //kernel = transposeMatrix(&kernel[0], in_channels, convLayer->_kernel_x);
                    convertedKernels.insert(convertedKernels.end(), kernel.begin(), kernel.end());
                }

                std::vector<float> biases;
                if (wl->_biases) {
                    biases.resize(out_channels);
                    memcpy(&biases[0], wl->_biases->buffer().as<float*>(), wl->_biases->size());
                }

                // perform covolution on random data
                std::vector<float> input = generateFloatNumbers(in_width * in_height * in_channels, -inputMaxValue, inputMaxValue);
                auto result = simulateConvolution(input, convertedKernels, biases,
                    convLayer->_stride_x, convLayer->_stride_y, 1,
                    convLayer->_padding_x, convLayer->_padding_y, 0,
                    in_width, in_height, in_channels, quantDataForInputLayer->_dst_quant.GetScale(),
                    out_width, out_height, out_channels,
                    convLayer->_kernel_x, 1, quant->_weights_quant.GetScale());

                // find minimum and maximum values
                auto outputValue = findMinMaxValues(result);
                minOutputValue = outputValue.first;
                maxOutputValue = outputValue.second;
            }

            maxOutputValue = fp32eq(maxOutputValue, 0.0f) ? inputMaxValue : maxOutputValue;

            quant->_dst_quant.SetMinValues({ minOutputValue });
            quant->_dst_quant.SetMaxValues({ maxOutputValue });
        }

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
