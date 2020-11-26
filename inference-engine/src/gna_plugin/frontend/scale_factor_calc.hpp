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
    const float MEMORY_INIT_MAX_VALUE = 8.0f;
    const float WEIGHTS_SCALE_FACTOR = 8.0f;
    const size_t MAX_NUM_MEMORY_LAYER_VISITS = 3;

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
     * @brief Calculates and updates scale factor for activation layers
     * @param cnnLayer Input layar for which scale factor is calculated
     * @param result Return paramater with a layer that scale factor calculator needs to restart from
     * @return Returns true if a scale factor was successfully calculated
     */
    bool updateActivationLayerScale(InferenceEngine::CNNLayer* cnnLayer, ScaleFactorUpdateResult& result) {
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);
        auto layerInfo = LayerInfo(*cnnLayer);
        auto inputMinValue = quant->_src_quant.GetMinValues().front();
        auto inputMaxValue = quant->_src_quant.GetMaxValues().front();

        if (1 || quant->_dst_quant.GetMaxValues().empty()) {
            auto outputMinValue = inputMinValue;
            auto outputMaxValue = inputMaxValue;

            if (layerInfo.isPower()) {
                auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer const*>(cnnLayer);
                if (!powerLayer) {
                    THROW_IE_EXCEPTION << "Incorrect Power Layer pointer \n";
                }

                // Special case: when power exponent is not integral number, it means that
                // minimum number must be equal or greater than 0.
                if (!fp32eq(fmod(powerLayer->power, 1.0f), 0.0f)) {
                    quant->_src_quant.SetMinValues({ 0.0f });
                    inputMinValue = quant->_src_quant.GetMinValues().front();
                }

                outputMinValue = pow(inputMinValue * powerLayer->scale + powerLayer->offset, powerLayer->power);
                outputMaxValue = pow(inputMaxValue * powerLayer->scale + powerLayer->offset, powerLayer->power);
            } else if (layerInfo.isSigmoid()) {
                outputMinValue = sigmoid(inputMinValue);
                outputMaxValue = sigmoid(inputMaxValue);
            } else if (layerInfo.isTanh()) {
                outputMinValue = tanh(inputMinValue);
                outputMaxValue = tanh(inputMaxValue);
            } else if (layerInfo.isExp()) {
                outputMinValue = exp(inputMinValue);
                outputMaxValue = exp(inputMaxValue);
            } else if (layerInfo.isLog()) {
                inputMinValue = 1.0f / quant->_src_quant.GetScale();
                outputMinValue = log(inputMinValue);
                outputMaxValue = log(inputMaxValue);
            } else if (layerInfo.isNegLog()) {
                inputMinValue = 1.0f / quant->_src_quant.GetScale();
                outputMinValue = neglog(inputMinValue);
                outputMaxValue = neglog(inputMaxValue);
            } else if (layerInfo.isNegHalfLog()) {
                inputMinValue = 1.0f / quant->_src_quant.GetScale();
                outputMinValue = neghalflog(inputMinValue);
                outputMaxValue = neghalflog(inputMaxValue);
            } else if (layerInfo.isClamp()) {
                outputMinValue = std::max(static_cast<float>(KALDI_LSTM_CLIP_LOWER), inputMinValue);
                outputMaxValue = std::min(static_cast<float>(KALDI_LSTM_CLIP_UPPER), inputMaxValue);
            } else if (layerInfo.isAbs()) {
                outputMinValue = std::abs(inputMinValue);
                outputMaxValue = std::abs(inputMaxValue);
            } else if (layerInfo.isSoftSign()) {
                outputMinValue = softsign(inputMinValue);
                outputMaxValue = softsign(inputMaxValue);
            } else if (layerInfo.isSign()) {
                outputMinValue = sign(inputMinValue);
                outputMaxValue = sign(inputMaxValue);
            } else if (layerInfo.isIdentity()) {
                outputMinValue = inputMinValue;
                outputMaxValue = inputMaxValue;
            } else if (layerInfo.isRelu()) {
                auto reluLayer = dynamic_cast<InferenceEngine::ReLULayer*>(cnnLayer);
                if (!reluLayer) {
                    THROW_IE_EXCEPTION << "Incorrect Relu Layer pointer \n";
                }

                outputMinValue = relu(inputMinValue, reluLayer->negative_slope);
                outputMaxValue = relu(inputMaxValue, reluLayer->negative_slope);
            } else if (layerInfo.isLeakyRelu()) {
                outputMinValue = leaky_relu(inputMinValue);
                outputMaxValue = leaky_relu(inputMaxValue);
            } else {
                THROW_IE_EXCEPTION << "Unknown activation function\n";
            }

            if (layerInfo.isSign()) {
                // sign can only return 3 values: -1, 0 and 1
                quant->_dst_quant.SetLevels(1);
            } else {
                quant->_dst_quant.SetLevels(MAX_VALUE);
            }
            
            quant->_dst_quant.SetMinValues({ outputMinValue });
            quant->_dst_quant.SetMaxValues({ outputMaxValue });
        }

        auto scaleFactor = activation_scale_factor;
        if (!quant->_dst_quant.GetMaxValues().empty()) {
            auto minOutValue = quant->_dst_quant.GetMinValues().front();
            auto maxOutValue = quant->_dst_quant.GetMaxValues().front();
            auto maxAbsOutValue = std::max(std::abs(minOutValue), std::abs(maxOutValue));
            scaleFactor = quant->_dst_quant.GetLevels() / maxAbsOutValue;
        }

        if (layerInfo.isIdentity() || layerInfo.isRelu() || layerInfo.isSign() ||
            layerInfo.isAbs() || layerInfo.isClamp()) {
            auto prevLayer = CNNNetPrevLayer(cnnLayer);
            while (LayerInfo(prevLayer).isNonFunctional() && CNNNetHasPrevLayer(prevLayer.get())) {
                prevLayer = CNNNetPrevLayer(prevLayer);
            }

            if (prevLayer == nullptr || prevLayer.get() == cnnLayer || LayerInfo(prevLayer).isNonFunctional()) {
                THROW_IE_EXCEPTION << "Activation layer '" << cnnLayer->name << "' does not have functional previous layer\n";
            }

            if (LayerInfo(prevLayer).isWeightable() &&
                (!fp32eq(quant->_dst_quant.GetScale(), scaleFactor) || layerInfo.isSign())) {
                auto wl = dynamic_cast<InferenceEngine::WeightableLayer*>(prevLayer.get());
                if (!wl) {
                    THROW_GNA_EXCEPTION << "Incorrect Weightable Layer pointer\n";
                }

                auto prevLayerQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
                auto maxWeightsVal = std::max(std::abs(prevLayerQuant->_weights_quant.GetMinValues().front()),
                    std::abs(prevLayerQuant->_weights_quant.GetMaxValues().front()));
                auto srcToDstRatio = prevLayerQuant->_src_quant.GetScale() / scaleFactor;
                if ((fp32eq(maxWeightsVal, 1.0f) &&
                    fp32eq(prevLayerQuant->_weights_quant.GetScale(), 1.0f) &&
                    srcToDstRatio < WEIGHTS_SCALE_FACTOR) || layerInfo.isSign()) {

                    gnalog() << "[INFO] Updating weights scale factor for '" << prevLayer->name << "' layer. Change from "
                        << prevLayerQuant->_weights_quant.GetScale() << " to " << WEIGHTS_SCALE_FACTOR << "\n";

                    prevLayerQuant->_weights_quant.SetScale(WEIGHTS_SCALE_FACTOR);
                    prevLayerQuant->_dst_quant.SetScale(prevLayerQuant->_src_quant.GetScale() * prevLayerQuant->_weights_quant.GetScale());
                    result = ScaleFactorUpdateResult(prevLayer.get());
                }
            }
        }

        if (quant->_dst_quant.IsScaleSet() && !fp32eq(quant->_dst_quant.GetScale(), scaleFactor)) {
            gnalog() << "[INFO] Updating scale factor for '" << cnnLayer->name << "' layer. Change from "
                << quant->_dst_quant.GetScale() << " to " << scaleFactor << "\n";
        }

        quant->_dst_quant.SetScale(scaleFactor);

        return true;
    }

    /**
     * @brief Calculates and updates scale factor for memory read and assign layers
     * @param cnnLayer Input layar for which scale factor is calculated
     * @param result Return paramater with a layer that scale factor calculator needs to restart from
     * @return Returns true if a scale factor was successfully calculated
     */
    bool updateMemoryLayerScale(InferenceEngine::CNNLayer * cnnLayer, ScaleFactorUpdateResult& result) {
        bool hasPreviousLayer = CNNNetHasPrevLayer(cnnLayer);
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);

        if (!hasPreviousLayer) {
            // it is memory assign layer
            if (0 && !quant->_dst_quant.GetMaxValues().empty()) {
                // The min and max values can be set when FQ layer was in a network
                auto minValue = quant->_dst_quant.GetMinValues().front();
                auto maxValue = quant->_dst_quant.GetMaxValues().front();
                quant->_dst_quant.SetScale((quant->_dst_quant.GetLevels() - 1) / (maxValue - minValue));
            } else if(quant->_dst_quant.GetMaxValues().empty()) {
                // Initialize quant statistics with initial values if memory read layer.
                quant->_dst_quant.SetMinValues({ -MEMORY_INIT_MAX_VALUE });
                quant->_dst_quant.SetMaxValues({ MEMORY_INIT_MAX_VALUE });
            }

            quant->_src_quant = quant->_dst_quant;

            return true;
        }

        static std::map<std::string, size_t> visitedMemoryLayers;
        if (visitedMemoryLayers.find(cnnLayer->name) == visitedMemoryLayers.end()) {
            visitedMemoryLayers[cnnLayer->name] = 0;
        }
        auto &visitCount = visitedMemoryLayers[cnnLayer->name];
        ++visitCount;
        gnalog() << "[INFO] Memory layer '" << cnnLayer->name << "' was visited " << visitCount << " times\n";

        auto prevLayer = CNNNetPrevLayer(cnnLayer);
        auto prevLayerQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
        // locating corresponding memory layers with same ID
        for (auto&& input : CNNNetGetAllInputLayers(cnnLayer)) {
            LayerInfo ll(input);
            if (!ll.isMemory() ||
                !InferenceEngine::details::CaselessEq<std::string>()(input->params["id"], cnnLayer->params["id"])) {
                continue;
            }

            auto quantSibling = InferenceEngine::getInjectedData<QuantizedLayerParams>(input);

            // after restarting from memory input - quant is fine
            if (fp32eq(quantSibling->_dst_quant.GetScale(), prevLayerQuant->_dst_quant.GetScale())) {
                quant->_src_quant = quant->_dst_quant = prevLayerQuant->_dst_quant;
                quantSibling->_src_quant = quantSibling->_dst_quant = prevLayerQuant->_dst_quant;
                continue;
            }

            if (visitCount > MAX_NUM_MEMORY_LAYER_VISITS && quantSibling->_dst_quant.IsScaleSet()) {
                // means we already restarted propagation input memory layer
                // need to search for requantiseable layer prior memory output layer
                InferenceEngine::CNNLayerPtr restartedLayer;

                gnalog() << "[WARNING] Memory layer '" << input->name << "' scale factor: " << quantSibling->_dst_quant.GetScale()
                    << " doesn't match its outputs counterpart '" << cnnLayer->name << "' scale factor: " << prevLayerQuant->_dst_quant.GetScale() << "\n";
                gnalog() << "[UFS] searching for quantizeable input layer for '" << cnnLayer->name << "'\n";

                CNNNetDFS(InferenceEngine::CNNLayerPtr(cnnLayer, [](InferenceEngine::CNNLayer*) {}),
                    [&restartedLayer, cnnLayer](InferenceEngine::CNNLayerPtr layer) {
                        gnalog() << "[UFS] from: '" << cnnLayer->name << "' reached '" << layer->name << "'";
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
                    THROW_GNA_EXCEPTION << "Cannot requantize input to '" << cnnLayer->name << "'";
                }

                auto restartedLayerQuantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*restartedLayer);

                auto restarLayerInfo = LayerInfo(restartedLayer);
                if (!restarLayerInfo.isActivation()) {
                    THROW_GNA_EXCEPTION << "Restarted layer cannot be requantized. Layer '" << restartedLayer->name << "'";
                }

                if (restartedLayerQuantData->_dst_quant.IsScaleSet()
                    && restartedLayerQuantData->_dst_quant.GetScale() < quantSibling->_dst_quant.GetScale()) {
                    gnalog() << "[WARNING] Possible output saturation. '"
                        << restartedLayer->name << "' layer scale factor change from "
                        << restartedLayerQuantData->_dst_quant.GetScale() << " to "
                        << quantSibling->_dst_quant.GetScale() << "\n";
                }

                gnalog() << "[WARNING] '" << restartedLayer->name << "' layer will be requantized. Output scale factor change from "
                    << restartedLayerQuantData->_dst_quant.GetScale() << " to "
                    << quantSibling->_dst_quant.GetScale() << "\n";

                // requantize activation by just changing it's output scale factor
                restartedLayerQuantData->_dst_quant.SetScale(quantSibling->_dst_quant.GetScale());
                result = ScaleFactorUpdateResult(restartedLayer.get());
                return true;
            }

            gnawarn() << "[INFO] quantization : input scale factor (" << prevLayerQuant->_dst_quant.GetScale() << ")"
                << " for '" << cnnLayer->name << "', that is child of '" << prevLayer->name << "' doesnt match : "
                << quantSibling->_dst_quant.GetScale() << ", restarting from corresponding memory: '" << input->name
                << "'" << std::endl;

            // try updating memory input layer scale factor and restart from it
            quant->_src_quant = quant->_dst_quant = prevLayerQuant->_dst_quant;
            quantSibling->_src_quant = quantSibling->_dst_quant = prevLayerQuant->_dst_quant;
            result = ScaleFactorUpdateResult(input.get());
            return true;
        }

        return true;
    }

    /**
     * @brief Calculates and updates scale factor for constant layers
     * @param cnnLayer Input layar for which scale factor is calculated
     * @return Returns true if a scale factor was successfully calculated
    */
    bool updateConstLayerScale(InferenceEngine::CNNLayer* cnnLayer) {
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);
        auto blob = cnnLayer->blobs["custom"];
        auto blob_precision = blob->getTensorDesc().getPrecision();

        if (quant->_dst_quant.GetMinValues().empty()) {
            auto size = blob->size();
            auto min_val = std::numeric_limits<float>::max();
            auto max_val = std::numeric_limits<float>::min();
            if (blob_precision == InferenceEngine::Precision::I16) {
                auto buf = blob->buffer().as<int16_t*>();
                for (size_t i = 0; i < size; i++) {
                    min_val = std::min(min_val, static_cast<float>(buf[i]));
                    max_val = std::max(max_val, static_cast<float>(buf[i]));
                }
            } else if (blob_precision == InferenceEngine::Precision::U8) {
                auto buf = blob->buffer().as<int8_t*>();
                for (size_t i = 0; i < size; i++) {
                    min_val = std::min(min_val, static_cast<float>(buf[i]));
                    max_val = std::max(max_val, static_cast<float>(buf[i]));
                }
            } else if (blob_precision == InferenceEngine::Precision::FP16) {
                blob = make_fp32_blob(blob);
                auto buf = blob->buffer().as<float*>();
                for (size_t i = 0; i < size; i++) {
                    min_val = std::min(min_val, buf[i]);
                    max_val = std::max(max_val, buf[i]);
                }
            } else if (blob_precision == InferenceEngine::Precision::FP32) {
                auto buf = blob->buffer().as<float*>();
                for (size_t i = 0; i < size; i++) {
                    min_val = std::min(min_val, buf[i]);
                    max_val = std::max(max_val, buf[i]);
                }
            } else {
                THROW_IE_EXCEPTION << "Unsupported precision. Layer: '" << cnnLayer->name << "'";
            }

            quant->_dst_quant.SetMinValues({ min_val });
            quant->_dst_quant.SetMaxValues({ max_val });
        }


        if (blob_precision != InferenceEngine::Precision::FP32 &&
            blob_precision != InferenceEngine::Precision::FP16) {
            quant->_dst_quant.SetScale(1.0f);
            return true;
        }

        auto maxAbsVal = std::max(std::abs(quant->_dst_quant.GetMinValues().front()),
            std::abs(quant->_dst_quant.GetMaxValues().front()));
        if (fp32eq(maxAbsVal, 0.0f)) {
            // TODO: Investigate what should be the scale in such cases (31910)
            quant->_dst_quant.SetScale(quant->_src_quant.GetScale());
        } else {
            auto scale_val = static_cast<float>(MAX_VALUE) / maxAbsVal;
            quant->_dst_quant.SetScale(scale_val);
        }

        quant->_src_quant = quant->_dst_quant;
        return true;
    }

    /**
    * @brief Calculates and updates scale factor for power layer that is not activation (exponent is 1.0)
    * @param cnnLayer Input layar for which scale factor is calculated
    * @param weightsSize Size of weights
    * @return Returns true if a scale factor was successfully calculated
    */
    bool updatePowerLayerScale(InferenceEngine::CNNLayer* cnnLayer, int weightsSize) {
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);
        auto powerLayer = dynamic_cast<InferenceEngine::PowerLayer const*>(cnnLayer);
        if (!powerLayer) {
            THROW_IE_EXCEPTION << "Incorrect Power Layer pointer \n";
        }

        auto powerScale = std::abs(powerLayer->scale);
        auto powerOffset = std::abs(powerLayer->offset);

        auto weightScaleRange = 0.0f;
        switch (weightsSize) {
        case 1:
            weightScaleRange = MAX_VAL_1B_WEIGHT;
            break;
        case 2:
            weightScaleRange = MAX_VAL_2B_WEIGHT;
            break;
        default:
            THROW_GNA_EXCEPTION << "Unsupported weights size of: " << weightsSize;
            break;
        }

        auto weightScaleFactor = weightScaleRange / powerScale;
        if (fp32eq(powerScale, 1.0f)) {
            weightScaleFactor = 1.0f;
        }

        quant->_weights_quant.SetScale(weightScaleFactor);

        auto minOutputValue = quant->_src_quant.GetMinValues().front() * powerScale + powerOffset;
        auto maxOutputValue = quant->_src_quant.GetMaxValues().front() * powerScale + powerOffset;

        auto scaleFactor = quant->_src_quant.GetScale() * weightScaleFactor;
        if (quant->_dst_quant.IsScaleSet() && !fp32eq(quant->_dst_quant.GetScale(), scaleFactor)) {
            gnalog() << "[INFO] Updating scale factor for '" << cnnLayer->name << "' layer. Change from "
                << quant->_dst_quant.GetScale() << " to " << scaleFactor << "\n";
        }

        quant->_dst_quant.SetScale(scaleFactor);
        quant->_dst_quant.SetMinValues({ minOutputValue });
        quant->_dst_quant.SetMaxValues({ maxOutputValue });

        return true;
    }

    /**
    * @brief Calculates and updates scale factor for pooling layer
    * @param cnnLayer Input layar for which scale factor is calculated
    * @return Returns true if a scale factor was successfully calculated
    */
    bool updatePoolingLayerScale(InferenceEngine::CNNLayer* cnnLayer) {
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);
        auto poolingLayer = dynamic_cast<InferenceEngine::PoolingLayer const*>(cnnLayer);
        if (!poolingLayer) {
            THROW_IE_EXCEPTION << "Incorrect Pooling Layer pointer \n";
        }

        if (!LayerInfo(cnnLayer).isMaxPooling()) {
            THROW_GNA_EXCEPTION << "Layer :" << cnnLayer->name << " not supported";
        }

        quant->_dst_quant = quant->_src_quant;
        if (0) {
            auto inputs = poolingLayer->insData.begin()->lock();
            auto outputs = *poolingLayer->outData.begin();

            auto getFromIRDimsOrderNCHW = [](InferenceEngine::Layout layout) -> std::vector<std::size_t> {
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
            };

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

            const int seed = 0;
            std::mt19937 gen(static_cast<float>(seed));
            auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
                std::vector<float> res;

                std::uniform_real_distribution<float> dist(min, max);
                for (int i = 0; i < vec_len; i++)
                    res.emplace_back(static_cast<float>(dist(gen)));

                return res;
            };

            auto minInputVal = quant->_src_quant.GetMinValues().front();
            auto maxInputVal = quant->_src_quant.GetMaxValues().front();

            std::vector<float> input = generateFloatNumbers(in_width * in_height * in_channels, minInputVal, maxInputVal);
            std::vector<float> output(out_width * out_height * out_channels);
            auto cStep = in_channels / out_channels;
            auto hStep = in_height / out_height;
            auto wStep = in_width / out_width;

            for (uint32_t oc = 0; oc < out_channels; ++oc) {
                for (uint32_t oh = 0; oh < out_height; ++oh) {
                    for (uint32_t ow = 0; ow < out_width; ++ow) {
                        size_t oidx = oc * out_height * out_width + oh * out_width + ow;
                        output[oidx] = std::numeric_limits<float>::min();

                        for (uint32_t kc = 0; kc < cStep; ++kc) {
                            for (uint32_t kh = 0; kh < hStep; ++kh) {
                                for (uint32_t kw = 0; kw < wStep; ++kw) {
                                    size_t ic = oc * cStep + kc;
                                    size_t ih = oh * hStep + kh;
                                    size_t iw = ow * wStep + kw;
                                    size_t iidx = ic * in_height * in_width + ih * in_width + iw;
                                    output[oidx] = std::max(input[iidx], output[oidx]);
                                }
                            }
                        }
                    }
                }
            }

            auto minVal = std::numeric_limits<float>::max();
            auto maxVal = std::numeric_limits<float>::min();
            for (size_t i = 0; i < output.size(); i++) {
                minVal = std::min(minVal, static_cast<float>(output[i]));
                maxVal = std::max(maxVal, static_cast<float>(output[i]));
            }

            quant->_dst_quant.SetScale(quant->_src_quant.GetScale());
            quant->_dst_quant.SetMinValues({ minVal });
            quant->_dst_quant.SetMaxValues({ maxVal });
        }


        return true;
    }

 public :
    bool operator()(InferenceEngine::CNNLayer *cnnLayer, int weightsSize, ScaleFactorUpdateResult &result) {
        if ( !cnnLayer ) {
            THROW_IE_EXCEPTION << "Incorrect layer pointer \n";
        }

        auto hasPreviousLayer = CNNNetHasPrevLayer(cnnLayer);
        auto layerInfo = LayerInfo(*cnnLayer);
        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*cnnLayer);

        // by default layer is pass thru its scale factor
        QuantizedLayerParams* inputQuant = nullptr;
        if (hasPreviousLayer) {
            inputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(CNNNetPrevLayer(cnnLayer));
            if (!inputQuant) {
                THROW_GNA_EXCEPTION << "layer: '" << CNNNetPrevLayer(cnnLayer)->name << "' not quantized";
            }

            if (inputQuant->_dst_quant.IsScaleSet() && quant->_src_quant.IsScaleSet() &&
                quant->_src_quant.IsEqual(inputQuant->_dst_quant)) {
                //avoid recalculation of scale factors if quant paramaters did not change for previous layer
                return true;
            }

            quant->_src_quant = inputQuant->_dst_quant;
            if (quant->_src_quant.GetMinValues().size() != 1 || quant->_src_quant.GetMaxValues().size() != 1) {
                THROW_GNA_EXCEPTION << "Invalid number of min and max input values for '"
                    << cnnLayer->name << "' layer. Min size: "
                    << quant->_src_quant.GetMinValues().size()
                    << ", max size: " << quant->_src_quant.GetMaxValues().size();
            }
        }

        if ((!quant->_dst_quant.GetMinValues().empty() || !quant->_dst_quant.GetMinValues().empty()) &&
            (quant->_dst_quant.GetMinValues().size() != 1 || quant->_dst_quant.GetMaxValues().size() != 1)) {
            THROW_GNA_EXCEPTION << "Invalid number of min and max output values for '"
                << cnnLayer->name << "' layer. Min size: "
                << quant->_src_quant.GetMinValues().size()
                << ", max size: " << quant->_src_quant.GetMaxValues().size();
        }

        if (0 && !quant->_dst_quant.GetMaxValues().empty()) {
            // The min and max values can be set when FQ layer was in a network
            auto minValue = quant->_dst_quant.GetMinValues().front();
            auto maxValue = quant->_dst_quant.GetMaxValues().front();
            quant->_dst_quant.SetScale((quant->_dst_quant.GetLevels() - 1) / (maxValue - minValue));
            quant->_src_quant = quant->_dst_quant;
            return true;
        }

        if (layerInfo.isMemory()) {
            return updateMemoryLayerScale(cnnLayer, result);
        } else if (layerInfo.isConst()) {
            return updateConstLayerScale(cnnLayer);
        } else if (!layerInfo.isActivation() && layerInfo.isPower()) {
            return updatePowerLayerScale(cnnLayer, weightsSize);
        } else if (layerInfo.isActivation()) {
            return updateActivationLayerScale(cnnLayer, result);
        } else if (layerInfo.isPooling()) {
            return updatePoolingLayerScale(cnnLayer);
        } else if (layerInfo.isInput()) {
            // Scale factor calculation for i.e. input layer
            if (quant->_src_quant.GetMinValues().empty()) {
                auto maxValue = MAX_VALUE / quant->_src_quant.GetScale();
                quant->_src_quant.SetMinValues({ -maxValue });
                quant->_src_quant.SetMaxValues({ maxValue });
            }
            quant->_dst_quant = quant->_src_quant;
            return true;
        }

        if (!hasPreviousLayer) {
            THROW_GNA_EXCEPTION << "Expected layer that has previous layer. Layer: '" << cnnLayer->name << "'";
        }

        quant->_src_quant = inputQuant->_dst_quant;
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
            THROW_GNA_EXCEPTION << "Invalid number of min and max input values for '" << in0->name << "' layer."
                << " Min: " << quantParams0->_dst_quant.GetMinValues().size()
                << ", max: " << quantParams0->_dst_quant.GetMaxValues().size();
        }

        if (quantParams1->_dst_quant.GetMinValues().size() != 1 ||
            quantParams1->_dst_quant.GetMaxValues().size() != 1) {
            THROW_GNA_EXCEPTION << "Invalid number of min and max input values for '" << in1->name << "' layer."
                << " Min: " << quantParams1->_dst_quant.GetMinValues().size()
                << ", max: " << quantParams1->_dst_quant.GetMaxValues().size();
        }

        auto min0 = quantParams0->_dst_quant.GetMinValues().front();
        auto max0 = quantParams0->_dst_quant.GetMaxValues().front();
        auto min1 = quantParams1->_dst_quant.GetMinValues().front();
        auto max1 = quantParams1->_dst_quant.GetMaxValues().front();

        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*eltwiseLayer);

        switch (eltwiseLayer->_operation) {
            case InferenceEngine::EltwiseLayer::Prod: {
                quantData->_weights_quant = quantParams1->_dst_quant;

                auto scaleFactor = quantParams0->_dst_quant.GetScale() * quantParams1->_dst_quant.GetScale();
                if (quantData->_dst_quant.IsScaleSet() && !fp32eq(quantData->_dst_quant.GetScale(), scaleFactor)) {
                    gnalog() << "[INFO] Updating scale factor for '" << eltwiseLayer->name << "' layer. Change from "
                        << quantData->_dst_quant.GetScale() << " to " << scaleFactor << "\n";
                }

                quantData->_dst_quant.SetScale(scaleFactor);

                std::vector<float> vals;
                vals.push_back(min0 * min1);
                vals.push_back(min0 * max1);
                vals.push_back(max0 * min1);
                vals.push_back(max0 * max1);

                auto minVal = std::numeric_limits<float>::max();
                auto maxVal = std::numeric_limits<float>::min();
                for (size_t i = 0; i < vals.size(); i++) {
                    minVal = std::min(minVal, static_cast<float>(vals[i]));
                    maxVal = std::max(maxVal, static_cast<float>(vals[i]));
                }
                
                quantData->_dst_quant.SetMinValues({ minVal });
                quantData->_dst_quant.SetMaxValues({ maxVal });
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
                        LayerInfo(findPrevFunctional(in0)).has32BOutput())) {
                    std::swap(in0, in1);
                    std::swap(quantParams0, quantParams1);
                }
          
                // Weights are used to do scaling of input 0
                auto dstScale1 = static_cast<double>(quantParams1->_dst_quant.GetScale());
                auto dstScale0 = static_cast<double>(quantParams0->_dst_quant.GetScale());
                quantData->_weights_quant.SetScale(dstScale1 / dstScale0);

                auto scaleFactor = quantParams1->_dst_quant.GetScale();
                if (quantData->_dst_quant.IsScaleSet() && !fp32eq(quantData->_dst_quant.GetScale(), scaleFactor)) {
                    gnalog() << "[INFO] Updating scale factor for '" << eltwiseLayer->name << "' layer. Change from "
                        << quantData->_dst_quant.GetScale() << " to " << scaleFactor << "\n";
                }
                quantData->_dst_quant.SetScale(scaleFactor);

                // min and max output value estimation
                std::vector<float> vals;
                if(eltwiseLayer->_operation == InferenceEngine::EltwiseLayer::Sum) {
                    vals.push_back(min0 + min1);
                    vals.push_back(min0 + max1);
                    vals.push_back(max0 + min1);
                    vals.push_back(max0 + max1);
                } else if (eltwiseLayer->_operation == InferenceEngine::EltwiseLayer::Sub) {
                    vals.push_back(min0 - min1);
                    vals.push_back(min0 - max1);
                    vals.push_back(max0 - min1);
                    vals.push_back(max0 - max1);
                } else {
                    THROW_GNA_EXCEPTION << "Unknown Eltwise operation \n";
                }

                auto minVal = std::numeric_limits<float>::max();
                auto maxVal = std::numeric_limits<float>::min();
                for (size_t i = 0; i < vals.size(); i++) {
                    minVal = std::min(minVal, static_cast<float>(vals[i]));
                    maxVal = std::max(maxVal, static_cast<float>(vals[i]));
                }

                quantData->_dst_quant.SetMinValues({ minVal });
                quantData->_dst_quant.SetMaxValues({ maxVal });

                // Assuming that weights are encoded as int16. The maximum value cannot
                // maximum value of int16, otherwise it will overflow.
                auto weightsScale = std::abs(quantData->_weights_quant.GetScale());
                if (static_cast<int32_t>(weightsScale + 0.5) > std::numeric_limits<int16_t>::max()) {
                    // The way to reduce weight scale factor is to either:
                    // 1) descrese scale factor of input 1 (int32 input) - descrese precision of input 0 values
                    // 2) increase scale factor of input 0 - it will result in saturation of input 1 values
                    // The first method is selected below

                    auto reducer = static_cast<double>(weightsScale) / std::numeric_limits<int16_t>::max();
                    reducer = std::max(1.0, reducer);

                    gnawarn() << "[WARNING] Possible saturation of output results for '" << eltwiseLayer->name << "' layer."
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

                    auto reductionFactor = reducer;
                    while (!layersToCheck.empty()) {
                        auto layer = *layersToCheck.begin();
                        auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*layer);
                        auto info = LayerInfo(*layer);

                        if (info.isWeightable() ||
                            (info.isPower() && !info.isActivation())) {
                            auto newWeightScale = quantParams->_weights_quant.GetScale() / reductionFactor;
                            gnalog() << "[WARNING] Output scale for '" << layer->name << "' is reduced. "
                                << "Layer new weight scale: " << newWeightScale
                                << ", old weight scale: " << quantParams->_weights_quant.GetScale()
                                << "\n";
                            quantParams->_weights_quant.SetScale(newWeightScale);
                            quantParams->_dst_quant.SetScale(quantParams->_weights_quant.GetScale()* quantParams->_src_quant.GetScale());
                            result = ScaleFactorUpdateResult(layer);
                            return true;
                        } else if (info.isActivation()) {
                            auto newOutputScale = quantParams->_dst_quant.GetScale() / reductionFactor;
                            gnalog() << "[WARNING] Output scale for '" << layer->name << "' is reduced. "
                                << "Layer new output scale: " << newOutputScale
                                << ", old output scale: " << quantParams->_dst_quant.GetScale()
                                << "\n";
                            quantParams->_dst_quant.SetScale(newOutputScale);
                            result = ScaleFactorUpdateResult(layer);
                            return true;
                        } else if (info.isConst()) {
                            auto newOutputScale = quantParams->_dst_quant.GetScale() / reductionFactor;
                            gnalog() << "[WARNING] Output scale for '" << layer->name << "' is reduced. "
                                << "Layer new input and output scale: " << newOutputScale
                                << ", old new input and output scale: " << quantParams->_dst_quant.GetScale()
                                << "\n";
                            quantParams->_src_quant.SetScale(newOutputScale);
                            quantParams->_dst_quant.SetScale(newOutputScale);
                            result = ScaleFactorUpdateResult(layer);
                            return true;
                        } else if (info.isMemory() || info.isInput()) {
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

                    THROW_GNA_EXCEPTION << "Scale factor for '" << eltwiseLayer->name << "' layer cannot be reduced\n";
                } else if (quantParams1->_dst_quant.GetScale() < quantParams0->_dst_quant.GetScale() ||
                    (std::abs(fmod(weightsScale, 1.0f)) > 0.05f && std::abs(fmod(weightsScale, 1.0f)) < 0.95f)) {
                    // Weights scale cannot be represented as integer without loss of precision.
                    // It is important to preserve remainder especially for small weight scales.
                    // If such case happen, input 0 scale has to be reduced.
                    auto newScale = static_cast<double>(quantParams1->_dst_quant.GetScale());
                    if (weightsScale > 1) {
                        auto roundedWeightScale = ceil(static_cast<double>(weightsScale));
                        newScale = newScale / roundedWeightScale;
                    }

                    if (fp32eq(newScale, quantParams0->_dst_quant.GetScale())) {
                        return true;
                    }

                    auto prevLayer = in0;
                    while (prevLayer != nullptr) {
                        auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*prevLayer);
                        LayerInfo info(prevLayer);
                        if (info.isActivation() || info.isConst() || info.isMemory()) {
                            gnalog() << "[WARNING] Output scale for '" << prevLayer->name << "' is reduced. "
                                << "Layer new output scale: " << newScale
                                << ", old output scale: " << quantParams->_dst_quant.GetScale()
                                << "\n";
                            quantParams->_dst_quant.SetScale(newScale);
                            result = ScaleFactorUpdateResult(prevLayer.get());
                            return true;
                        }
                        else if (info.isNonFunctional() || info.isSplit()) {
                            prevLayer = CNNNetPrevLayer(prevLayer);
                            continue;
                        }

                        THROW_GNA_EXCEPTION << "Scale factor for '" << eltwiseLayer->name << "' layer cannot be reduced\n";
                    }
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

        auto ignoreLayerQuantParams = [concatLayer](InferenceEngine::CNNLayerPtr& layer) -> bool {
            auto prevLayer = layer;
            while (prevLayer) {
                auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
                if (!quantParams->_dst_quant.IsScaleSet()) {
                    // Quantization parameters are not yet calculated for thus layer
                    // Default values are used and they should be ignored
                    return true;
                }

                GNAPluginNS::LayerInfo info(prevLayer);
                if (info.isInput() || info.isConst() || info.isConcat()) {
                    return false;
                }

                if (info.isMemory()) {
                    // search for output layer that corresponds to input memory layer to requanitize it
                    InferenceEngine::CNNLayerSet outputLayers;
                    std::unordered_set<InferenceEngine::CNNLayer*> allLayers;

                    InferenceEngine::CNNLayerPtr layerPtr(concatLayer, [](InferenceEngine::CNNLayer*) {});

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
                            !InferenceEngine::details::CaselessEq<std::string>()(output->params["id"], prevLayer->params["id"])) {
                            continue;
                        }

                        auto prevLayer2 = CNNNetPrevLayer(output);
                        if (prevLayer2.get() == concatLayer) {
                            // Memory assign layer is connected to the same concat layer as memory read
                            return true;
                        }

                        GNAPluginNS::LayerInfo info2(prevLayer2);
                        if (info2.isNonFunctional() || info2.isCopy()) {
                            prevLayer2 = CNNNetPrevLayer(prevLayer2);
                        }

                        break;
                        
                    }

                    return false;
                }

                prevLayer = CNNNetPrevLayer(prevLayer);
            }

            return false;
        };

        auto concatQuantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*concatLayer);
        std::vector<InferenceEngine::CNNLayerPtr> inputLayers;
        std::vector<bool> ignoreLayers;
        auto minInputValue = std::numeric_limits<float>::max();
        auto maxInputValue = std::numeric_limits<float>::min();
        for (auto inputIdx = 0; inputIdx != concatLayer->insData.size(); inputIdx++) {
            inputLayers.push_back(InferenceEngine::CNNNetPrevLayer(concatLayer, inputIdx));

            auto quantParamsIn = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayers[inputIdx]);

            if (quantParamsIn->_dst_quant.GetMinValues().size() != 1 ||
                quantParamsIn->_dst_quant.GetMaxValues().size() != 1) {
                THROW_GNA_EXCEPTION << "Incorrect number of maximum values for '" << inputLayers[inputIdx]->name << "' layer. "
                    << "Min: " << quantParamsIn->_src_quant.GetMinValues().size()
                    << ", max: " << quantParamsIn->_src_quant.GetMaxValues().size();
            }

            ignoreLayers.push_back(ignoreLayerQuantParams(inputLayers[inputIdx]));
            if (ignoreLayers[inputIdx]) {
                continue;
            }

            minInputValue = std::min(minInputValue, quantParamsIn->_dst_quant.GetMinValues().front());
            maxInputValue = std::max(maxInputValue, quantParamsIn->_dst_quant.GetMaxValues().front());
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
            concatQuantData->_dst_quant.SetMinValues({ minInputValue });
            concatQuantData->_dst_quant.SetMaxValues({ maxInputValue });
            concatQuantData->_src_quant = concatQuantData->_dst_quant;
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
                    THROW_GNA_EXCEPTION << "Two Input layers '" << (*firstInputIt)->name
                        << "' and '" << (*nextInputIt)->name << "' have different scales in concat!!! \n";
                }
            }
        }

        // find a source quant value
        // - 1st candidate - input layer
        // - 2nd candidate - non-input layer with minimum scale factor
        if (firstInputIt != inputLayers.end()) {
            sourceQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*firstInputIt);
        } else {
            for (size_t i = 0; i < inputLayers.size(); ++i) {
                if (ignoreLayers[i]) {
                    continue;
                }

                auto quantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayers[i]);

                auto inputMaxValue = std::max(std::abs(quantParams->_dst_quant.GetMinValues().front()),
                    std::abs(quantParams->_dst_quant.GetMaxValues().front()));

                auto sourceMinVal = sourceQuantParams ? sourceQuantParams->_dst_quant.GetMinValues().front(): 0;
                auto sourceMaxVal = sourceQuantParams ? sourceQuantParams->_dst_quant.GetMaxValues().front(): 0;
                auto sourceMaxValue = std::max(std::abs(sourceMinVal), std::abs(sourceMaxVal));

                if (inputMaxValue > sourceMaxValue) sourceQuantParams = quantParams;
            }

            sourceQuantParams = sourceQuantParams == nullptr ?
                InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayers.front()) : sourceQuantParams;
        }

        std::set<size_t> concatIdxToUpdate;
        for (int i = 0; i < inputLayers.size(); i++) {
            auto quantParamsIn = InferenceEngine::getInjectedData<QuantizedLayerParams>(inputLayers[i]);

            // Skip updating layer if the new scale factor and old one are the equal
            if (fp32eq(quantParamsIn->_dst_quant.GetScale(), sourceQuantParams->_dst_quant.GetScale())) continue;

            // Scale factor for an input layar has to be updated
            concatIdxToUpdate.insert(i);
        }

        if (concatQuantData->_dst_quant.IsScaleSet() &&
            !fp32eq(concatQuantData->_dst_quant.GetScale(), sourceQuantParams->_dst_quant.GetScale())) {
            gnalog() << "[INFO] Updating scale factor for '" << concatLayer->name << "' layer. Change from "
                << concatQuantData->_dst_quant.GetScale() << " to " << sourceQuantParams->_dst_quant.GetScale() << "\n";
        }

        concatQuantData->_dst_quant = sourceQuantParams->_dst_quant;
        concatQuantData->_dst_quant.SetMinValues({ minInputValue });
        concatQuantData->_dst_quant.SetMaxValues({ maxInputValue });
        concatQuantData->_src_quant = concatQuantData->_dst_quant;

        if (concatIdxToUpdate.empty()) {
            return true;
        }

        for (auto& layerIdToUpdate : concatIdxToUpdate) {
            InferenceEngine::CNNLayerPtr restartedLayer;
            // making a link activation possible without extra layer if first input to concat not a parent / indirect parent of second input
            // using ufs - upper first search
            gnalog() << "[UFS] searching for quantizeable layer prior: '" << concatLayer->name << "', via " << layerIdToUpdate << "\n";

            CNNNetDFS(InferenceEngine::CNNLayerPtr(concatLayer, [](InferenceEngine::CNNLayer*) {}),
                [&restartedLayer, concatLayer](InferenceEngine::CNNLayerPtr layer) {
                    gnalog() << "[UFS] from : '" << concatLayer->name << "' reached: '" << layer->name << "'";
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
                THROW_GNA_EXCEPTION << "cannot requantize '" << inputLayers[layerIdToUpdate]->name << "' input to concat: '"
                    << concatLayer->name << "'";
            }
            auto quantDataForConcatInput = InferenceEngine::getInjectedData<QuantizedLayerParams>(*restartedLayer);
            auto restarLayerInfo = LayerInfo(restartedLayer);
            if (restarLayerInfo.isActivation()) {
                // requantize activation by just changing it's output scale factor
                if (quantDataForConcatInput->_dst_quant.GetScale() < sourceQuantParams->_dst_quant.GetScale()) {
                    gnalog() << "[WARNING] Possible output saturation. '"
                        << restartedLayer->name << "' layer scale factor change from "
                        << quantDataForConcatInput->_dst_quant.GetScale() << " to "
                        << sourceQuantParams->_dst_quant.GetScale() << "\n";
                }

                gnalog() << "[WARNING] Activation layer '" << restartedLayer->name
                    << "' will be requantized. Output scale factor change from "
                    << quantDataForConcatInput->_dst_quant.GetScale() << " to "
                    << sourceQuantParams->_dst_quant.GetScale() << "\n";
                quantDataForConcatInput->_dst_quant.SetScale(sourceQuantParams->_dst_quant.GetScale());
                result = ScaleFactorUpdateResult(restartedLayer.get());
                return true;
            } else if (restarLayerInfo.isMemory()) {
                gnalog() << "[WARNING] Memory layer '" << restartedLayer->name
                    << "' will be requantized. Scale factor change from "
                    << quantDataForConcatInput->_src_quant.GetScale() << " to "
                    << sourceQuantParams->_dst_quant.GetScale() << "\n";
                quantDataForConcatInput->_src_quant = sourceQuantParams->_dst_quant;
                quantDataForConcatInput->_dst_quant = sourceQuantParams->_dst_quant;
                result = ScaleFactorUpdateResult(restartedLayer.get());
                return true;
            } else if (restarLayerInfo.isConst()) {
                gnalog() << "[WARNING] Const layer '" << restartedLayer->name
                    << "' will be requantized. Scale factor change from "
                    << quantDataForConcatInput->_src_quant.GetScale() << " to "
                    << sourceQuantParams->_dst_quant.GetScale() << "\n";
                quantDataForConcatInput->_src_quant = sourceQuantParams->_dst_quant;
                quantDataForConcatInput->_dst_quant = sourceQuantParams->_dst_quant;
                result = ScaleFactorUpdateResult(restartedLayer.get());
                return true;
            } else {
                THROW_GNA_EXCEPTION << "Requantization of unsupported layer '"
                    << restartedLayer->name << "'";
            }
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

    const float MAX_VALUE_GUARDBAND = 1.0f;

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
    * @brief Function to find minimum and maximum values in an array
    * @param A Array to search for min and max values
    * @return Returns a pair of two FP values where minimum is first value and maximum second
    */
    std::pair<float, float> findMinMaxValues(const std::vector<float>& A) {
        auto minValue = std::numeric_limits<float>::max();
        auto maxValue = std::numeric_limits<float>::min();
        for (auto val : A) {
            minValue = std::min(minValue, val);
            maxValue = std::max(maxValue, val);
        }

        return { minValue, maxValue };
    }

    /**
    * @brief Function returns NCHW order of dimensions in given layout
    * @param layout Input layout
    * @return Returns an array of dimension indicies
    */
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

    /**
    * @brief Function transponses provided floating point matrix
    * @param ptr_matrix Input matrix
    * @param num_rows Number of rows in input matrix
    * @param num_cols Number of columns in input matrix
    * @return Returns trasponsed matrix
    */
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

    /**
    * @brief Function simulates affine transformation W * I + B
    * @param weights Weights matrix (W)
    * @param input Input matrix (I)
    * @param biases Biases (B)
    * @param numRowsA Number of rows in weights matrix
    * @param numColA Number of columnts in weights matrix
    * @param numRowsB Number of rows in input matrix
    * @param numColB Number of columnts in input matrix
    * @return Returns result of affine transformation
    */
    auto simulateAffine(const std::vector<float>& weights, const std::vector<float>& input, const std::vector<float>& biases,
        size_t numRowsA, size_t numColA, size_t numRowsB, size_t numColB) {
        if (numColA != numRowsB) {
            THROW_GNA_EXCEPTION << "Invalid number of rows in weights";
        }

        const size_t numRowsC = numRowsA;
        const size_t numColC = numColB;
        std::vector<float> result(numRowsC * numColC, 0.0f);
        for (size_t rowC = 0; rowC < numRowsC; ++rowC) {
            for (size_t colC = 0; colC < numColC; ++colC) {
                auto& sum = result[rowC * numColC + colC];
                sum = biases.empty() ? 0.0f : biases[rowC];
                for (size_t colA = 0; colA < numColA; ++colA) {
                    auto valA = weights[rowC * numColA + colA];
                    auto valB = input[colA * numColB + colC];
                    sum += valA * valB;
                }
            }
        }

        return result;
    }

    /**
    * @brief Function simulates convolutin operation
    * @param src_buffer Input matrix
    * @param kernels_data Kernel matrix
    * @param bias_data Biases
    * @param strideX Stride in X dimension
    * @param strideY Stride in Y dimension
    * @param strideZ Stride in Z dimension
    * @param paddingX Padding in X dimension
    * @param paddingY Padding in Y dimension
    * @param paddingZ Padding in Z dimension
    * @param inputWidth Width of input matrix
    * @param inputHeight Height of input matrix
    * @param inputChannels Depth of input matrix
    * @param outputWidth Width of output matrix
    * @param outputHeight Height of output matrix
    * @param outputChannels Depth of output matrix
    * @param kernelWidth Width of kernel matrix
    * @param kernelHeight Height of kernel matrix
    * @return Returns result of convolutin transformation
    */
    auto simulateConvolution(const std::vector<float>& src_buffer,
        const std::vector<float>& kernels_data,
        const std::vector<float>& bias_data,
        size_t strideX, size_t strideY, size_t strideZ,
        size_t paddingX, size_t paddingY, size_t paddingZ,
        size_t inputWidth, size_t inputHeight, size_t inputChannels,
        size_t outputWidth, size_t outputHeight, size_t outputChannels,
        size_t kernelWidth, size_t kernelHeight) {
        std::vector<float> dst_data(outputWidth * outputHeight * outputChannels, 0.0f);

        if (inputHeight != 1 || outputHeight != 1) {
            THROW_GNA_EXCEPTION << "2D convolutions are not supported yet";
        }

        for (uint32_t oc = 0; oc < outputChannels; ++oc) {
            for (uint32_t oh = 0; oh < outputHeight; ++oh) {
                for (uint32_t ow = 0; ow < outputWidth; ++ow) {
                    size_t oidx = oc * outputHeight * outputWidth + oh * outputWidth + ow;
                    if (!bias_data.empty()) {
                        dst_data[oidx] = bias_data[oc];
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

                                auto inputVal = src_buffer[inputIdx];
                                auto kernelVal = kernels_data[kernelIdx];
                                dst_data[oidx] += inputVal * kernelVal;
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
            THROW_GNA_EXCEPTION << "Incorrect weight value for '" << wl->name << "': " << wl->type << "\n";
        }

        auto prevLayer = CNNNetPrevLayer(wl);
        auto quantDataForInputLayer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(*InferenceEngine::CNNNetPrevLayer(wl).get());

        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*wl);
        if (quant->_src_quant.IsScaleSet() && quant->_src_quant.IsEqual(quantDataForInputLayer->_dst_quant)) {
            //avoid recalculation of scale factors if no change
            return true;
        }

        quant->_src_quant = quantDataForInputLayer->_dst_quant;

        if (quant->_src_quant.GetMinValues().size() != 1 ||
            quant->_src_quant.GetMaxValues().size() != 1) {
            THROW_GNA_EXCEPTION << "Incorrect number of minimum and/or maximum values for '" << wl->name << "' layer. "
                << "Min: " << quant->_src_quant.GetMinValues().size()
                << ", max: " << quant->_src_quant.GetMaxValues().size();
        }

        auto maxAbsInputVal = std::max(std::abs(quant->_src_quant.GetMinValues().front()),
            std::abs(quant->_src_quant.GetMaxValues().front()));

        // TODO: pass 8 bits somehow
        if (!quant->_weights_quant.IsScaleSet()) {
            size_t scaleRange = 0;
            if (weightsSize == 2) {
                scaleRange = MAX_VAL_2B_WEIGHT;
            } else if (weightsSize == 1) {
                scaleRange = MAX_VAL_1B_WEIGHT;
            } else {
                THROW_GNA_EXCEPTION << "Unsupported weights size of: " << weightsSize;
            }

            auto minWeightsVal = std::numeric_limits<float>::max();
            auto maxWeightsVal = std::numeric_limits<float>::min();
            auto weightsBuf = wl->_weights->buffer().as<float*>();
            for (size_t i = 0; i < wl->_weights->size(); i++) {
                minWeightsVal = std::min(minWeightsVal, static_cast<float>(weightsBuf[i]));
                maxWeightsVal = std::max(maxWeightsVal, static_cast<float>(weightsBuf[i]));
            }

            auto maxAbsVal = std::max(std::abs(minWeightsVal), std::abs(maxWeightsVal));
            auto weightsScaleFactor = static_cast<float>(scaleRange) / maxAbsVal;
            if (fp32eq(maxAbsVal, 0.0f)) {
                // non-zero value is required to avoid division by 0 in scale factor calculation
                maxWeightsVal = 1.0f;
                weightsScaleFactor = 1.0f;
            }

            if (fp32eq(minWeightsVal, maxWeightsVal) && fp32eq(minWeightsVal, 1.0f) ||
                LayerInfo(wl).isCropAffined() || LayerInfo(wl).isConcatAlignFilter()) {
                // If all values in weights are equal to -1, 0 and 1 only 1 level is required
                // to describe all possible weight values.
                weightsScaleFactor = 1.0f;
            }

            quant->_weights_quant.SetScale(weightsScaleFactor);
            quant->_weights_quant.SetMinValues({ minWeightsVal });
            quant->_weights_quant.SetMaxValues({ maxWeightsVal });

            if (wl->_biases) {
                auto minBiasVal = std::numeric_limits<float>::max();
                auto maxBiasVal = std::numeric_limits<float>::min();
                auto biasBuf = wl->_biases->buffer().as<float*>();
                for (size_t i = 0; i < wl->_biases->size(); i++) {
                    minBiasVal = std::min(minBiasVal, static_cast<float>(biasBuf[i]));
                    maxBiasVal = std::max(maxBiasVal, static_cast<float>(biasBuf[i]));
                }

                auto maxBiasAbsVal = std::max(std::abs(minBiasVal), std::abs(maxBiasVal));
                if (!fp32eq(maxAbsVal, 0.0f)) {
                    quant->_bias_quant.SetMinValues({ minBiasVal });
                    quant->_bias_quant.SetMaxValues({ maxBiasVal });
                    auto biasScaleFactor = MAX_VAL_4B_BIAS / maxBiasAbsVal;
                    quant->_bias_quant.SetScale(
                        std::min(quant->_weights_quant.GetScale() * quant->_src_quant.GetScale(), biasScaleFactor));
                    quant->_weights_quant.SetScale(quant->_bias_quant.GetScale() / quant->_src_quant.GetScale());
                }
            }

            if (weightsSize == 1) {
                // Per channel scale multiplier is used to extend precision for 8bit weights.
                // However there is only 1 output scale factor and maximum muliplier must be used.
                quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() * MAX_OUT_MULTIPLIER);
            }

            auto inputs = wl->insData.begin()->lock();
            //double maxVal = inputs[1] * quant->_weights_quant.GetScale() * maxAbsVal * quant->_src_quant.GetScale() * maxAbsInputVal;
            //if (maxVal > std::numeric_limits<int32_t>::max()) {
            //}

            double weights_reducer = 1.0;
            auto conv = dynamic_cast<InferenceEngine::ConvolutionLayer *>(wl);
            if (conv) {
                auto in_order = getFromIRDimsOrderNCHW(inputs->getLayout());                
                auto in_width = FROM_IR_DIM(inputs, in_order[3]);
                weights_reducer = MAX_VAL_2B_FEAT * weightsScaleFactor * in_width / std::numeric_limits<int32_t>::max();

                weights_reducer = std::max(1.0, weights_reducer);

                if (weights_reducer > 1.0f) {
                    gnalog() << "[WARNING] Kernel for convolution layer '" << wl->name
                        << "' will be requantized. Weights scale factor change from "
                        << quant->_weights_quant.GetScale() << " to "
                        << quant->_weights_quant.GetScale() / weights_reducer << "\n";
                }
            }
            quant->_weights_quant.SetScale(quant->_weights_quant.GetScale() / weights_reducer);
        }

        double tmp_dst_quant_scale = quant->_weights_quant.GetScale() * quant->_src_quant.GetScale();
        if (weightsSize == 1 &&
            static_cast<uint64_t>(tmp_dst_quant_scale * quant->_src_quant.GetScale()) >
            static_cast<uint64_t>(std::numeric_limits<int32_t>::max() - 1) * _scale_change_req_threshold) {
            gnalog() << "[WARNING] Output scale for '" << wl->name
                << "' too large and are being reduced. Else saturations likely will happen \n";
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

        auto maxAbsWeightsVal = std::max(std::abs(quant->_weights_quant.GetMinValues().front()),
            std::abs(quant->_weights_quant.GetMaxValues().front()));
        auto maxOutputScaledVal = maxAbsInputVal * quant->_src_quant.GetScale() *
            maxAbsWeightsVal * quant->_weights_quant.GetScale();

        auto scaleFactor = quant->_weights_quant.GetScale() * quant->_src_quant.GetScale();
        if (quant->_dst_quant.IsScaleSet() && !fp32eq(quant->_dst_quant.GetScale(), scaleFactor)) {
            gnalog() << "[INFO] Updating scale factor for '" << wl->name << "' layer. Change from "
                << quant->_dst_quant.GetScale() << " to " << scaleFactor << "\n";
        }

        quant->_dst_quant.SetScale(scaleFactor);

        if (1 || quant->_dst_quant.GetMaxValues().empty()) {
            // needs to estimate minimum and maximum values out of weightable layer
            if (wl->_weights->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
                THROW_GNA_EXCEPTION << "Invalid weights precision '" << wl->name << "' layer. Expected: FP32, got: "
                    << wl->_weights->getTensorDesc().getPrecision();
            }

            const int seed = 0;
            std::mt19937 gen(static_cast<float>(seed));
            auto generateFloatNumbers = [gen](std::size_t vec_len, float min, float max) mutable {
                std::vector<float> res;

                std::uniform_real_distribution<float> dist(min, max);
                for (size_t i = 0; i < vec_len; i++)
                    res.emplace_back(static_cast<float>(dist(gen)));

                return res;
            };

            auto minInputVal = quant->_src_quant.GetMinValues().front();
            auto maxInputVal = quant->_src_quant.GetMaxValues().front();

            if (LayerInfo(wl).isFullyConnected()) {
                // calculate minimum and maximum value from layer by simulating GNA affine layer
                auto inputs = wl->insData.begin()->lock();
                auto outputs = *wl->outData.begin();
                auto num_rows_in = FROM_IR_DIM(inputs, 1);
                auto num_columns_in = FROM_IR_DIM(inputs, 2);
                auto num_rows_out = FROM_IR_DIM(outputs, 1);

                // generate random input
                std::vector<float> input = generateFloatNumbers(num_rows_in * num_columns_in, minInputVal, maxInputVal);

                // get weights and convert them to float
                auto weights = std::vector<float>(wl->_weights->size());
                memcpy(&weights[0], wl->_weights->buffer().as<float*>(), weights.size());

                // get biases and convert them to float
                auto biases = std::vector<float>();
                if (wl->_biases) {
                    biases.resize(wl->_biases->size());
                    memcpy(&biases[0], wl->_biases->buffer().as<float*>(), wl->_biases->size());
                }

                // simulate GNA affine layer
                auto result = simulateAffine(weights, input, biases, num_rows_out, num_rows_in, 
                    num_rows_in, num_columns_in);

                // find minimum and maximum values
                auto outputValue = findMinMaxValues(result);
                quant->_dst_quant.SetMinValues({ outputValue.first * MAX_VALUE_GUARDBAND });
                quant->_dst_quant.SetMaxValues({ outputValue.second * MAX_VALUE_GUARDBAND });
            } else if (LayerInfo(wl).isConvolution()) {
                // calculate minimum and maximum value from layer by simulating GNA convolution layer
                auto convLayer = dynamic_cast<InferenceEngine::ConvolutionLayer*>(wl);
                if (!convLayer) {
                    THROW_GNA_EXCEPTION << "Incorrect Convolution Layer pointer  \n";
                }

                auto inputs = wl->insData.begin()->lock();
                auto outputs = *wl->outData.begin();
                if (inputs->getLayout() != InferenceEngine::Layout::NHWC &&
                    inputs->getLayout() != InferenceEngine::Layout::NCHW &&
                    inputs->getLayout() != InferenceEngine::Layout::NC) {
                    THROW_GNA_EXCEPTION << "'" << convLayer->name << "' layer with layout "
                        << inputs->getLayout() << " isn't currently supported on GNA";
                }
                if (inputs->getLayout() != outputs->getLayout()) {
                    THROW_GNA_EXCEPTION << "'" << convLayer->name << "' layer I/O layout mismatch: "
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
                    // If this check is removed, the convolution calculations needs to be corrected.
                    THROW_GNA_EXCEPTION << "'" << convLayer->name << "' layer with batch size not equals 1 is not supported";
                }
                if ((in_channels > 1) && (in_height > 1) && (in_width > 1)) {
                    THROW_GNA_EXCEPTION << "'" << convLayer->name << "' layer with 3D input is not supported on GNA";
                }

                std::vector<float> convertedKernels;
                for (uint32_t k = 0; k < convLayer->_out_depth; k++) {
                    auto kernel = std::vector<float>(convLayer->_kernel_x * in_channels);
                    memcpy(&kernel[0], convLayer->_weights->cbuffer().as<float*>() + k * kernel.size(), kernel.size() * sizeof(float));

                    //kernel = transposeMatrix(&kernel[0], in_channels, convLayer->_kernel_x);
                    convertedKernels.insert(convertedKernels.end(), kernel.begin(), kernel.end());
                }

                std::vector<float> biases;
                if (wl->_biases) {
                    biases.resize(out_channels);
                    memcpy(&biases[0], wl->_biases->buffer().as<float*>(), wl->_biases->size());
                }

                // perform covolution on random data
                std::vector<float> input = generateFloatNumbers(in_width * in_height * in_channels, minInputVal, maxInputVal);
                auto result = simulateConvolution(input, convertedKernels, biases,
                    convLayer->_stride_x, convLayer->_stride_y, 1,
                    convLayer->_padding_x, convLayer->_padding_y, 0,
                    in_width, in_height, in_channels,
                    out_width, out_height, out_channels,
                    convLayer->_kernel_x, 1);

                // find minimum and maximum values
                auto outputValue = findMinMaxValues(result);
                quant->_dst_quant.SetMinValues({ outputValue.first * MAX_VALUE_GUARDBAND });
                quant->_dst_quant.SetMaxValues({ outputValue.second * MAX_VALUE_GUARDBAND });
            } else {
                auto minWeightsVal = quant->_weights_quant.GetMinValues().front();
                auto maxWeightsVal = quant->_weights_quant.GetMaxValues().front();

                std::vector<float> vals;
                vals.push_back(minInputVal * minWeightsVal);
                vals.push_back(minInputVal * maxWeightsVal);
                vals.push_back(maxInputVal * minWeightsVal);
                vals.push_back(maxInputVal * maxWeightsVal);
                if (!quant->_bias_quant.GetMaxValues().empty()) {
                    auto val1 = vals;
                    auto val2 = vals;
                    for (size_t i = 0; i < vals.size(); ++i) {
                        val1[i] += quant->_bias_quant.GetMinValues().front();
                        val2[i] += quant->_bias_quant.GetMaxValues().front();
                    }

                    vals.clear();
                    vals.insert(vals.end(), val1.begin(), val1.end());
                    vals.insert(vals.end(), val2.begin(), val2.end());
                }

                auto outputValue = findMinMaxValues(vals);
                quant->_dst_quant.SetMinValues({ outputValue.first });
                quant->_dst_quant.SetMaxValues({ outputValue.second });
            }
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
