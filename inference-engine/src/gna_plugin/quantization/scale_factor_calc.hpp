// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include <algorithm>
#include <utility>
#include <limits>
#include <string>
#include <map>
#include "gna_layer_info.hpp"
#include "ie_layers.h"
#include "gna_plugin_log.hpp"

namespace GNAPluginNS {
namespace details {
using namespace InferenceEngine;
struct ScaleFactorUpdateResult {
    CNNLayer *restartLayer = nullptr;
    ScaleFactorUpdateResult() = default;
    explicit ScaleFactorUpdateResult(CNNLayer * restartlayer) : restartLayer(restartlayer) {
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
     * @param inputScaleFactor
     * @param result
     * @return
     */
    bool operator()(T cnnLayer, int weightsSize, float inputScaleFactor, ScaleFactorUpdateResult &result) {
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

 protected :
    static bool fp32eq(float p1, float p2) {
        return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
    }
    float getActivationScale(GNAPluginNS::LayerInfo const&  layer, QuantizedLayerParams const* qunatizedParams) {
            // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
            // set the initial value
            float result = 1.0f;
            result = (layer.isIdentity()) ? identity_scale_factor : activation_scale_factor;
            // if activation is one from relu family, we need to apply heuruistic to avoid activation output overflow
            if (layer.isRelu() &&
                    static_cast<uint64_t>(result * qunatizedParams->_src_quant.scale)
                                                                > std::numeric_limits<int32_t>::max()-1) {
                result = (result * 0.5);
            }
            return result;
    }

 public :
    bool operator()(InferenceEngine::CNNLayer *cnnLayer, int weightsSize, float inputScaleFactor, ScaleFactorUpdateResult &result) {
        if ( !cnnLayer ) {
            THROW_IE_EXCEPTION << "Incorrect Convolutional Layer pointer \n";
        }
        LayerInfo layerInfo(*cnnLayer);
        // TODO: current approach set input scale factor for true input layer(s) equals to provided factor,
        auto quant = getInjectedData<QuantizedLayerParams>(*cnnLayer);
        if (InferenceEngine::details::CaselessEq<std::string>()(cnnLayer->type, "Memory")) {
             if (CNNNetHasPrevLayer(cnnLayer)) {
                auto prevLayer = CNNNetPrevLayer(cnnLayer);
                auto prevInfo = LayerInfo(prevLayer);
                auto inputQuant = getInjectedData<QuantizedLayerParams>(prevLayer);
               // locating corresponding memory layers ith same ID
                for (auto && input : CNNNetGetAllInputLayers(cnnLayer)) {
                    LayerInfo ll(input);
                    if (!ll.isMemory() ||
                        !InferenceEngine::details::CaselessEq<std::string>()(input->params["id"], cnnLayer->params["id"])) {
                        continue;
                    }

                    auto quantSibling = getInjectedData<QuantizedLayerParams>(input);

                    // after restarting from memory input - quant is fine
                    if (fp32eq(quantSibling->_dst_quant.scale, inputQuant->_dst_quant.scale)) {
                        quant->_src_quant.scale = quant->_dst_quant.scale = inputQuant->_dst_quant.scale;
                        return true;
                    }

                    if (!fp32eq(quantSibling->_dst_quant.scale, 1)) {
                        // means we already restarted propagation from that memory layer - we cannot do mach here
                        THROW_GNA_EXCEPTION << "quantization error : input scale factor ( " << inputQuant->_dst_quant.scale <<") "
                                  << " for " << cnnLayer->name << ", that is child of " << prevLayer->name <<" doesnt match : "
                                  << activation_scale_factor;
                    }

                    gnawarn() << "[INFO] quantization : input scale factor (" << inputQuant->_dst_quant.scale <<")"
                              << " for " << cnnLayer->name << ", that is child of " << prevLayer->name <<" doesnt match : "
                              << activation_scale_factor << ", restarting from corresponding memory: "<< input->name << std::endl;

                    // try updating memory input layer scale factor and restart from it
                    quantSibling->_src_quant.scale = quantSibling->_dst_quant.scale = inputQuant->_dst_quant.scale;
                    result = ScaleFactorUpdateResult(input.get());
                    return true;
                }
            }
            return true;
        }

        if (!CNNNetHasPrevLayer(cnnLayer)) {
            quant->_dst_quant.scale = inputScaleFactor;
            return ScaleFactorUpdateResult();
        }

        // by default layer is pass thru its scale factor
        auto inputQuant = getInjectedData<QuantizedLayerParams>(CNNNetPrevLayer(cnnLayer));
        quant->_dst_quant.scale = inputQuant->_dst_quant.scale;
        quant->_src_quant.scale = inputQuant->_dst_quant.scale;

        if (layerInfo.isActivation()) {
            // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
            // set the initial value
            quant->_dst_quant.scale = getActivationScale(layerInfo, quant);
        }
        return true;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::EltwiseLayer*> {
 public:
    bool operator()(InferenceEngine::EltwiseLayer* eltwiseLayer, int weightsSize, float inputScaleFactor, ScaleFactorUpdateResult &result) {
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
                quantData->_weights_quant.scale = quantParams1->_dst_quant.scale;
                quantData->_dst_quant.scale     = quantParams0->_dst_quant.scale * quantParams1->_dst_quant.scale;
                break;
            }
            case InferenceEngine::EltwiseLayer::Sum: {
                // detect which input will be used as biases
                if (LayerInfo(in0).has32BOutput()) {
                    std::swap(in0, in1);
                    std::swap(quantParams0, quantParams1);
                }

                // this path might result in significant data loss
                quantData->_weights_quant.scale = quantParams1->_dst_quant.scale / quantParams0->_dst_quant.scale;
                quantData->_dst_quant.scale = quantParams1->_dst_quant.scale;

                // eltwise will always work in int16
                auto maxValue = std::numeric_limits<int16_t>::max() - 1;
                if (quantData->_weights_quant.scale > maxValue + 1) {
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
                                auto newOutputScale = quantParams->_dst_quant.scale / maxValue;
                                if (newOutputScale > static_cast<float>(std::numeric_limits<int16_t>::max()) / 2) {
                                    break;
                                }
                                auto quantDataForActivation = InferenceEngine::getInjectedData<QuantizedLayerParams>(*in);
                                gnawarn() << "[WARNING] saturated weights for " << eltwiseLayer->name
                                         << ". Layer new output scale: " << in->name << ", output_scale=" << newOutputScale
                                         << ", was " << quantDataForActivation->_dst_quant.scale <<"\n" << std::flush;
                                quantDataForActivation->_dst_quant.scale = newOutputScale;
                                result = ScaleFactorUpdateResult(in.get());
                                return true;
                            } else if (info.has16BOutput()) {
                                break;
                            }

                            // if we are here it means that we are in the port 1
                            if (info.isFullyConnected() || info.isConvolution()) {
                                auto quantDataForInputLayer = InferenceEngine::getInjectedData<QuantizedLayerParams>(*in);
                                auto newOutputScale = quantParams->_dst_quant.scale * maxValue;
                                auto newWeightScale = newOutputScale / quantDataForInputLayer->_src_quant.scale;
                                quantDataForInputLayer->_dst_quant.scale = newOutputScale;
                                quantDataForInputLayer->_weights_quant.scale = newWeightScale;
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
    bool operator()(InferenceEngine::ConcatLayer* concatLayer, int weightsSize, float inputScaleFactor, ScaleFactorUpdateResult &result) {
        if ( !concatLayer ) {
            THROW_GNA_EXCEPTION << "Incorrect Concat Layer pointer \n";
        }
        auto in0 = InferenceEngine::CNNNetPrevLayer(concatLayer, 0);
        auto in1 = InferenceEngine::CNNNetPrevLayer(concatLayer, 1);
        auto infoIn0 = LayerInfo(in0);
        auto infoIn1 = LayerInfo(in1);
        auto quantParams0 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in0);
        auto quantParams1 = InferenceEngine::getInjectedData<QuantizedLayerParams>(in1);
        GNAPluginNS::QuantizedLayerParams* sourceQuantParams = NULL;
        auto quantData = InferenceEngine::getInjectedData<QuantizedLayerParams>(*concatLayer);

        if (quantParams0->_dst_quant.scale == quantParams1->_dst_quant.scale) {
            return true;
        } else if (infoIn0.isInput() && infoIn1.isInput()) {
            THROW_GNA_EXCEPTION << "Two Input layers has different scales in concat!!! \n";
        }

        int i = 0;
        if (infoIn0.isInput()) {
            sourceQuantParams = quantParams0;
        } else if (infoIn1.isInput()) {
            ++i;
            sourceQuantParams = quantParams1;
        }

        if (!sourceQuantParams) {
            THROW_GNA_EXCEPTION << "Concat quantization for this case need to be implemented!!! \n";
        }
        auto destinationQuantParams =
                InferenceEngine::getInjectedData<QuantizedLayerParams>(InferenceEngine::CNNNetPrevLayer(concatLayer, !i));
        InferenceEngine::CNNLayerPtr in = InferenceEngine::CNNNetPrevLayer(concatLayer, !i);

        quantData->_dst_quant.scale = sourceQuantParams->_dst_quant.scale;
        quantData->_src_quant.scale = sourceQuantParams->_dst_quant.scale;

        destinationQuantParams->_dst_quant.scale = sourceQuantParams->_dst_quant.scale;
        result = ScaleFactorUpdateResult(in.get());

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
    bool operator()(InferenceEngine::WeightableLayer *wl, int weightsSize, float inputScaleFactor, ScaleFactorUpdateResult &result) {
        if ( !wl ) {
            THROW_GNA_EXCEPTION << "Incorrect Weightable Layer pointer  \n";
        } else if (!wl->_weights) {
            THROW_GNA_EXCEPTION << "Incorrect weight value for " << wl->name << ":" << wl->type << "\n";
        }

        auto prevLayer = CNNNetPrevLayer(wl);
        auto quantDataForInputLayer =
            InferenceEngine::getInjectedData<QuantizedLayerParams>(*InferenceEngine::CNNNetPrevLayer(wl).get());

        auto quant = InferenceEngine::getInjectedData<QuantizedLayerParams>(*wl);
        // TODO: pass 8 bits somehow
        if (quant->_weights_quant.scale == 1.0f) {
            size_t scaleRange = 0;
            if (weightsSize == 2) {
                scaleRange = MAX_VAL_2B_WEIGHT;
            } else if (weightsSize == 1) {
                scaleRange = MAX_VAL_1B_WEIGHT;
            } else {
                THROW_GNA_EXCEPTION << "Unsupported weights size of: " << weightsSize;
            }
            quant->_weights_quant.scale =
                ScaleFactorForQuantization(wl->_weights->buffer().as<float *>(), scaleRange, wl->_weights->size());

            // TODO: findout why ???
            if (weightsSize == 1) {
                quant->_weights_quant.scale *= MAX_OUT_MULTIPLIER;
            }
        }

        quant->_src_quant.scale = quantDataForInputLayer->_dst_quant.scale;

        double tmp_dst_quant_scale = quant->_weights_quant.scale * quantDataForInputLayer->_dst_quant.scale;

        if (weightsSize == 1 &&
            static_cast<uint64_t>(tmp_dst_quant_scale * quant->_src_quant.scale) >
                                                    static_cast<uint64_t>(std::numeric_limits<int32_t>::max()-1) * _scale_change_req_threshold) {
            gnawarn() << "Output scale for " << wl->name
                                            << " too large and are being reduced. Else saturations likely will happen \n";
            // reduce weight scale according experimentatl heuruistic
            if (quant->_dst_quant.scale * quant->_src_quant.scale / std::numeric_limits<int32_t>::max() < _scale_change_threshold_100) {
                quant->_weights_quant.scale *= _scale_reduction_50;
                tmp_dst_quant_scale *= _scale_reduction_50;
            } else if (quant->_dst_quant.scale * quant->_src_quant.scale / std::numeric_limits<int32_t>::max() < _scale_change_threshold_150) {
                quant->_weights_quant.scale *= _scale_reduction_45;
                tmp_dst_quant_scale *= _scale_reduction_45;
            } else if (quant->_dst_quant.scale * quant->_src_quant.scale / std::numeric_limits<int32_t>::max() < _scale_change_threshold_200) {
                quant->_weights_quant.scale *= _scale_reduction_40;
                tmp_dst_quant_scale *= _scale_reduction_40;
            } else {
                quant->_weights_quant.scale *= _scale_reduction_35;
                tmp_dst_quant_scale *= _scale_reduction_35;
            }
        }

        quant->_dst_quant.scale = tmp_dst_quant_scale;

        return true;
    }
};

template<>
class ScaleFactorPerLayer<InferenceEngine::ScaleShiftLayer*> : public ScaleFactorPerLayer<InferenceEngine::WeightableLayer*> {
 public:
    bool operator()(InferenceEngine::WeightableLayer *wl, int weightsSize, float inputScaleFactor, ScaleFactorUpdateResult &result) {
        return ScaleFactorPerLayer<InferenceEngine::WeightableLayer*>::operator()(wl, 2, inputScaleFactor, result);
    }
};

/**
 * GNA convolutions cannot be quantized in int8, remove when library starts support that
 */
template<>
class ScaleFactorPerLayer<InferenceEngine::ConvolutionLayer*> : public ScaleFactorPerLayer<InferenceEngine::ScaleShiftLayer*> {
};


}  // namespace details

/**
 * @brief scale factor calculator will calculate only output scale factors for the layer
 * if scale factor propagation not possible, it will fall indicate a restart condition
 */
class ScaleFactorCalculator {
    using Cnt = std::vector<InferenceEngine::CNNLayerPtr>;
    Cnt  net;
    mutable Cnt::const_iterator idx;
    float inputScaleFactor;
    mutable bool needRestart = false;
    int weightsBytesSize;

 public:
    ScaleFactorCalculator(Cnt &net, int weightsBytesSize, float inputScaleFactor)
            : net(net), inputScaleFactor(inputScaleFactor), weightsBytesSize(weightsBytesSize) {
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
        details::ScaleFactorUpdateResult result;
        if (!details::ScaleFactorPerLayer<T>()(ptr, weightsBytesSize, inputScaleFactor, result)) {
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
