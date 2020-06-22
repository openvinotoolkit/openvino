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

#include <ie_layers.h>
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

 protected :
    static bool fp32eq(float p1, float p2) {
        return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
    }

    float getActivationScale(GNAPluginNS::LayerInfo const&  layer, QuantizedLayerParams const* quantizedParams) {
        // todo: calculate proper scale factor where we need to expand it a bit to be safe to stay in int16 weights
        // set the initial value
        float result = activation_scale_factor;
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
                auto mink = min_range * slope_scale / quantizedParams->_src_quant.scale;
                auto maxk = max_range * slope_scale / quantizedParams->_src_quant.scale;

                if (mink < std::numeric_limits<int16_t>::max()) {
                    auto localMaxK = std::min(static_cast<double>(std::numeric_limits<int16_t>::max()), maxk);
                    if (localMaxK > optimalK) {
                        result = localMaxK / slope_scale *  quantizedParams->_src_quant.scale;
                        optimalK = localMaxK;
                    }
                }
            }
#else
            // GNA scale factor encoding might poor represent target slop scale, we are probing 2 values
            auto s = gna_slope(1.0, quantizedParams->_src_quant.scale, identity_scale_factor);
            auto scale_default = s.slope * s.slope_scale;
            // probing one more quite good approximation for identity
            s = gna_slope(1.0, quantizedParams->_src_quant.scale, identity_scale_factor / 2);
            auto scale_extra = s.slope * s.slope_scale;
            result = fabs(scale_extra) > fabs(scale_default) ?  identity_scale_factor / 2 : identity_scale_factor;

#endif
        } else if (layer.isRelu() &&
                static_cast<uint64_t>(activation_scale_factor * quantizedParams->_src_quant.scale)
                                                            > std::numeric_limits<int32_t>::max()-1) {
            // if activation is one from relu family, we need to apply heuristic to avoid activation output overflow
            result = (activation_scale_factor * 0.5);
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
             if (CNNNetHasPrevLayer(cnnLayer)) {
                auto prevLayer = CNNNetPrevLayer(cnnLayer);
                auto prevInfo = LayerInfo(prevLayer);
                auto inputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(prevLayer);
               // locating corresponding memory layers with same ID
                for (auto && input : CNNNetGetAllInputLayers(cnnLayer)) {
                    LayerInfo ll(input);
                    if (!ll.isMemory() ||
                        !InferenceEngine::details::CaselessEq<std::string>()(input->params["id"], cnnLayer->params["id"])) {
                        continue;
                    }

                    auto quantSibling = InferenceEngine::getInjectedData<QuantizedLayerParams>(input);

                    // after restarting from memory input - quant is fine
                    if (fp32eq(quantSibling->_dst_quant.scale, inputQuant->_dst_quant.scale)) {
                        quant->_src_quant.scale = quant->_dst_quant.scale = inputQuant->_dst_quant.scale;
                        return true;
                    }

                    if (!fp32eq(quantSibling->_dst_quant.scale, 1)) {
                        // means we already restarted propagation input memory layer
                        // need to search for requantiseable layer prior memory output layer
                        InferenceEngine::CNNLayerPtr  restartedLayer;

                        gnalog() << "Memory layer :"<< input->name << " scale factor: " << quantSibling->_dst_quant.scale
                            << " doesn't match its outputs counterpart: " << cnnLayer->name << " scale factor: " << inputQuant->_dst_quant.scale << "\n";
                        gnalog() << "[UFS] searching for quantizeable input layer for: "<< cnnLayer->name << "\n";

                        CNNNetDFS(InferenceEngine::CNNLayerPtr(cnnLayer, [](InferenceEngine::CNNLayer *) {}),
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
                                  }, true, [&restartedLayer, &cnnLayer](InferenceEngine::CNNLayer *from) {
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
                            quantDataForMemoryOutput->_dst_quant.scale = quantSibling->_dst_quant.scale;
                        } else {
                            THROW_GNA_EXCEPTION << "quantization error : input scale factor ( " << inputQuant->_dst_quant.scale <<") "
                                  << " for " << cnnLayer->name << ", that is child of " << prevLayer->name <<" doesnt match : "
                                  << activation_scale_factor;
                        }

                        result = ScaleFactorUpdateResult(restartedLayer.get());
                        return true;
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

        if (cnnLayer->type == "Const") {
            auto blob = cnnLayer->blobs["custom"];
            auto blob_precision = blob->getTensorDesc().getPrecision();

            if (blob_precision != InferenceEngine::Precision::FP32 && blob_precision != InferenceEngine::Precision::FP16) {
                quant->_dst_quant.scale = 1.0f;
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
            auto scale_val = static_cast<float>(std::numeric_limits<int16_t>::max()) / abs_val;

            // TODO: Investigate what should be the scale in such cases (31910)
            if (std::isinf(scale_val)) {
                quant->_dst_quant.scale = quant->_src_quant.scale;
            } else {
                quant->_dst_quant.scale = scale_val;
            }

            return ScaleFactorUpdateResult();
        }

        if (!CNNNetHasPrevLayer(cnnLayer)) {
            quant->_dst_quant.scale = quant->_src_quant.scale;
            return ScaleFactorUpdateResult();
        }

        // by default layer is pass thru its scale factor
        auto inputQuant = InferenceEngine::getInjectedData<QuantizedLayerParams>(CNNNetPrevLayer(cnnLayer));
        if (!inputQuant) {
            THROW_GNA_EXCEPTION << "layer: " << CNNNetPrevLayer(cnnLayer)->name << "not quantized";
        }
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
                quantData->_weights_quant.scale = quantParams1->_dst_quant.scale;
                quantData->_dst_quant.scale     = quantParams0->_dst_quant.scale * quantParams1->_dst_quant.scale;
                break;
            }
            case InferenceEngine::EltwiseLayer::Sub:
            case InferenceEngine::EltwiseLayer::Sum: {
                // detect which input will be used as biases
                if (LayerInfo(in0).has32BOutput()) {
                    std::swap(in0, in1);
                    std::swap(quantParams0, quantParams1);
                }

                // this path might result in significant data loss
                quantData->_bias_quant.scale = quantParams1->_dst_quant.scale / quantParams0->_dst_quant.scale;
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
    bool operator()(InferenceEngine::ConcatLayer* concatLayer, int weightsSize, ScaleFactorUpdateResult &result) {
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

        auto fp32eq = [](float p1, float p2) -> bool {
            return (std::abs(p1 - p2) <= 0.00001f * std::min(std::abs(p1), std::abs(p2)));
        };

        // if both inputs have same quant value - trivial propagation
        if (fp32eq(quantParams0->_dst_quant.scale, quantParams1->_dst_quant.scale)) {
            quantData->_dst_quant.scale = quantParams0->_dst_quant.scale;
            quantData->_src_quant.scale = quantParams0->_dst_quant.scale;
            return true;
        }
        // support only cases when one of input is network input
        if (infoIn0.isInput() && infoIn1.isInput()) {
            THROW_GNA_EXCEPTION << "Two Input layers " << in0->name << "and" << in1->name << " has different scales in concat!!! \n";
        }

        int concatIdxToUpdate = -1;

        if (infoIn0.isInput()) {
            sourceQuantParams = quantParams0;
        } else if (infoIn1.isInput()) {
            concatIdxToUpdate = 0;
            sourceQuantParams = quantParams1;
        }

        // possible case when some of the concat inputs are free to select scale ex: const->concat<-affine
        if (quantParams1->_dst_quant.scale == 1.0) {
            quantParams1->_weights_quant = quantParams0->_dst_quant;
            quantParams1->_dst_quant     = quantParams0->_dst_quant;

            sourceQuantParams = quantParams0;
        }

        if (quantParams0->_dst_quant.scale == 1.0) {
            quantParams0->_weights_quant = quantParams1->_dst_quant;
            quantParams0->_dst_quant     = quantParams1->_dst_quant;
            sourceQuantParams = quantParams1;
        }

        if (!sourceQuantParams) {
            auto in0LayerInfo = LayerInfo(in0);
            auto in1LayerInfo = LayerInfo(in1);
            if (in0LayerInfo.isActivation()) {
                quantParams0->_weights_quant = quantParams1->_dst_quant;
                quantParams0->_dst_quant = quantParams1->_dst_quant;
                sourceQuantParams = quantParams1;
            } else if (in1LayerInfo.isActivation()) {
                quantParams1->_weights_quant = quantParams0->_dst_quant;
                quantParams1->_dst_quant = quantParams0->_dst_quant;
                sourceQuantParams = quantParams0;
            } else {
                THROW_GNA_LAYER_EXCEPTION(concatLayer) << "Concat quantization for " << in0->type << ": " << in0->name
                    << " and " << in1->type << ": " << in1->name
                    << " as inputs needs to be implemented! None of these inputs is an activation.\n";
            }
        }

        if (!fp32eq(quantParams0->_dst_quant.scale, quantParams1->_dst_quant.scale) && concatIdxToUpdate == -1) {
            THROW_GNA_EXCEPTION << "layers entered into concat have different scale factors" << concatLayer->name;
        }

        quantData->_dst_quant.scale = sourceQuantParams->_dst_quant.scale;
        quantData->_src_quant.scale = sourceQuantParams->_dst_quant.scale;

        if (fp32eq(quantParams0->_dst_quant.scale, quantParams1->_dst_quant.scale) || concatIdxToUpdate == -1) {
            return true;
        }

        auto destinationQuantParams = InferenceEngine::getInjectedData<QuantizedLayerParams>(*concatLayer);
        destinationQuantParams->_dst_quant.scale = sourceQuantParams->_dst_quant.scale;

        InferenceEngine::CNNLayerPtr  restartedLayer;
        // making a link activation possible without extra layer if first input to concat not a parent / indirect parent of second input
        // using ufs - upper first search
        gnalog() << "[UFS] searching for quantizeable layer prior: "<< concatLayer->name << ", via " << concatIdxToUpdate << "\n";

        CNNNetDFS(InferenceEngine::CNNLayerPtr(concatLayer, [](InferenceEngine::CNNLayer *) {}),
                  [&restartedLayer, concatLayer](InferenceEngine::CNNLayerPtr layer) {
                      gnalog() << "[UFS] from : " << concatLayer->name << " reached: " << layer->name;
                      // found that direct input to concat is a indirect parent of align filter - so no link required
                      auto info = LayerInfo(layer);
                      if (!info.isWeightable() && !info.isActivation()) {
                          gnalog() << "... skipped\n";
                          return;
                      }
                      restartedLayer = layer;
                      gnalog() << "... OK,  need requantize\n";
                  }, true, [&restartedLayer, &concatLayer, &concatIdxToUpdate](InferenceEngine::CNNLayer *from) {
                    // aborting UFS once found functional layer, and using only specified input of concat
                    return make_upstream_order(restartedLayer == nullptr ? from : nullptr,
                                               from == concatLayer ? concatIdxToUpdate : -1);
                });

        if (restartedLayer == nullptr) {
            THROW_GNA_EXCEPTION << "cannot requantize " << concatIdxToUpdate << "input to concat: " << concatLayer->name;
        }
        auto quantDataForConCatInput = InferenceEngine::getInjectedData<QuantizedLayerParams>(*restartedLayer);

        auto restarLayerInfo = LayerInfo(restartedLayer);
        if (restarLayerInfo.isActivation()) {
            // requantize activation by just changing it's output scale factor
            quantDataForConCatInput->_dst_quant.scale = sourceQuantParams->_dst_quant.scale;
        }

        result = ScaleFactorUpdateResult(restartedLayer.get());

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
        quant->_src_quant.scale = quantDataForInputLayer->_dst_quant.scale;
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
            if (quant->_weights_quant.scale == -1.0f) {
                quant->_weights_quant.scale = 1.0f;
            }

            if (wl->_biases) {
                quant->_bias_quant.scale = ScaleFactorForQuantization(wl->_biases->buffer().as<float *>(),
                                                                      MAX_VAL_4B_BIAS,
                                                                      wl->_biases->size());
                if (quant->_bias_quant.scale != -1.0f) {
                    quant->_bias_quant.scale = std::min(quant->_weights_quant.scale * quant->_src_quant.scale, quant->_bias_quant.scale);
                    quant->_weights_quant.scale = quant->_bias_quant.scale / quant->_src_quant.scale;
                }
            }

            // TODO: findout why ???
            if (weightsSize == 1) {
                quant->_weights_quant.scale *= MAX_OUT_MULTIPLIER;
            }

            double weights_reducer = 1.0;
            auto conv = dynamic_cast<InferenceEngine::ConvolutionLayer*>(wl);
            if (conv) {
                auto dims = conv->insData.front().lock()->getDims();

                weights_reducer = MAX_VAL_2B_FEAT * scaleRange * dims[1] / std::numeric_limits<int32_t>::max();
                weights_reducer = std::max(1.0, weights_reducer);
            }
            quant->_weights_quant.scale /= weights_reducer;
        }


        double tmp_dst_quant_scale = quant->_weights_quant.scale * quantDataForInputLayer->_dst_quant.scale;

        if (weightsSize == 1 &&
            static_cast<uint64_t>(tmp_dst_quant_scale * quant->_src_quant.scale) >
                                                    static_cast<uint64_t>(std::numeric_limits<int32_t>::max()-1) * _scale_change_req_threshold) {
            gnawarn() << "Output scale for " << wl->name
                                            << " too large and are being reduced. Else saturations likely will happen \n";
            // reduce weight scale according experimental heuristic
            if (quant->_dst_quant.scale * quant->_src_quant.scale /
                    static_cast<float>(std::numeric_limits<int32_t>::max()) < _scale_change_threshold_100) {
                quant->_weights_quant.scale *= _scale_reduction_50;
                tmp_dst_quant_scale *= _scale_reduction_50;
            } else if (quant->_dst_quant.scale * quant->_src_quant.scale /
                    static_cast<float>(std::numeric_limits<int32_t>::max()) < _scale_change_threshold_150) {
                quant->_weights_quant.scale *= _scale_reduction_45;
                tmp_dst_quant_scale *= _scale_reduction_45;
            } else if (quant->_dst_quant.scale * quant->_src_quant.scale /
                    static_cast<float>(std::numeric_limits<int32_t>::max()) < _scale_change_threshold_200) {
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
