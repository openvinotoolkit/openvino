// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gna_layer_info.hpp"
#include "gna_plugin_log.hpp"
#include "gna_layer_helpers.hpp"
#include "frontend/weights_converter.hpp"

#include <ie_algorithm.hpp>

namespace GNAPluginNS {
class GNAFakeQuantizeLayer {
    InferenceEngine::CNNLayerPtr fqLayer;
 public :
    GNAFakeQuantizeLayer(InferenceEngine::CNNLayerPtr fqLayer)
        : fqLayer(fqLayer) {
        if (!LayerInfo(fqLayer).isFakeQuantize()) {
            THROW_GNA_LAYER_EXCEPTION(fqLayer) << "cannot parse as fake quantize";
        }
    }

    /**
     * @brief convert FQ layer directly to gna-pwl activation layer
     */
    DnnActivation parseAsActivation() const {
        DnnActivation fqActivation{};

        fqActivation.fqParams.levels = fqLayer->GetParamAsSizeT("levels");
        auto inputShape  = getShapeForRange(fqLayer, 1);
        auto outputShape = getShapeForRange(fqLayer, 3);

        // TODO: check shapes broadcasting to shape of input at 0
        auto inputRangeSize = InferenceEngine::details::product(inputShape.begin(), inputShape.end());
        auto outputRangeSize = InferenceEngine::details::product(outputShape.begin(), outputShape.end());

        fqActivation.fqParams.set = true;

        fqActivation.fqParams.inputPerChannel = inputRangeSize != 1;
        fqActivation.fqParams.input_low   = getParamFromInputAsFloats(fqLayer, 1);
        fqActivation.fqParams.input_high  = getParamFromInputAsFloats(fqLayer, 2);

        fqActivation.fqParams.outputPerChannel = outputRangeSize != 1;
        fqActivation.fqParams.output_low  = getParamFromInputAsFloats(fqLayer, 3);
        fqActivation.fqParams.output_high = getParamFromInputAsFloats(fqLayer, 4);
        fqActivation.type = kActFakeQuantize;

        return fqActivation;
     }
     /**
      * retrieves input blob for FQ layer that connected to const layer
      */
     InferenceEngine::Blob::Ptr getConstInputData() const {
         return LayerUtils::getParamFromInputAsBlob(fqLayer, 0);
     }

     /**
      * fake quantize has 5 input layers, while 4 of them always constant layer, and 1 might be a tensor - connection
      */
    InferenceEngine::CNNLayerPtr getInputLayer() const {
        return getInputLayerAt(fqLayer, 0);
    }

    size_t getLevels() {
        return fqLayer->GetParamAsSizeT("levels");
    }

    std::pair<std::vector<float>, std::vector<float>> getInputRange() {
        return getRange(fqLayer, 1);
    }

    std::pair<std::vector<float>, std::vector<float>> getOutputRange() {
        return getRange(fqLayer, 3);
    }

    operator InferenceEngine::CNNLayerPtr () const {
        return fqLayer;
    }

    InferenceEngine::CNNLayerPtr operator -> () const {
        return fqLayer;
    }
    InferenceEngine::CNNLayerPtr operator * () const {
        return fqLayer;
    }
 protected :

    static std::pair<std::vector<float>, std::vector<float>> getRange(InferenceEngine::CNNLayerPtr input, size_t idx) {
        auto shape     = getShapeForRange(input, idx);
        auto rangeSize = InferenceEngine::details::product(shape.begin(), shape.end());

        auto dataMin = LayerUtils::getParamFromInputAsBlob(input, idx);
        auto dataMax = LayerUtils::getParamFromInputAsBlob(input, idx + 1);
        std::vector<float> minValues(rangeSize), maxValues(rangeSize);
        switch (dataMin->getTensorDesc().getPrecision()) {
        case InferenceEngine::Precision::FP32: {
            memcpy(&minValues[0], dataMin->buffer().as<float*>(), rangeSize * sizeof(float));
            memcpy(&maxValues[0], dataMax->buffer().as<float*>(), rangeSize * sizeof(float));
            break;
        }
        case InferenceEngine::Precision::FP16: {
            auto dataMinFP32 = make_fp32_blob(dataMin);
            memcpy(&minValues[0], dataMinFP32->buffer().as<float*>(), rangeSize * sizeof(float));

            auto dataMaxFP32 = make_fp32_blob(dataMax);
            memcpy(&maxValues[0], dataMaxFP32->buffer().as<float*>(), rangeSize * sizeof(float));
            break;
        }
        default:
            THROW_GNA_LAYER_EXCEPTION(input) << "cannot cast custom blob to type FP32, since it is of type: "
                << dataMin->getTensorDesc().getPrecision();
            break;
        }

        return {minValues, maxValues};
    }

    static float*  getParamFromInputAsFloats(InferenceEngine::CNNLayerPtr input, size_t idx) {
        auto data = LayerUtils::getParamFromInputAsBlob(input, idx);
        if (data->getTensorDesc().getPrecision() != InferenceEngine::Precision::FP32) {
            THROW_GNA_LAYER_EXCEPTION(input) << "cannot cast custom blob to type FP32, since it is of type: "
                << data->getTensorDesc().getPrecision();
        }
        return data->buffer().as<float*>();
    }

    static InferenceEngine::SizeVector  getShapeFromInput(InferenceEngine::CNNLayerPtr input, size_t idx) {
        auto data = LayerUtils::getParamFromInputAsBlob(input, idx);
        return data->getTensorDesc().getDims();
    }

    static InferenceEngine::CNNLayerPtr  getInputLayerAt(InferenceEngine::CNNLayerPtr input, size_t idx) {
        if (input->insData.size() <= idx) {
            THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx << "input";
        }
        auto iLayerData = input->insData[idx].lock();
        if (!iLayerData) {
            THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
                                             << ", input: cannot dereference data weak-pointer";
        }
        auto iLayer = getCreatorLayer(iLayerData).lock();
        if (!iLayer) {
            THROW_GNA_LAYER_EXCEPTION(input) << "cannot get data from " << idx
                                             << ", input: cannot dereference creator layer weak-pointer";
        }
        return iLayer;
    }

    static InferenceEngine::SizeVector getShapeForRange(InferenceEngine::CNNLayerPtr input, size_t idx) {
        auto lowShape  = getShapeFromInput(input, idx);
        auto highShape = getShapeFromInput(input, idx + 1);
        if (lowShape.size() != highShape.size()) {
            THROW_GNA_LAYER_EXCEPTION(input) << "shapes mismatch for " << idx << " and " << idx + 1 << " inputs";
        }
        for (size_t i = 0; i != lowShape.size(); i++) {
            if (lowShape[i] != highShape[i]) {
                THROW_GNA_LAYER_EXCEPTION(input) << "shapes mismatch for " << idx << " and " << idx + 1 << " inputs";
            }
        }
        return lowShape;
     }
};
}  // namespace GNAPluginNS
