// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#pragma once

#include "gna_layer_info.hpp"
#include "gna_plugin_log.hpp"
#include "gna_layer_helpers.hpp"

#include <ie_algorithm.hpp>

namespace GNAPluginNS {
class GNAFakeQuantizeLayer {
    InferenceEngine::CNNLayerPtr fqLayer;
 public :
    GNAFakeQuantizeLayer(InferenceEngine::CNNLayerPtr fqLayer)
        : fqLayer(fqLayer) {}

    /**
     * @brief convert FQ layer directly to gna-pwl activation layer
     */
    DnnActivation parseAsActivation() const {
        DnnActivation fqActivation;
        if (!LayerInfo(fqLayer).isFakeQuantize()) {
            THROW_GNA_LAYER_EXCEPTION(fqLayer) << "cannot parse as fake quantize";
        }
        fqActivation.args.fakeQuantize.levels      = fqLayer->GetParamAsInt("levels");
        auto inputShape  = getShapeForRange(fqLayer, 1);
        auto outputShape = getShapeForRange(fqLayer, 3);

        // TODO: check shapes broadcasting to shape of input at 0
        auto inputRangeSize = InferenceEngine::details::product(inputShape.begin(), inputShape.end());
        auto outputRangeSize = InferenceEngine::details::product(outputShape.begin(), outputShape.end());

        fqActivation.args.fakeQuantize.inputPerChannel = inputRangeSize != 1;
        fqActivation.args.fakeQuantize.input_low   = getParamFromInputAsFloats(fqLayer, 1);
        fqActivation.args.fakeQuantize.input_high  = getParamFromInputAsFloats(fqLayer, 2);

        fqActivation.args.fakeQuantize.outputPerChannel = outputRangeSize != 1;
        fqActivation.args.fakeQuantize.output_low  = getParamFromInputAsFloats(fqLayer, 3);
        fqActivation.args.fakeQuantize.output_high = getParamFromInputAsFloats(fqLayer, 4);
        fqActivation.type = kActFakeQuantize;

        return fqActivation;
     }
     /**
      * retrieves input blob for FQ layer that connected to const layer
      */
     InferenceEngine::Blob::Ptr getConstInputData() const {
         return LayerUtils::getParamFromInputAsBlob(fqLayer, 0);
     }

 protected :

    static float*  getParamFromInputAsFloats(InferenceEngine::CNNLayerPtr input, size_t idx) {
        auto data = LayerUtils::getParamFromInputAsBlob(input, idx);
        return data->buffer().as<float*>();
    }

    static InferenceEngine::SizeVector  getShapeFromInput(InferenceEngine::CNNLayerPtr input, size_t idx) {
        auto data = LayerUtils::getParamFromInputAsBlob(input, idx);
        return data->getTensorDesc().getDims();
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
