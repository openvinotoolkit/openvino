// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/squeeze.hpp"
#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <details/caseless.hpp>
#include <string>
#include <vector>


using namespace InferenceEngine;
using namespace InferenceEngine::details;

void SqueezeTransformation::transform(TransformationContext& context, CNNLayer& layer) const {
    if (!canBeTransformed(context, layer)) {
        return;
    }

    if ((layer.insData.size() == 0) || (layer.insData.size() > 2)) {
        THROW_IE_EXCEPTION << "layer inputs '" << layer.insData.size() << "' is not correct";
    }

    if (!CaselessEq<std::string>()(layer.type, "Squeeze")) {
        THROW_IE_EXCEPTION << "layer '" << layer.name << "' is not correct";
    }

    if (layer.insData.size() > 1) {
        CNNLayerPtr constLayer = CNNNetworkHelper::getParent(layer, 1);
        if ((constLayer != nullptr) && (constLayer->type != "Const")) {
            return;
        }

        Blob::Ptr paramsBlob = CNNNetworkHelper::getBlob(constLayer, "custom");
        if (paramsBlob->getTensorDesc().getPrecision() != Precision::I32) {
            THROW_IE_EXCEPTION << "unexpected precision " << paramsBlob->getTensorDesc().getPrecision();
        }

        DataPtr inputData = layer.insData[0].lock();
        if (inputData == nullptr) {
            THROW_IE_EXCEPTION << "input data is absent";
        }

        const std::vector<size_t> inputDims = inputData->getTensorDesc().getDims();
        if (inputDims.size() < paramsBlob->size()) {
            return;
        }

        const signed int* paramsBuffer = paramsBlob->buffer().as<const signed int*>();
        for (size_t index = 0; index < paramsBlob->size(); ++index) {
            if ((paramsBuffer[index] == 0) || (paramsBuffer[index] == 1)) {
                return;
            }
        }
    } else {
        if (layer.outData.size() != 1) {
            THROW_IE_EXCEPTION << "unexpected output count " << layer.outData.size();
        }
        const std::vector<size_t> outputDims = layer.outData[0]->getDims();

        auto it = std::find(outputDims.begin(), outputDims.end(), 1lu);
        if (it != outputDims.end()) {
            return;
        }

        if (layer.insData.size() != 1) {
            THROW_IE_EXCEPTION << "unexpected input count " << layer.insData.size();
        }
        const DataPtr input = layer.insData[0].lock();
        if (input == nullptr) {
            THROW_IE_EXCEPTION << "input is absent";
        }
        const std::vector<size_t> inputDims = input->getDims();
        for (size_t i = 0; (i < 2) && (i < outputDims.size()); ++i) {
            if (inputDims[i] != outputDims[i]) {
                return;
            }
        }
    }

    TransparentBaseTransformation::transform(context, layer);
}
