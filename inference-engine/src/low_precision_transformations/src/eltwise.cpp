// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/eltwise.hpp"

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>

#include <algorithm>
#include <details/caseless.hpp>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ie_util_internal.hpp"
#include "low_precision_transformations/common/ie_lpt_exception.hpp"
#include "low_precision_transformations/network_helper.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

bool EltwiseTransformation::isSupported(const TensorDesc& tensorDesc1, const TensorDesc& tensorDesc2) noexcept {
    if (tensorDesc1.getPrecision() != tensorDesc2.getPrecision()) {
        return false;
    }

    const std::vector<size_t> dims1 = tensorDesc1.getDims();
    const size_t channelsCount1 = dims1.size() == 1ul ? dims1[0] : dims1[1];
    const std::vector<size_t> dims2 = tensorDesc2.getDims();
    const size_t channelsCount2 = dims2.size() == 1ul ? dims2[0] : dims2[1];
    if ((channelsCount1 != channelsCount2) && (channelsCount1 != 1ul) && (channelsCount2 != 1ul)) {
        return false;
    }

    if (((dims1.size() == 2ul) && (channelsCount1 == 1ul)) ||
        ((dims2.size() == 2ul) && (channelsCount2 == 1ul))) {
        return true;
    }

    if ((dims1 == dims2) && (tensorDesc1.getLayout() != tensorDesc2.getLayout())) {
        return false;
    }

    if (dims1 == dims2) {
        return true;
    }

    if ((dims1.size() > 1ul) && (dims2.size() > 1ul)) {
        if (dims1[1] != dims2[1]) {
            return false;
        }

        const size_t dimensionsSize = std::min(dims1.size(), dims2.size());
        for (size_t dimension = 2ul; dimension < dimensionsSize; ++dimension) {
            if ((dims1[dimension] != dims2[dimension]) && (dims1[dimension] != 1ul) && (dims2[dimension] != 1ul)) {
                return false;
            }
        }
    }

    return true;
}

bool EltwiseTransformation::isBroadcasted(const TensorDesc& tensorDesc) noexcept {
    const std::vector<size_t> dims = tensorDesc.getDims();
    const size_t channelIndex = dims.size() == 1 ? 0ul : (dims.size() == 2ul ? 1ul : 2ul);
    for (size_t dimension = channelIndex; dimension < dims.size(); ++dimension) {
        if (dims[dimension] != 1ul) {
            return false;
        }
    }

    return true;
}


bool EltwiseTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if ((!LayerTransformation::canBeTransformed(context, layer)) || isBroadcastByChannels(layer)) {
        return false;
    }

    if (!CaselessEq<std::string>()(layer.type, "Eltwise")) {
        THROW_IE_EXCEPTION << "layer type '" << layer.name << "' is not correct";
    }

    const DataPtr insData0 = layer.insData[0].lock();
    if (insData0 == nullptr) {
        THROW_IE_LPT_EXCEPTION(layer) << "input data 0 is absent";
    }

    const TensorDesc& tensorDesc0 = insData0->getTensorDesc();
    for (size_t i = 1ul; i < layer.insData.size(); ++i) {
        const DataPtr insData = layer.insData[i].lock();
        if (insData == nullptr) {
            THROW_IE_LPT_EXCEPTION(layer) << "input data " << i << " is absent";
        }
        if (!isSupported(tensorDesc0, insData->getTensorDesc())) {
            return false;
        }
    }

    const EltwiseLayer* eltwiseLayer = dynamic_cast<const EltwiseLayer*>(&layer);
    if (eltwiseLayer == nullptr) {
        THROW_IE_EXCEPTION << "unexpected layer type for layer " << layer.name;
    }

    if ((eltwiseLayer->_operation != EltwiseLayer::eOperation::Sum) && (eltwiseLayer->_operation != EltwiseLayer::eOperation::Prod)) {
        return false;
    }

    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
    if ((parents.size() != 2) || (parents[0]->type != "ScaleShift") || (parents[1]->type != "ScaleShift")) {
        return false;
    }

    return true;
}

bool EltwiseTransformation::isBroadcastByChannels(const CNNLayer& layer) const {
    const int fullPathIndex = getNotEmpty(layer);
    if (fullPathIndex == -1) {
        return false;
    }
    const DataPtr fullPathInsData = layer.insData[fullPathIndex].lock();
    if (fullPathInsData == nullptr) {
        THROW_IE_EXCEPTION << "parent ins data is absent";
    }
    const std::vector<size_t> fullDims = fullPathInsData->getTensorDesc().getDims();
    const size_t fullChannelsCount = fullDims.size() == 1ul ? fullDims[0] : fullDims[1];

    const size_t emptyPathIndex = fullPathIndex == 0ul ? 1lu : 0lu;
    const DataPtr emptyPathInsData = layer.insData[emptyPathIndex].lock();
    if (emptyPathInsData == nullptr) {
        THROW_IE_EXCEPTION << "parent ins data is absent";
    }
    const std::vector<size_t> emptyDims = emptyPathInsData->getTensorDesc().getDims();
    const size_t emptyChannelsCount = emptyDims.size() == 1ul ? emptyDims[0] : emptyDims[1];

    return (fullChannelsCount != emptyChannelsCount) && (fullChannelsCount == 1ul);
}

void EltwiseTransformation::transform(TransformationContext& context, CNNLayer& eltwise) const {
    if (!canBeTransformed(context, eltwise)) {
        return;
    }

    const int fullPathIndex = getNotEmpty(eltwise);
    if (fullPathIndex == -1) {
        return;
    }

    const EltwiseLayer* eltwiseLayer = dynamic_cast<const EltwiseLayer*>(&eltwise);
    if (eltwiseLayer == nullptr) {
        THROW_IE_EXCEPTION << "unexpected layer type for layer " << eltwise.name;
    }

    const size_t emptyPathIndex = fullPathIndex == 0 ? 1lu : 0lu;
    std::vector<float> emptyPathDequantizationScales;
    std::vector<float> emptyPathDequantizationShifts;
    const DataPtr emptyPathData = eltwise.insData[emptyPathIndex].lock();
    if (emptyPathData == nullptr) {
        THROW_IE_LPT_EXCEPTION(eltwise) << "data for empty path is absent";
    }
    const CNNLayerPtr emptyPathDequantizationLayer = getCreatorLayer(emptyPathData).lock();
    {
        fillFromDequantizationLayer(*emptyPathDequantizationLayer, emptyPathDequantizationScales, emptyPathDequantizationShifts);

        if ((eltwiseLayer->_operation == EltwiseLayer::eOperation::Prod) && std::any_of(
            emptyPathDequantizationShifts.begin(),
            emptyPathDequantizationShifts.end(),
            [](const float value) { return value != 0.f; })) {
            return;
        }
    }

    {
        const DataPtr fullPathData = eltwise.insData[fullPathIndex].lock();
        if (fullPathData == nullptr) {
            THROW_IE_LPT_EXCEPTION(eltwise) << "data for full path is absent";
        }
        const CNNLayerPtr fullPathDequantizationLayer = getCreatorLayer(fullPathData).lock();
        std::vector<float> fullPathDequantizationScales;
        std::vector<float> fullPathDequantizationShifts;
        fillFromDequantizationLayer(*fullPathDequantizationLayer, fullPathDequantizationScales, fullPathDequantizationShifts);

        if ((emptyPathDequantizationScales.size() != fullPathDequantizationScales.size()) ||
            (emptyPathDequantizationShifts.size() != fullPathDequantizationShifts.size())) {
            return;
        }

        if (eltwiseLayer->_operation == EltwiseLayer::eOperation::Sum) {
            for (size_t i = 0ul; i < emptyPathDequantizationScales.size(); ++i) {
                fullPathDequantizationScales[i] = fullPathDequantizationScales[i] / emptyPathDequantizationScales[i];
                fullPathDequantizationShifts[i] = (fullPathDequantizationShifts[i] + emptyPathDequantizationShifts[i]) / emptyPathDequantizationScales[i];
            }

            CNNNetworkHelper::updateBlobs(*fullPathDequantizationLayer, "weights", fullPathDequantizationScales);
            CNNNetworkHelper::updateBlobs(*fullPathDequantizationLayer, "biases", fullPathDequantizationShifts);
        } else if (eltwiseLayer->_operation == EltwiseLayer::eOperation::Prod) {
            for (size_t i = 0ul; i < emptyPathDequantizationScales.size(); ++i) {
                fullPathDequantizationScales[i] = fullPathDequantizationScales[i] * emptyPathDequantizationScales[i];
                fullPathDequantizationShifts[i] = fullPathDequantizationShifts[i] * emptyPathDequantizationScales[i];
            }

            CNNNetworkHelper::updateBlobs(*fullPathDequantizationLayer, "weights", fullPathDequantizationScales);
            CNNNetworkHelper::updateBlobs(*fullPathDequantizationLayer, "biases", fullPathDequantizationShifts);
        } else {
            THROW_IE_EXCEPTION << "unexpected operation '" << eltwiseLayer->_operation << "'";
        }
    }

    context.quantizedFakeQuantizeNames.erase(emptyPathDequantizationLayer->name);
    CNNNetworkHelper::removeLayer(context.network, emptyPathDequantizationLayer);

    if (eltwiseLayer->_operation == EltwiseLayer::eOperation::Sum) {
        std::vector<float> eltwiseDequantizationScales(emptyPathDequantizationScales.size());
        for (size_t i = 0lu; i < eltwiseDequantizationScales.size(); ++i) {
            eltwiseDequantizationScales[i] = emptyPathDequantizationScales[i];
        }

        const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(eltwise);

        if ((eltwiseDequantizationScales.size() == 1ul) && (eltwiseDequantizationScales.size() != outputChannelsCount)) {
            eltwiseDequantizationScales.resize(outputChannelsCount);
            std::fill(eltwiseDequantizationScales.begin(), eltwiseDequantizationScales.end(), eltwiseDequantizationScales[0]);
        }

        const std::vector<float> eltwiseDequantizationShifts(emptyPathDequantizationShifts.size());
        addDequantizationLayer(context, eltwise, eltwiseDequantizationScales, eltwiseDequantizationShifts);
    } else if (eltwiseLayer->_operation != EltwiseLayer::eOperation::Prod) {
        THROW_IE_EXCEPTION << "unexpected operation '" << eltwiseLayer->_operation << "'";
    }
}

bool isBranchWithTargetType(const CNNLayer& fakeQuantize, const std::string& type) {
    if (!CaselessEq<std::string>()(fakeQuantize.type, "FakeQuantize")) {
        return false;
    }

    if ((fakeQuantize.outData.size() == 1) && (getInputTo(fakeQuantize.outData[0]).size() == 1)) {
        const CNNLayerPtr parentOnActivation = CNNNetworkHelper::getParent(fakeQuantize, 0);
        if ((parentOnActivation != nullptr) && CaselessEq<std::string>()(parentOnActivation->type, type) &&
            (parentOnActivation->outData.size() == 1) && (getInputTo(parentOnActivation->outData[0]).size() == 1)) {
            return true;
        }
    }

    return false;
}

bool isBranchWithTargetType(const CNNLayer& fakeQuantize, const std::vector<std::string> types) {
    if (!CaselessEq<std::string>()(fakeQuantize.type, "FakeQuantize")) {
        return false;
    }

    return std::any_of(types.begin(), types.end(), [&](const std::string& type) { return isBranchWithTargetType(fakeQuantize, type); });
}

int EltwiseTransformation::getNotEmpty(const CNNLayer& eltwise) {
    // TODO: Pooling specific operations are supported only
    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(eltwise, {"Pooling", "ScaleShift"});
    if (parents.size() != 2lu) {
        return -1;
    }

    if ((CaselessEq<std::string>()(parents[0]->type, "FakeQuantize")) && (!CaselessEq<std::string>()(parents[1]->type, "FakeQuantize"))) {
        return 0;
    }

    if ((CaselessEq<std::string>()(parents[1]->type, "FakeQuantize")) && (!CaselessEq<std::string>()(parents[0]->type, "FakeQuantize"))) {
        return 1;
    }

    const std::vector<std::string> targetTypes = { "Convolution", "Gemm", "FullyConnected" };
    const bool allBranchesAreEqual =
        std::all_of(parents.begin(), parents.end(), [&](const CNNLayerPtr& layer) { return isBranchWithTargetType(*layer, targetTypes); }) ||
        std::all_of(parents.begin(), parents.end(), [&](const CNNLayerPtr& layer) { return !isBranchWithTargetType(*layer, targetTypes); });

    for (size_t index = 0ul; index < parents.size(); ++index) {
        const CNNLayerPtr& parent = parents[index];
        if ((allBranchesAreEqual && isBroadcasted(parent->outData[0]->getTensorDesc())) ||
            ((!allBranchesAreEqual) && isBranchWithTargetType(*parent, targetTypes))) {
            return index;
        }
    }

    int fullPathIndex = 0;
    int constBranchID = CNNNetworkHelper::getConstParentBranchID(eltwise);
    if (constBranchID == -1) {
        for (size_t i = 0ul; i < parents.size(); ++i) {
            if (parents[i]->outData.size() != 1) {
                continue;
            }

            if (getInputTo(parents[i]->outData[0]).size() == 1) {
                return i;
            }
        }
    } else {
        fullPathIndex = constBranchID == 0 ? 1 : 0;
    }

    return fullPathIndex;
}
