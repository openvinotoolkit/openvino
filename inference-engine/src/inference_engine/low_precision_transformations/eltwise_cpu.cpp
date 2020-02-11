// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/eltwise_cpu.hpp"
#include "low_precision_transformations/network_helper.hpp"

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

using namespace InferenceEngine;
using namespace InferenceEngine::details;

size_t getBranchForChild(const CNNLayer& parent, const CNNLayer& child,
                         const std::unordered_set<std::string>& ignoreLayerTypes) {
    const std::vector<CNNLayerPtr> emptyPathFakeQuantizeChildrenRecursively =
        CNNNetworkHelper::getChildrenRecursivelyExceptTypes(parent, ignoreLayerTypes);

    for (size_t i = 0lu; i < emptyPathFakeQuantizeChildrenRecursively.size(); ++i) {
        const CNNLayerPtr tmpChild = emptyPathFakeQuantizeChildrenRecursively[i];
        if (tmpChild->name == child.name) {
            return i;
        }
    }

    THROW_IE_EXCEPTION << "branch where child layer '" << parent.name << "' is placed from parent '" << child.name
                       << "' was not found";
}

bool EltwiseCpuTransformation::canBeTransformed(const TransformationContext& context, const CNNLayer& layer) const {
    if ((!EltwiseTransformation::canBeTransformed(context, layer)) || isIncreasingTensor(layer)) {
        return false;
    }

    return true;
}

bool EltwiseCpuTransformation::isIncreasingTensor(const CNNLayer& layer) const {
    const int fullPathIndex = getNotEmpty(layer);
    if (fullPathIndex == -1) {
        return false;
    }
    const size_t emptyIndex = fullPathIndex == 0 ? 1lu : 0lu;

    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(layer, { "Pooling" });

    const std::vector<size_t> fullDims = parents[fullPathIndex]->outData[0]->getTensorDesc().getDims();
    const size_t fullChannelsCount = fullDims.size() == 1ul ? fullDims[0] : fullDims[1];
    const std::vector<size_t> emptyDims = parents[emptyIndex]->outData[0]->getTensorDesc().getDims();
    const size_t emptyChannelsCount = emptyDims.size() == 1ul ? emptyDims[0] : emptyDims[1];

    return (fullChannelsCount != emptyChannelsCount) && (fullChannelsCount == 1ul);
}

void EltwiseCpuTransformation::transform(TransformationContext& context, CNNLayer& eltwise) const {
    if (!canBeTransformed(context, eltwise)) {
        return;
    }

    const std::string& operation = eltwise.GetParamAsString("operation", "");
    if ((operation != "sum") && (operation != "mul") && (operation != "prod")) {
        return;
    }

    std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(eltwise, {"Pooling"});
    if ((parents.size() != 2) || (parents[0]->type != "FakeQuantize") || (parents[1]->type != "FakeQuantize")) {
        return;
    }

    const int fullPathIndex = getNotEmpty(eltwise);
    if (fullPathIndex == -1) {
        return;
    }

    const size_t emptyPathIndex = fullPathIndex == 0 ? 1lu : 0lu;
    std::vector<float> fullPathFakeQuantizeQuantizationScales;
    std::vector<float> fullPathFakeQuantizeQuantizationShifts;
    std::vector<float> eltwiseDequantizationScales;
    std::vector<float> eltwiseDequantizationShifts;

    const CNNLayer& emptyPathFakeQuantize = *parents[emptyPathIndex];
    const QuantizationDetails emptyPathQuantizationDetails = QuantizationDetails::getDetails(emptyPathFakeQuantize);
    const DataPrecision emptyPathDataPrecision = getDataPrecision(emptyPathFakeQuantize, emptyPathQuantizationDetails, false, false);
    if (emptyPathDataPrecision.precision == Precision::UNSPECIFIED) {
        return;
    }

    {
        fillFromQuantizationDetails(
            emptyPathQuantizationDetails,
            emptyPathDataPrecision,
            eltwiseDequantizationScales,
            eltwiseDequantizationShifts);

        if (((operation == "mul") || (operation == "prod")) &&
            std::any_of(eltwiseDequantizationShifts.begin(), eltwiseDequantizationShifts.end(), [](const float value) { return value != 0.f; })) {
            return;
        }

        fullPathFakeQuantizeQuantizationScales.resize(eltwiseDequantizationScales.size());
        for (size_t i = 0lu; i < eltwiseDequantizationScales.size(); ++i) {
            fullPathFakeQuantizeQuantizationScales[i] = 1.f / eltwiseDequantizationScales[i];
        }

        fullPathFakeQuantizeQuantizationShifts.resize(eltwiseDequantizationShifts.size());
        for (size_t i = 0lu; i < eltwiseDequantizationShifts.size(); ++i) {
            fullPathFakeQuantizeQuantizationShifts[i] = -eltwiseDequantizationShifts[i] / eltwiseDequantizationScales[i];
        }

        const std::vector<CNNLayerPtr> emptyPathFakeQuantizeChildrenRecursively = CNNNetworkHelper::getChildrenRecursivelyExceptTypes(
            emptyPathFakeQuantize,
            {"Pooling"});
        const std::vector<CNNLayerPtr> emptyPathFakeQuantizeChildren = CNNNetworkHelper::getChildren(emptyPathFakeQuantize);
        if (emptyPathFakeQuantizeChildrenRecursively.size() != emptyPathFakeQuantizeChildren.size()) {
            return;
        }

        for (size_t i = 0lu; i < emptyPathFakeQuantizeChildrenRecursively.size(); ++i) {
            const CNNLayerPtr child = emptyPathFakeQuantizeChildrenRecursively[i];
            if (child->name == eltwise.name) {
                continue;
            }

            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(emptyPathFakeQuantize),
                emptyPathFakeQuantizeChildren[i],
                DequantizationDetails(eltwiseDequantizationScales, eltwiseDequantizationShifts));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }

        const size_t branchWithEltwise = getBranchForChild(emptyPathFakeQuantize, eltwise, {"Pooling"});

        CNNNetworkHelper::updateBlobs(emptyPathFakeQuantize, 3, emptyPathDataPrecision.min);
        CNNNetworkHelper::updateBlobs(emptyPathFakeQuantize, 4, emptyPathDataPrecision.max);

        if (updatePrecisions) {
            CNNNetworkHelper::setOutDataPrecision(emptyPathFakeQuantize, branchWithEltwise, eltwise, emptyPathDataPrecision.precision);
        }
        context.quantizedFakeQuantizeNames.insert(emptyPathFakeQuantize.name);
    }

    {
        // TODO: refactor: extract to standalone method: full path
        CNNLayerPtr fullPathFakeQuantize = parents[fullPathIndex];

        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*fullPathFakeQuantize);
        if (children.size() != 1) {
            THROW_IE_EXCEPTION << "Invalid transformation validation for layer '" << eltwise.name << "'";
        }

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            fullPathFakeQuantize,
            children[0],
            DequantizationDetails(fullPathFakeQuantizeQuantizationScales, fullPathFakeQuantizeQuantizationShifts));
        context.dequantizationLayersNames.insert(dequantizationLayer->name);

        context.quantizedFakeQuantizeNames.insert(fullPathFakeQuantize->name);
    }

    if (operation == "sum") {
        const float parentsCount = static_cast<float>(parents.size());
        for (size_t i = 0lu; i < eltwiseDequantizationShifts.size(); ++i) {
            eltwiseDequantizationShifts[i] = parentsCount * eltwiseDequantizationShifts[i];
        }
    } else if ((operation == "mul") || (operation == "prod")) {
        const float parentsCount = static_cast<float>(parents.size());
        for (size_t i = 0lu; i < eltwiseDequantizationScales.size(); ++i) {
            eltwiseDequantizationScales[i] = pow(eltwiseDequantizationScales[i], parentsCount);
        }
    } else {
        THROW_IE_EXCEPTION << "unexpected operation '" << operation << "'";
    }

    const size_t outputChannelsCount = CNNNetworkHelper::getOutputChannelsCount(eltwise);

    if ((eltwiseDequantizationScales.size() == 1ul) && (eltwiseDequantizationScales.size() != outputChannelsCount)) {
        eltwiseDequantizationScales.resize(outputChannelsCount);
        std::fill(eltwiseDequantizationScales.begin(), eltwiseDequantizationScales.end(), eltwiseDequantizationScales[0]);
    }
    if ((eltwiseDequantizationShifts.size() == 1ul) && (eltwiseDequantizationShifts.size() != outputChannelsCount)) {
        eltwiseDequantizationShifts.resize(outputChannelsCount);
        std::fill(eltwiseDequantizationShifts.begin(), eltwiseDequantizationShifts.end(), eltwiseDequantizationShifts[0]);
    }

    const std::vector<CNNLayerPtr> ew_children = CNNNetworkHelper::getChildren(eltwise);
    if (ew_children.size() == 0) {
        const std::string originalName = eltwise.name;
        CNNNetworkHelper::renameLayer(context.network, eltwise.name, eltwise.name + LayerTransformation::lastLayerPrefix);

        CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
            context,
            std::make_shared<CNNLayer>(eltwise),
            nullptr,
            DequantizationDetails(eltwiseDequantizationScales, eltwiseDequantizationShifts, outputChannelsCount),
            originalName);
        context.dequantizationLayersNames.insert(dequantizationLayer->name);
    } else {
        for (const CNNLayerPtr& child : ew_children) {
            CNNLayerPtr dequantizationLayer = CNNNetworkHelper::addScaleShiftBetween(
                context,
                std::make_shared<CNNLayer>(eltwise),
                child,
                DequantizationDetails(eltwiseDequantizationScales, eltwiseDequantizationShifts, outputChannelsCount));
            context.dequantizationLayersNames.insert(dequantizationLayer->name);
        }
    }
}

bool isBranchWithTargetType(const CNNLayer& fakeQuantize, const std::string& type) {
    if (!CaselessEq<std::string>()(fakeQuantize.type, "FakeQuantize")) {
        return false;
    }

    if ((fakeQuantize.outData.size() == 1) && (fakeQuantize.outData[0]->getInputTo().size() == 1)) {
        const CNNLayerPtr parentOnActivation = CNNNetworkHelper::getParent(fakeQuantize, 0);
        if ((parentOnActivation != nullptr) && CaselessEq<std::string>()(parentOnActivation->type, type)) {
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

int EltwiseCpuTransformation::getNotEmpty(const CNNLayer& eltwise) {
    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(eltwise, {"Pooling"});
    if ((parents.size() != 2lu) || !CaselessEq<std::string>()(parents[0]->type, "FakeQuantize") ||
        !CaselessEq<std::string>()(parents[1]->type, "FakeQuantize")) {
        return -1;
    }

    const std::vector<std::string> targetTypes = { "Convolution", "GEMM", "FullyConnected" };
    const bool allBranchesAreEqual =
        std::all_of(parents.begin(), parents.end(), [&](CNNLayerPtr layer) { return isBranchWithTargetType(*layer, targetTypes); }) ||
        std::all_of(parents.begin(), parents.end(), [&](CNNLayerPtr layer) { return !isBranchWithTargetType(*layer, targetTypes); });

    for (size_t index = 0lu; index < parents.size(); ++index) {
        const CNNLayerPtr parent = parents[index];
        if ((allBranchesAreEqual && isBroadcasted(parent->outData[0]->getTensorDesc())) ||
            ((!allBranchesAreEqual) && isBranchWithTargetType(*parent, targetTypes))) {
            return index;
        }
    }

    int fullPathIndex = 0;
    int constBranchID = CNNNetworkHelper::getConstParentBranchID(eltwise);
    if (constBranchID == -1) {
        fullPathIndex = CNNNetworkHelper::getFakeQuantizeBranchWithOneChild(eltwise);
        if (fullPathIndex == -1)
            return 0;
    } else {
        fullPathIndex = constBranchID == 0 ? 1lu : 0lu;
    }

    return fullPathIndex;
}
