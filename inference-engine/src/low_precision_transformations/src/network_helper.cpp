// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/network_helper.hpp"

#include <algorithm>
#include <blob_factory.hpp>
#include <cmath>
#include <details/caseless.hpp>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include <details/ie_cnn_network_tools.h>
#include <ie_common.h>
#include <precision_utils.h>
#include "cnn_network_impl.hpp"
#include "ie_util_internal.hpp"
#include "ie_parallel.hpp"
#include "low_precision_transformations/common/ie_lpt_exception.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

static const std::unordered_set<std::string> intermediateLayers{
    "Pooling",
    "Resample"
};

bool Subgraph::fillSubgraphForQuantization(const CNNLayerPtr& fakeQuantize, std::unordered_set<std::string>& handledLayers) {
    if (fakeQuantize->type != "FakeQuantize") {
        THROW_IE_EXCEPTION << "unexpected layer type " << fakeQuantize->type;
    }

    if (!QuantizationDetails::outputLayoutIsSupported(*fakeQuantize)) {
        return false;
    }

    quantizationLayers.push_back(fakeQuantize);
    handledLayers.insert(fakeQuantize->name);
    layers.emplace(fakeQuantize->name, fakeQuantize.get());

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*fakeQuantize);
    for (const CNNLayerPtr& child : children) {
        if (handledLayers.find(child->name) != handledLayers.end()) {
            continue;
        }

        if (child->type == "Concat") {
            if (!fillSubgraphForConcat(child, handledLayers)) {
                return false;
            }
        } else if (child->type == "FakeQuantize") {
            //
        } else if (intermediateLayers.find(child->type) != intermediateLayers.end()) {
            if (!fillSubgraphForIntermediate(child, handledLayers)) {
                return false;
            }
        }
    }

    return true;
}

bool Subgraph::fill(const CNNLayerPtr& layer, std::unordered_set<std::string>& handledLayers) {
    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*layer);
    for (const CNNLayerPtr& parent : parents) {
        if (handledLayers.find(parent->name) != handledLayers.end()) {
            continue;
        }

        if (parent->type == "Concat") {
            if (!fillSubgraphForConcat(parent, handledLayers)) {
                return false;
            }
        } else if (parent->type == "FakeQuantize") {
            if (!fillSubgraphForQuantization(parent, handledLayers)) {
                return false;
            }
        } else if (intermediateLayers.find(parent->type) != intermediateLayers.end()) {
            if (!fillSubgraphForIntermediate(parent, handledLayers)) {
                return false;
            }
        } else {
            return false;
        }
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*layer);
    for (const CNNLayerPtr& child : children) {
        if (handledLayers.find(child->name) != handledLayers.end()) {
            continue;
        }

        if (child->type == "Concat") {
            if (!fillSubgraphForConcat(child, handledLayers)) {
                return false;
            }
        } else if (child->type == "FakeQuantize") {
            //
        } else if (intermediateLayers.find(child->type) != intermediateLayers.end()) {
            if (!fillSubgraphForIntermediate(child, handledLayers)) {
                return false;
            }
        }
    }

    return true;
}

bool Subgraph::fillSubgraphForIntermediate(const CNNLayerPtr& intermediate, std::unordered_set<std::string>& handledLayers) {
    if (intermediateLayers.find(intermediate->type) == intermediateLayers.end()) {
        THROW_IE_EXCEPTION << "unexpected layer type " << intermediate->type;
    }

    handledLayers.insert(intermediate->name);
    layers.emplace(intermediate->name, intermediate.get());

    return fill(intermediate, handledLayers);
}

bool Subgraph::empty() const {
    return quantizationLayers.empty();
}

bool Subgraph::fillSubgraphForConcat(const CNNLayerPtr& concat, std::unordered_set<std::string>& handledLayers) {
    if (concat->type != "Concat") {
        THROW_IE_EXCEPTION << "unexpected layer type " << concat->type;
    }

    concatLayers.push_back(concat);
    handledLayers.insert(concat->name);
    layers.emplace(concat->name, concat.get());

    return fill(concat, handledLayers);
}

Subgraph CNNNetworkHelper::getSubgraph(const CNNLayer& concat) {
    if (concat.type != "Concat") {
        THROW_IE_EXCEPTION << "unexpected layer type " << concat.type;
    }

    Subgraph subgraph;
    std::unordered_set<std::string> handledLayers;
    if (!subgraph.fillSubgraphForConcat(std::make_shared<CNNLayer>(concat), handledLayers)) {
        return Subgraph();
    }

    return subgraph;
}

CNNLayerPtr CNNNetworkHelper::getLayer(const ICNNNetwork& network, const std::string& layerName) {
    std::vector<CNNLayerPtr> layers = InferenceEngine::details::CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : layers) {
        if (layer->name == layerName) {
            return layer;
        }
    }

    return nullptr;
}

Blob::Ptr CNNNetworkHelper::makeNewBlobPtr(const TensorDesc& desc) {
    Blob::Ptr newBlob;
    if (desc.getPrecision() == Precision::FP32)
        newBlob = make_shared_blob<PrecisionTrait<Precision::FP32>::value_type>(desc);
    else if (desc.getPrecision() == Precision::FP16)
        newBlob = make_shared_blob<PrecisionTrait<Precision::FP16>::value_type>(desc);
    else if (desc.getPrecision() == Precision::I8)
        newBlob = make_shared_blob<PrecisionTrait<Precision::I8>::value_type>(desc);
    else if (desc.getPrecision() == Precision::U8)
        newBlob = make_shared_blob<PrecisionTrait<Precision::U8>::value_type>(desc);
    else if (desc.getPrecision() == Precision::I32)
        newBlob = make_shared_blob<PrecisionTrait<Precision::I32>::value_type>(desc);
    else
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << desc.getPrecision();

    return newBlob;
}

void CNNNetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, float value) {
    const auto existingBlobIt = layer.blobs.find(blobName);
    if (existingBlobIt == layer.blobs.end()) {
        THROW_IE_EXCEPTION << "blob '" << blobName << "' was not found in layer " << layer.name;
    }
    const auto& existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    Blob::Ptr newBlob = makeNewBlobPtr(existingBlobTensorDesc);

    newBlob->allocate();
    fillBlobByFP32(newBlob, value);
    layer.blobs[existingBlobIt->first] = newBlob;
}

void CNNNetworkHelper::invertFakeQuantize(const CNNLayer& fakeQuantize) {
    if (fakeQuantize.type != "FakeQuantize") {
        THROW_IE_EXCEPTION << "invalid layer type " << fakeQuantize.type;
    }
    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(fakeQuantize);
    const size_t valuesCount =
        std::max(quantizationDetails.inputLowValues.size(), quantizationDetails.outputLowValues.size());
    std::vector<float> inputLowValues(valuesCount);
    std::vector<float> inputHightValues(valuesCount);
    std::vector<float> outputLowValues(valuesCount);
    std::vector<float> outputHighValues(valuesCount);
    bool wasInverted = false;
    for (size_t i = 0ul; i < valuesCount; ++i) {
        if ((quantizationDetails.getInputLowValue(i) > quantizationDetails.getInputHighValue(i)) &&
            (quantizationDetails.getOutputLowValue(i) > quantizationDetails.getOutputHighValue(i))) {
            inputLowValues[i] = quantizationDetails.getInputHighValue(i);
            inputHightValues[i] = quantizationDetails.getInputLowValue(i);
            outputLowValues[i] = quantizationDetails.getOutputHighValue(i);
            outputHighValues[i] = quantizationDetails.getOutputLowValue(i);
            wasInverted = true;
        } else {
            inputLowValues[i] = quantizationDetails.getInputLowValue(i);
            inputHightValues[i] = quantizationDetails.getInputHighValue(i);
            outputLowValues[i] = quantizationDetails.getOutputLowValue(i);
            outputHighValues[i] = quantizationDetails.getOutputHighValue(i);
        }
    }

    if (wasInverted) {
        CNNNetworkHelper::updateBlobs(fakeQuantize, 1, inputLowValues);
        CNNNetworkHelper::updateBlobs(fakeQuantize, 2, inputHightValues);
        CNNNetworkHelper::updateBlobs(fakeQuantize, 3, outputLowValues);
        CNNNetworkHelper::updateBlobs(fakeQuantize, 4, outputHighValues);
    }
}
void CNNNetworkHelper::updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex,
                                   const std::vector<float>& values) {
    CNNLayerPtr blobLayer = CNNNetworkHelper::getParent(quantizeLayer, constLayerIndex);
    if (blobLayer == nullptr) {
        THROW_IE_EXCEPTION << "layer is absent";
    }

    const auto existingBlobIt = blobLayer->blobs.find("custom");
    if (existingBlobIt == blobLayer->blobs.end()) {
        THROW_IE_EXCEPTION << "custom blob was not found ";
    }

    TensorDesc newBlobTensorDesc;

    const TensorDesc existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {
        if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {
            //
        } else if (existingBlobTensorDesc.getLayout() == Layout::C) {
            if (existingBlobTensorDesc.getDims().size() != 1) {
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {
            if (existingBlobTensorDesc.getDims().size() != 4) {
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            // OIHW
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        }

        const std::vector<size_t> dims = {values.size()};
        const Layout layout = Layout::C;
        newBlobTensorDesc = TensorDesc(existingBlobTensorDesc.getPrecision(), dims, layout);
        for (DataPtr data : blobLayer->outData) {
            data->reshape(dims, layout);
        }
    } else {
        newBlobTensorDesc = existingBlobTensorDesc;
    }

    Blob::Ptr newBlob = makeNewBlobPtr(newBlobTensorDesc);
    newBlob->allocate();
    blobLayer->blobs[existingBlobIt->first] = newBlob;

    if (values.size() == 1)
        fillBlobByFP32(newBlob, values[0]);
    else
        fillBlobByFP32(newBlob, values.data());
}

void CNNNetworkHelper::updateBlobs(CNNLayer& layer, const std::string& blobName, const std::vector<float>& values) {
    const auto existingBlobIt = layer.blobs.find(blobName);
    if (existingBlobIt == layer.blobs.end()) {
        THROW_IE_EXCEPTION << "custom blob was not found ";
    }

    TensorDesc newBlobTensorDesc;

    const TensorDesc existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    if ((existingBlobIt->second->size() != values.size()) && (values.size() != 1)) {
        if (existingBlobTensorDesc.getLayout() == Layout::SCALAR) {
            //
        } else if (existingBlobTensorDesc.getLayout() == Layout::C) {
            if (existingBlobTensorDesc.getDims().size() != 1) {
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        } else if (existingBlobTensorDesc.getLayout() == Layout::NCHW) {
            if (existingBlobTensorDesc.getDims().size() != 4) {
                THROW_IE_EXCEPTION << "temporary dimensions size " << existingBlobTensorDesc.getDims().size()
                                   << " for layout " << existingBlobTensorDesc.getLayout() << " is not supported";
            }
            // OIHW
            if (existingBlobTensorDesc.getDims()[0] != 1) {
                THROW_IE_EXCEPTION << "temporary is not supported";
            }
        }

        const std::vector<size_t> dims = {values.size()};
        const Layout layout = Layout::C;
        newBlobTensorDesc = TensorDesc(existingBlobTensorDesc.getPrecision(), dims, layout);
        for (DataPtr data : layer.outData) {
            data->reshape(dims, layout);
        }
    } else {
        newBlobTensorDesc = existingBlobTensorDesc;
    }

    Blob::Ptr newBlob = makeNewBlobPtr(newBlobTensorDesc);
    newBlob->allocate();
    layer.blobs[existingBlobIt->first] = newBlob;

    if ((blobName == "weights") || (blobName == "biases")) {
        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(&layer);
        if (weightableLayer == nullptr) {
            THROW_IE_EXCEPTION << "layer '" << layer.name << "' with blob name '" << blobName << "' is not weightable";
        }
        if (blobName == "weights") {
            weightableLayer->_weights = newBlob;
        } else if (blobName == "biases") {
            weightableLayer->_biases = newBlob;
        } else {
            THROW_IE_EXCEPTION << "unexpected blob name '" << blobName << "' for layer " << layer.name;
        }
    }

    if (values.size() == 1)
        fillBlobByFP32(newBlob, values[0]);
    else
        fillBlobByFP32(newBlob, values.data());
}

void CNNNetworkHelper::updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, float value) {
    auto inData = quantizeLayer.insData[constLayerIndex].lock();
    if (inData == nullptr) {
        THROW_IE_EXCEPTION << "data is absent";
    }

    CNNLayerPtr blobLayer = inData->getCreatorLayer().lock();
    if (blobLayer == nullptr) {
        THROW_IE_EXCEPTION << "layer is absent";
    }

    if (blobLayer->blobs.size() != 1) {
        THROW_IE_EXCEPTION << "unexpected blobs size";
    }

    const auto existingBlobIt = blobLayer->blobs.begin();
    const auto& existingBlobTensorDesc = existingBlobIt->second->getTensorDesc();
    Blob::Ptr newBlob = makeNewBlobPtr(existingBlobTensorDesc);

    newBlob->allocate();
    fillBlobByFP32(newBlob, value);
    blobLayer->blobs[existingBlobIt->first] = newBlob;
}

int CNNNetworkHelper::onWeightsInDepth(const CNNLayer& layer) {
    const std::vector<CNNLayerPtr> children = getChildren(layer);
    for (const CNNLayerPtr& child : children) {
        if ((CaselessEq<std::string>()(child->type, "Convolution") ||
            CaselessEq<std::string>()(child->type, "FullyConnected") ||
            CaselessEq<std::string>()(child->type, "Gemm")) &&
            (child->insData.size() >= 2lu)) {
            const std::vector<CNNLayerPtr> parents = getParentsRecursivelyExceptTypes(*child, {}, 1);
            for (const CNNLayerPtr& parent : parents) {
                if (parent->name == layer.name) {
                    return 1;
                }
            }
            return -1;
        }

        const int result = onWeightsInDepth(*child);
        if (result != 0) {
            return result;
        }
    }
    return 0;
}

bool CNNNetworkHelper::onWeights(const CNNLayer& layer) {
    const int result = onWeightsInDepth(layer);
    return result == 1;
}

bool CNNNetworkHelper::onConstWeightsPath(const CNNLayer& quantize) {
    CNNLayerPtr parent = CNNNetworkHelper::getParent(quantize, 0);
    if (parent == nullptr) {
        THROW_IE_LPT_EXCEPTION(quantize) << "parent layer is nullable";
    }

    return parent->type == "Const";
}

size_t CNNNetworkHelper::getIndex(const CNNLayer& layer) {
    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
    if (children.size() != 1) {
        THROW_IE_EXCEPTION << "not supported";
    }

    for (size_t i = 0; i < children[0]->insData.size(); ++i) {
        const DataPtr insData = children[0]->insData[i].lock();
        if (insData == nullptr) {
            continue;
        }
        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if ((parent != nullptr) && (parent->name == layer.name)) {
            return i;
        }
    }

    THROW_IE_EXCEPTION << "not found";
}

std::vector<CNNLayerPtr> CNNNetworkHelper::transformFakeQuantizeToConst(TransformationContext& context,
                                                                        const CNNLayerPtr fakeQuantize,
                                                                        const Blob::Ptr weights,
                                                                        const std::string& constLayerName) {
    std::vector<CNNLayerPtr> constLayersToRemove;
    constLayersToRemove.reserve(fakeQuantize->insData.size());

    for (const DataWeakPtr& insDataWeak : fakeQuantize->insData) {
        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_IE_EXCEPTION << "input data for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
            THROW_IE_EXCEPTION << "input layer for FakeQuantize '" << fakeQuantize->name << "' is nullable";
        }
        if (!CaselessEq<std::string>()(parent->type, "Const") || (parent->insData.size() != 0lu)) {
            THROW_IE_EXCEPTION << "unexpected FakeQuantize input layer type " << parent->type << " for layer '"
                               << fakeQuantize->name << "' is nullable";
        }

        constLayersToRemove.push_back(parent);
    }

    for (const CNNLayerPtr& parent : constLayersToRemove) {
        CNNNetworkHelper::removeLayer(context.network, parent);
        context.removeLayer(*parent);
    }

    if (fakeQuantize->outData.size() != 1lu) {
        THROW_IE_EXCEPTION << "FakeQuantize " << fakeQuantize->name << " has several outputs";
    }

    const DataPtr outData = fakeQuantize->outData[0];
    if (outData == nullptr) {
        THROW_IE_EXCEPTION << "FakeQuantize output data is nullable";
    }

    // const Precision precision = outData->getPrecision();
    const auto inputTo = outData->getInputTo();
    std::vector<CNNLayerPtr> constLayers;
    for (auto it : inputTo) {
        const CNNLayerPtr child = it.second;
        if (child == nullptr) {
            THROW_IE_EXCEPTION << "child layer for FakeQuantize " << fakeQuantize->name << " is nullable";
        }

        constLayers.push_back(
            CNNNetworkHelper::addConstBetween(context.network, fakeQuantize, child, weights, constLayerName));
    }

    CNNNetworkHelper::removeLayer(context.network, fakeQuantize);
    context.removeLayer(*fakeQuantize);

    return constLayers;
}

void CNNNetworkHelper::setOutDataPrecision(const CNNLayer& layer, const Precision& precision) {
    for (const DataPtr& data : layer.outData) {
        data->setPrecision(precision);
    }
}

void CNNNetworkHelper::setOutDataPrecision(const std::vector<CNNLayerPtr>& layers, const Precision& precision) {
    for (const CNNLayerPtr layer : layers) {
        setOutDataPrecision(*layer, precision);
    }
}

void CNNNetworkHelper::setOutDataPrecision(const CNNLayer& beginLayer, const size_t branchWithEndBeforeLayer,
                                           const CNNLayer& endBeforeLayer, const Precision& precision) {
    CNNLayerPtr child = std::make_shared<CNNLayer>(beginLayer);
    while (child->name != endBeforeLayer.name) {
        CNNNetworkHelper::setOutDataPrecision(*child, precision);
        std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*child);
        if (child->name == beginLayer.name) {
            if (branchWithEndBeforeLayer >= children.size()) {
                THROW_IE_EXCEPTION << "branch with end before layer is out of children count " << children.size();
            }
            child = children[branchWithEndBeforeLayer];
        } else {
            if (children.size() != 1) {
                THROW_IE_EXCEPTION << "not supported";
            }

            child = children[0];
        }
    }
}

bool CNNNetworkHelper::IsChild(const std::vector<CNNLayerPtr>& children,
                               const std::unordered_set<std::string>& layerTypes,
                               const std::unordered_set<std::string>& ignoreLayerTypes) {
    for (const CNNLayerPtr& child : children) {
        if (layerTypes.find(child->type) != layerTypes.end()) {
            return true;
        }
        if (ignoreLayerTypes.find(child->type) != ignoreLayerTypes.end()) {
            if (child->outData.size() != 1) {
                return true;
            }
            if (IsChild(CNNNetworkHelper::getChildren(*child), layerTypes, ignoreLayerTypes)) {
                return true;
            }
        }
    }
    return false;
}

size_t CNNNetworkHelper::getOutputChannelsCount(const CNNLayer& layer, bool isOnWeights) {
    if (layer.outData.empty()) {
        THROW_IE_EXCEPTION << "Layer " << layer.name << " doesn't have output tensors";
    }

    auto& data = layer.outData[0];
    if (isOnWeights) {
        if (data->getDims().empty()) {
            THROW_IE_EXCEPTION << "Invalid dimensions count (0) in output of " << layer.name << " layer on weights";
        }
        return data->getDims()[0];
    } else {
        if (data->getDims().empty()) {
            THROW_IE_EXCEPTION << "Invalid dimensions count (0) in output of " << layer.name << " layer on activations";
        }
        if (data->getDims().size() == 1ul) {
            return data->getDims()[0];
        }
        return data->getDims()[1];
    }
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getLayers(const CNNLayer& parent, const CNNLayer& child) {
    std::vector<CNNLayerPtr> layers;
    CNNLayerPtr tmpChild = std::make_shared<CNNLayer>(child);
    while (tmpChild != nullptr) {
        const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*tmpChild);
        for (const CNNLayerPtr tmpParent : parents) {
            if (tmpParent->name == parent.name) {
                return layers;
            }
        }

        if (parents.size() == 0) {
            THROW_IE_EXCEPTION << "not found";
        }

        if (parents.size() != 1ul) {
            THROW_IE_EXCEPTION << "not supported";
        }

        layers.push_back(parents[0]);
        tmpChild = parents[0];
    }
    return layers;
}

Blob::Ptr CNNNetworkHelper::getBlob(const CNNLayer* layer, const std::string& blobName) {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "layer is nullable";
    }

    if (blobName.empty()) {
        if (layer->blobs.empty()) {
            THROW_IE_LPT_EXCEPTION(*layer) << "does not have any blob";
        }

        if (layer->blobs.size() != 1) {
            THROW_IE_LPT_EXCEPTION(*layer) << "there are several blobs";
        }
        return layer->blobs.begin()->second;
    }

    const auto it = layer->blobs.find(blobName);
    if (it == layer->blobs.end()) {
        THROW_IE_LPT_EXCEPTION(*layer) << " does not have blob " << blobName;
    }

    return it->second;
}

Blob::Ptr CNNNetworkHelper::getBlob(CNNLayerPtr layer, const std::string& blobName) {
    return getBlob(layer.get(), blobName);
}

std::shared_ptr<float> CNNNetworkHelper::getFloatData(const Blob::Ptr& srcBlob) {
    if (srcBlob == nullptr) {
        THROW_IE_EXCEPTION << "Invalid blob";
    }

    const auto& precision = srcBlob->getTensorDesc().getPrecision();
    if (!isBlobPrecisionSupported(precision)) {
        THROW_IE_EXCEPTION << "precision '" << precision << "' is not supported";
    }

    const size_t dataSize = srcBlob->size();
    std::shared_ptr<float> floatPtr(new float[dataSize], std::default_delete<float[]>());

    if (precision == Precision::FP32) {
        const float* srcData = srcBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::FP16) {
        const short* srcData = srcBlob->buffer().as<short*>();
        PrecisionUtils::f16tof32Arrays(floatPtr.get(), srcData, dataSize, 1.f, 0.f);
    } else if (precision == Precision::I8) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::U8) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::I32) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::I64) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::I64>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else if (precision == Precision::U64) {
        const auto* srcData = srcBlob->buffer().as<PrecisionTrait<Precision::U64>::value_type*>();
        std::copy(srcData, srcData + dataSize, floatPtr.get());
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }

    return floatPtr;
}

bool CNNNetworkHelper::isBlobPrecisionSupported(const Precision precision) {
    return (precision == Precision::FP32) ||
        (precision == Precision::FP16) ||
        (precision == Precision::I8) ||
        (precision == Precision::U8) ||
        (precision == Precision::I32) ||
        (precision == Precision::I64) ||
        (precision == Precision::U64);
}

std::shared_ptr<float> CNNNetworkHelper::getFloatData(const CNNLayerPtr& layer, const std::string& blobName) {
    const Blob::Ptr blob = getBlob(layer, blobName);
    if (blob == nullptr) THROW_IE_EXCEPTION << "Could not find blob '" << blobName << "' for layer " << layer->name;

    return getFloatData(blob);
}

void CNNNetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, const float* srcData) {
    if (dstBlob == nullptr) THROW_IE_EXCEPTION << "Invalid blob";

    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == Precision::FP32) {
        float* dstData = dstBlob->buffer().as<float*>();
        std::copy(srcData, srcData + dataSize, dstData);
    } else if (precision == Precision::FP16) {
        short* dstData = dstBlob->buffer().as<short*>();
        PrecisionUtils::f32tof16Arrays(dstData, srcData, dataSize, 1.f, 0.f);
    } else if (precision == Precision::I8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<PrecisionTrait<Precision::I8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == Precision::U8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<PrecisionTrait<Precision::U8>::value_type>(std::roundf(srcData[i]));
        }
    } else if (precision == Precision::I32) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData[i] = static_cast<PrecisionTrait<Precision::I32>::value_type>(std::roundf(srcData[i]));
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

std::shared_ptr<float> CNNNetworkHelper::convertFloatData(const float* srcData, const size_t dataSize,
                                                          const Precision precision) {
    std::shared_ptr<float> dstData(new float[dataSize], std::default_delete<float[]>());

    if (precision == Precision::FP32) {
        std::copy(srcData, srcData + dataSize, dstData.get());
    } else if (precision == Precision::FP16) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] = PrecisionUtils::f16tof32(PrecisionUtils::f16tof32(srcData[i]));
        }
    } else if (precision == Precision::I8) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::I8>::value_type>(std::roundf(srcData[i])));
        }
    } else if (precision == Precision::U8) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::U8>::value_type>(std::roundf(srcData[i])));
        }
    } else if (precision == Precision::I32) {
        for (size_t i = 0ul; i < dataSize; ++i) {
            dstData.get()[i] =
                static_cast<float>(static_cast<PrecisionTrait<Precision::I32>::value_type>(std::roundf(srcData[i])));
        }
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }

    return dstData;
}

void CNNNetworkHelper::fillBlobByFP32(const CNNLayerPtr& layer, const std::string& blobName, const float* srcData) {
    Blob::Ptr blob = getBlob(layer, blobName);
    return fillBlobByFP32(blob, srcData);
}

void CNNNetworkHelper::fillBlobByFP32(Blob::Ptr& dstBlob, float value) {
    const auto& precision = dstBlob->getTensorDesc().getPrecision();
    const size_t dataSize = dstBlob->size();

    if (precision == Precision::FP32) {
        float* dstData = dstBlob->buffer().as<float*>();
        std::fill(dstData, dstData + dataSize, value);
    } else if (precision == Precision::FP16) {
        short* dstData = dstBlob->buffer().as<short*>();
        const short s_value = PrecisionUtils::f32tof16(value);
        std::fill(dstData, dstData + dataSize, s_value);
    } else if (precision == Precision::I8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::I8>::value_type>(value));
    } else if (precision == Precision::U8) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::U8>::value_type>(value));
    } else if (precision == Precision::I32) {
        auto* dstData = dstBlob->buffer().as<PrecisionTrait<Precision::I32>::value_type*>();
        std::fill(dstData, dstData + dataSize, static_cast<PrecisionTrait<Precision::I32>::value_type>(value));
    } else {
        THROW_IE_EXCEPTION << "Unsupported transformation precision: " << precision;
    }
}

CNNLayerPtr CNNNetworkHelper::getParent(const CNNLayer& layer, const size_t index, const std::string& ignoreLayerType) {
    if (index >= layer.insData.size()) {
        return nullptr;
    }

    DataPtr inputLayerData = layer.insData[index].lock();
    if (inputLayerData == nullptr) {
        THROW_IE_EXCEPTION << "input data is absent";
    }

    CNNLayerPtr inputLayer;
    do {
        inputLayer = inputLayerData->getCreatorLayer().lock();
        if (!inputLayer) {
            THROW_IE_EXCEPTION << "input is absent";
        }

        if (inputLayer->type != ignoreLayerType) {
            break;
        }

        if (inputLayer->insData.size() == 0) {
            inputLayer = nullptr;
            break;
        }

        if (inputLayer->insData.size() != 1) {
            THROW_IE_EXCEPTION << "too much branches";
        }

        inputLayerData = inputLayer->insData[0].lock();
        if (inputLayerData == nullptr) {
            THROW_IE_EXCEPTION << "input data is absent";
        }
    } while (true);

    return inputLayer;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getParents(const CNNLayer& layer, const std::string& exceptionLayerName) {
    std::vector<CNNLayerPtr> parents;
    for (const DataWeakPtr insDataWeak : layer.insData) {
        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_IE_EXCEPTION << "input data is absent";
        }

        CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
            THROW_IE_EXCEPTION << "input layer is absent";
        }

        if (exceptionLayerName.empty() || parent->name != exceptionLayerName) {
            parents.push_back(parent);
        }
    }
    return parents;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getParentsRecursivelyExceptTypes(
    const CNNLayer& layer, const std::unordered_set<std::string>& exceptionLayerTypes, const int portIndex) {
    std::vector<CNNLayerPtr> parents;
    size_t i = 0ul;
    for (DataWeakPtr insDataWeak : layer.insData) {
        if (insDataWeak.expired()) {
            continue;
        }

        const DataPtr insData = insDataWeak.lock();
        if (insData == nullptr) {
            THROW_IE_EXCEPTION << "input data is absent";
        }

        CNNLayerWeakPtr parentWeak = insData->getCreatorLayer();
        if (parentWeak.expired()) {
            continue;
        }

        if ((portIndex == -1) || (portIndex == i)) {
            CNNLayerPtr parent = parentWeak.lock();
            if (parent == nullptr) {
                THROW_IE_EXCEPTION << "input layer is absent";
            }

            if (exceptionLayerTypes.find(parent->type) != exceptionLayerTypes.end()) {
                const std::vector<CNNLayerPtr> tmpParents = CNNNetworkHelper::getParentsRecursivelyExceptTypes(*parent, exceptionLayerTypes);
                parents.insert(parents.end(), tmpParents.begin(), tmpParents.end());
            } else {
                parents.push_back(parent);
            }
        }

        i++;
    }
    return parents;
}

size_t CNNNetworkHelper::getInputChannelsCount(const CNNLayer& layer) {
    if (layer.insData.size() == 0) {
        THROW_IE_EXCEPTION << "There are no input layers";
    }

    const DataPtr insertData = layer.insData[0].lock();
    if (insertData == nullptr) {
        THROW_IE_EXCEPTION << "insert data is absent";
    }

    switch (insertData->getLayout()) {
    case Layout::NC:
    case Layout::NCHW:
    case Layout::NCDHW: {
        return insertData->getDims()[1];
    }
    case Layout::CHW: {
        if (insertData->getDims().size() != 3lu) {
            THROW_IE_EXCEPTION << "Unexpected dimensions size " << insertData->getDims().size() << " for layer "
                               << layer.name;
        }

        // Actually MO assumes NCH layout for 3D blobs, so we get channels count from dimension 1
        return insertData->getDims()[1];
    }
    default: {
        THROW_IE_EXCEPTION << "Not supported layout " << insertData->getLayout();
    }
    }
}

size_t CNNNetworkHelper::getParamOutput(const CNNLayer& layer) {
    if (!layer.CheckParamPresence("output")) {
        THROW_IE_EXCEPTION << "convolution parameter 'output' is absent";
    }
    return layer.GetParamAsUInt("output");
}

size_t CNNNetworkHelper::getKernelSize(const CNNLayer& layer) {
    if (!layer.CheckParamPresence("kernel")) {
        THROW_IE_EXCEPTION << "convolution parameter 'kernel' is absent";
    }
    const auto dims = layer.GetParamAsUInts("kernel");
    if (dims.size() == 2) {
        return dims[0] * dims[1];
    } else if (dims.size() == 3) {
        return dims[0] * dims[1] * dims[2];
    } else {
        THROW_IE_EXCEPTION << "kernel dimensions are not correct";
    }
}

void CNNNetworkHelper::renameLayer(ICNNNetwork& net, const std::string& currentName, const std::string& newName) {
    CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
    if (netImpl == nullptr) {
        THROW_IE_EXCEPTION << "unexpected network type";
    }

    netImpl->renameLayer(currentName, newName);
}

CNNLayerPtr CNNNetworkHelper::addLayer(
        TransformationContext& context,
        const CNNLayerPtr parent,
        const CNNLayerPtr child,
        const CNNLayerPtr newLayer) {
    DataPtr outData;
    Precision precision;
    if (parent != nullptr) {
        // Searching the connection between the layers
        int l1_out_i = 0;
        if (child != nullptr) {
            for (; l1_out_i < parent->outData.size(); l1_out_i++) {
                if (parent->outData[l1_out_i]->getInputTo().find(child->name) !=
                    parent->outData[l1_out_i]->getInputTo().end()) {
                    break;
                }
            }
        }
        if (l1_out_i == parent->outData.size()) {
            if (child != nullptr)
                THROW_IE_EXCEPTION << "Can't find layer " << child->name << " among layer " << parent->name << " outputs";
            else
                THROW_IE_EXCEPTION << "Layer '" << parent->name << "' has invalid output";
        }

        outData = parent->outData[l1_out_i];
        precision = context.getOriginalLayerPrecision(parent->name, outData->getName());
        if (precision == Precision::UNSPECIFIED) {
            if (child != nullptr)
                precision = child->precision;
            else
                precision = Precision::FP32;
        }
    } else {
        // TODO: FIXME
        precision = Precision::FP32;
        outData = nullptr;
    }
    addLayerToCNNNetworkAfterData(outData, newLayer, child != nullptr ? child->name : "", context.network);

    CNNNetworkHelper::setOutDataPrecision(*newLayer, precision);
    return newLayer;
}

void CNNNetworkHelper::replaceLayer(TransformationContext& context, const CNNLayerPtr source, const CNNLayerPtr target) {
    CNNNetworkImpl* networkImpl = dynamic_cast<CNNNetworkImpl*>(&context.network);
    networkImpl->removeLayer(source->name);

    std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*source);
    for (CNNLayerPtr parent : parents) {
        for (size_t outDataIndex = 0ul; outDataIndex < parent->outData.size(); ++outDataIndex) {
            const DataPtr outData = parent->outData[outDataIndex];
            std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
            inputTo[source->name] = target;
            target->insData.push_back(outData);
        }
    }

    const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(*source);

    target->outData.resize(source->outData.size());
    for (size_t outDataIndex = 0ul; outDataIndex < source->outData.size(); ++outDataIndex) {
        const DataPtr outData = source->outData[outDataIndex];
        networkImpl->removeData(outData->getName());

        DataPtr newOutData(new Data(outData->getName(), outData->getTensorDesc()));
        newOutData->getCreatorLayer() = target;
        target->outData[outDataIndex] = newOutData;
        networkImpl->addData(newOutData->getName().c_str(), newOutData);

        std::map<std::string, CNNLayerPtr> inputTo = outData->getInputTo();
        for (const auto it : inputTo) {
            const CNNLayerPtr child = it.second;
            newOutData->getInputTo().emplace(it.first, child);

            for (const CNNLayerPtr& child : children) {
                for (size_t insDataIndex = 0ul; insDataIndex < child->insData.size(); ++insDataIndex) {
                    const DataPtr insData = child->insData[insDataIndex].lock();
                    if (insData == nullptr) {
                        THROW_IE_LPT_EXCEPTION(*child) << "insert data " << insDataIndex << " is absent";
                    }

                    const CNNLayerPtr parent = insData->getCreatorLayer().lock();
                    if (parent == nullptr) {
                        THROW_IE_LPT_EXCEPTION(*child) << "parent layer for insert data " << insDataIndex << " is absent";
                    }
                    if (parent->name == source->name) {
                        const auto it = target->outData[outDataIndex];
                        child->insData[insDataIndex] = newOutData;
                    }
                }
            }
        }
        outData->getInputTo().clear();
    }

    networkImpl->addLayer(target);
}

CNNLayerPtr CNNNetworkHelper::addScaleShiftBetween(TransformationContext& context, const CNNLayerPtr parent,
                                                   const CNNLayerPtr child,
                                                   const DequantizationDetails& dequantizationDetails,
                                                   const std::string& name) {
    if (parent == nullptr)
        THROW_IE_EXCEPTION << "Parent layer is nullable";

    if (child && (child->type == "ScaleShift") && (CNNNetworkHelper::getParents(*child).size() == 1)) {
        auto scalesIt = child->blobs.find("weights");
        if (scalesIt == child->blobs.end()) {
            THROW_IE_EXCEPTION << "weights for layer " << child->name << " was not found";
        }
        const std::shared_ptr<float> scales = CNNNetworkHelper::getFloatData(scalesIt->second);
        std::vector<float> updatedScales(scalesIt->second->size());
        for (size_t i = 0ul; i < updatedScales.size(); ++i) {
            updatedScales[i] = scales.get()[i] * dequantizationDetails.scales[i];
        }
        CNNNetworkHelper::updateBlobs(*child, "weights", updatedScales);

        auto shiftsIt = child->blobs.find("biases");
        if (shiftsIt != child->blobs.end()) {
            const std::shared_ptr<float> shifts = CNNNetworkHelper::getFloatData(shiftsIt->second);
            std::vector<float> updatedShifts(shiftsIt->second->size());
            for (size_t i = 0ul; i < updatedShifts.size(); ++i) {
                updatedShifts[i] = scales.get()[i] * dequantizationDetails.shifts[i] + shifts.get()[i];
            }
            CNNNetworkHelper::updateBlobs(*child, "biases", updatedShifts);
        }

        return child;
    }

    // Searching the connection between the layers
    int l1_out_i = 0;
    if (child != nullptr) {
        for (; l1_out_i < parent->outData.size(); l1_out_i++) {
            if (parent->outData[l1_out_i]->getInputTo().find(child->name) !=
                parent->outData[l1_out_i]->getInputTo().end()) {
                break;
            }
        }
    }
    if (l1_out_i == parent->outData.size()) {
        if (child != nullptr)
            THROW_IE_EXCEPTION << "Can't find layer " << child->name << " among layer " << parent->name << " outputs";
        else
            THROW_IE_EXCEPTION << "Layer '" << parent->name << "' has invalid output";
    }

    DataPtr outData = parent->outData[l1_out_i];

    std::string layerName = name.empty() ? (child != nullptr ? (parent->name + "_ScaleShift_" + child->name)
                                                             : (parent->name + "_ScaleShift"))
                                         : name;

    Precision ssPrecision = context.getOriginalLayerPrecision(parent->name, outData->getName());
    if (ssPrecision == Precision::UNSPECIFIED) {
        if (child != nullptr)
            ssPrecision = child->precision;
        else
            ssPrecision = Precision::FP32;
    }

    LayerParams ssCnnLayerParams {layerName, "ScaleShift", ssPrecision};
    CNNLayerPtr ssCnnLayer(new ScaleShiftLayer(ssCnnLayerParams));

    const std::vector<size_t> dims = outData->getDims();

    if ((dims.size() != 2ul) || ((dims.size() == 2ul) && (dims[0] != dequantizationDetails.channelsCount))) {
        if ((dims.size() > 1) && (dims[1] != dequantizationDetails.channelsCount)) {
            THROW_IE_EXCEPTION << "unexpected parent channels count " << dims[1];
        }
    }
    addLayerToCNNNetworkAfterData(outData, ssCnnLayer, child != nullptr ? child->name : "", context.network);

    {
        ScaleShiftLayer* scshLayer = dynamic_cast<ScaleShiftLayer*>(ssCnnLayer.get());
        if (scshLayer == nullptr) {
            THROW_IE_EXCEPTION << "Layer " << ssCnnLayer->name << " is not instance of ScaleShiftLayer class";
        }
        fillInScaleShift(
            scshLayer,
            dequantizationDetails.channelsCount,
            dequantizationDetails.scales.data(),
            dequantizationDetails.shifts.data());
    }

    CNNNetworkHelper::setOutDataPrecision(*ssCnnLayer, ssPrecision);
    return ssCnnLayer;
}

CNNLayerPtr CNNNetworkHelper::addConstBetween(ICNNNetwork& net, const CNNLayerPtr layer1, const CNNLayerPtr layer2,
                                              const Blob::Ptr customBlob, const std::string& name) {
    if (layer1 == nullptr)
        THROW_IE_EXCEPTION << "First layer is nullable";
    // Searching the connection between the layers
    int l1_out_i = 0;
    if (layer2 != nullptr) {
        for (; l1_out_i < layer1->outData.size(); l1_out_i++) {
            if (layer1->outData[l1_out_i]->getInputTo().find(layer2->name) !=
                layer1->outData[l1_out_i]->getInputTo().end()) {
                break;
            }
        }
    }

    if (l1_out_i == layer1->outData.size()) {
        if (layer2 != nullptr)
            THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " outputs";
        else
            THROW_IE_EXCEPTION << "Layer " << layer1->name << " has invalid outputs";
    }

    DataPtr outData = layer1->outData[l1_out_i];

    std::string layerName = name.empty() ? layer1->name + "_Const" : name;
    CNNLayerPtr layer(new CNNLayer({layerName, "Const", customBlob->getTensorDesc().getPrecision()}));

    addLayerToCNNNetworkAfterData(outData, layer, layer2 != nullptr ? layer2->name : "", net);
    layer->blobs.emplace("custom", customBlob);
    layer->outData[0]->setPrecision(customBlob->getTensorDesc().getPrecision());
    return layer;
}

void CNNNetworkHelper::addLayerToCNNNetworkAfterData(
    DataPtr parentOutData,
    CNNLayer::Ptr layer,
    const std::string& nextLayerName,
    ICNNNetwork& net) {
    CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
    if (netImpl == nullptr) {
        THROW_IE_EXCEPTION << "unexpected network type";
    }

    CNNLayerPtr nextLayer;
    if (!nextLayerName.empty()) {
        netImpl->getLayerByName(nextLayerName.c_str(), nextLayer, nullptr);
    }

    if (layer && (nextLayerName.empty() || (parentOutData == nullptr) ||
                  (parentOutData->getInputTo().find(nextLayerName) != parentOutData->getInputTo().end()))) {
        auto getTensorDesc = [](CNNLayerPtr& nextLayer) {
            const DataPtr insData = nextLayer->insData[0].lock();
            if (insData == nullptr) {
                THROW_IE_LPT_EXCEPTION(*nextLayer) << "insert data is absent";
            }
            return insData->getTensorDesc();
        };

        const TensorDesc& parentTensorDesc = parentOutData != nullptr ? parentOutData->getTensorDesc() : getTensorDesc(nextLayer);
        DataPtr newEdgeAfterLayer(new Data(layer->name, parentTensorDesc));
        newEdgeAfterLayer->setName(layer->name);
        newEdgeAfterLayer->getCreatorLayer() = layer;
        newEdgeAfterLayer->getInputTo().clear();

        CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
        if (netImpl == nullptr) {
            THROW_IE_EXCEPTION << "unexpected network type";
        }
        netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
        IE_SUPPRESS_DEPRECATED_START
        netImpl->addLayer(layer);
        IE_SUPPRESS_DEPRECATED_END

        if (parentOutData != nullptr) {
            parentOutData->getInputTo()[layer->name] = layer;
            layer->insData.push_back(parentOutData);
        }
        layer->outData.push_back(newEdgeAfterLayer);

        if (!nextLayerName.empty()) {
            // CNNLayerPtr nextLayer = parentOutData->getInputTo()[nextLayerName];
            newEdgeAfterLayer->getInputTo()[nextLayerName] = nextLayer;
            if (parentOutData != nullptr) {
                parentOutData->getInputTo().erase(nextLayerName);
                for (size_t i = 0; i < nextLayer->insData.size(); i++) {
                    if (nextLayer->insData[i].lock() == parentOutData) {
                        nextLayer->insData[i] = newEdgeAfterLayer;
                    }
                }
            } else {
                // TODO: why new?
                nextLayer->insData.push_back(newEdgeAfterLayer);
            }
        } else {
            CNNLayerPtr parent = parentOutData->getCreatorLayer().lock();
            if (parent == nullptr) {
                THROW_IE_EXCEPTION << "parent data is absent";
            }
            netImpl->removeOutput(parent->name);
            netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
            netImpl->addOutput(layer->name);
        }
    } else {
        THROW_IE_EXCEPTION << "Invalid argument";
    }
}

void CNNNetworkHelper::fillInScaleShift(ScaleShiftLayer* layer, const size_t channels, const float* scales,
                                        const float* shifts) {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "ScaleShiftLayer is nullable";
    }

    layer->_weights = makeNewBlobPtr({layer->precision, {channels}, Layout::C});
    layer->_weights->allocate();
    fillBlobByFP32(layer->_weights, scales);
    layer->blobs["weights"] = layer->_weights;

    layer->_biases = makeNewBlobPtr({layer->precision, {channels}, Layout::C});
    layer->_biases->allocate();
    fillBlobByFP32(layer->_biases, shifts);
    layer->blobs["biases"] = layer->_biases;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getChildren(const CNNLayer& layer, const std::string& exceptionLayerName) {
    std::vector<CNNLayerPtr> children;
    for (const DataPtr outData : layer.outData) {
        const std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
        for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
            CNNLayerPtr child = it->second;
            if (exceptionLayerName.empty() || child->name != exceptionLayerName) {
                children.push_back(child);
            }
        }
    }
    return children;
}

std::vector<CNNLayerPtr> CNNNetworkHelper::getChildrenRecursivelyExceptTypes(
    const CNNLayer& layer, const std::unordered_set<std::string>& exceptionLayerTypes) {
    std::vector<CNNLayerPtr> children;
    for (const DataPtr outData : layer.outData) {
        const std::map<std::string, CNNLayerPtr>& inputTo = outData->getInputTo();
        for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
            CNNLayerPtr child = it->second;
            if (exceptionLayerTypes.find(child->type) != exceptionLayerTypes.end()) {
                const std::vector<CNNLayerPtr> tmpChildren =
                    getChildrenRecursivelyExceptTypes(*child, exceptionLayerTypes);
                children.insert(children.end(), tmpChildren.begin(), tmpChildren.end());
                continue;
            }

            children.push_back(child);
        }
    }
    return children;
}

void CNNNetworkHelper::checkConstWithBlobs(const CNNLayerPtr layer) {
    if (layer->type != "Const") {
        THROW_IE_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 0) {
        THROW_IE_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name
                           << "'";
    }
    if (layer->outData.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name
                           << "'";
    }
}

void CNNNetworkHelper::checkQuantizeOnWeights(const CNNLayerPtr layer) {
    if (layer->type != "FakeQuantize") {
        THROW_IE_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 0) {
        THROW_IE_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 5) {
        THROW_IE_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name
                           << "'";
    }
    if (layer->outData.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name
                           << "'";
    }
}

void CNNNetworkHelper::updateInput(CNNNetworkImpl* network, CNNLayerPtr& layer, DataPtr outData) {
    if (!CaselessEq<std::string>()(layer->type, "Input")) {
        return;
    }

    InputInfo::Ptr inputInfo = network->getInput(layer->name);
    if (inputInfo->name() == layer->name) {
        inputInfo->setInputData(outData);
    }
}

size_t CNNNetworkHelper::disconnectLayers(CNNNetworkImpl* network, const CNNLayerPtr& parentLayer,
                                          const CNNLayerPtr& childLayer) {
    bool wasFound = false;
    for (auto dataIt = parentLayer->outData.begin(); dataIt != parentLayer->outData.end(); ++dataIt) {
        auto data = *dataIt;
        for (auto inputIt = data->getInputTo().begin(); inputIt != data->getInputTo().end(); ++inputIt) {
            auto currentChildLayer = inputIt->second;
            if (currentChildLayer == nullptr) {
                THROW_IE_EXCEPTION << "Output layer for '" << parentLayer->name << "'is absent";
            }
            if (currentChildLayer->name == childLayer->name) {
                data->getInputTo().erase(inputIt);
                wasFound = true;
                break;
            }
        }

        if (wasFound) {
            break;
        }
    }
    if (!wasFound) {
        THROW_IE_EXCEPTION << "Output layer '" << childLayer->name << "' was not found for '" << parentLayer->name
                           << "'";
    }

    wasFound = false;
    for (auto it = childLayer->insData.begin(); it != childLayer->insData.end(); ++it) {
        auto data = it->lock();
        if (data == nullptr) {
            THROW_IE_EXCEPTION << "Input layer data for '" << childLayer->name << "'is absent";
        }
        auto currentParentLayer = data->getCreatorLayer().lock();
        if (currentParentLayer == nullptr) {
            THROW_IE_EXCEPTION << "Input layer for '" << childLayer->name << "'is absent";
        }
        if (currentParentLayer->name == parentLayer->name) {
            childLayer->insData.erase(it);
            wasFound = true;
            break;
        }
    }
    if (!wasFound) {
        THROW_IE_EXCEPTION << "Input layer '" << parentLayer->name << "' was not found for '" << childLayer->name
                           << "'";
    }
    return 0;
}

size_t CNNNetworkHelper::getInputIndex(const CNNLayerPtr& childLayer, const CNNLayerPtr& parentLayer) {
    for (size_t index = 0; index < childLayer->insData.size(); ++index) {
        DataPtr currentParenData = childLayer->insData[index].lock();
        if (currentParenData == nullptr) {
            THROW_IE_EXCEPTION << "parent layer data is absent";
        }
        CNNLayerPtr currentParrentLayer = currentParenData->getCreatorLayer().lock();
        if (currentParrentLayer == nullptr) {
            THROW_IE_EXCEPTION << "parent layer is absent";
        }
        if (currentParrentLayer->name == parentLayer->name) {
            return index;
        }
    }

    THROW_IE_EXCEPTION << "parent layer was not found";
}

void CNNNetworkHelper::removeLayer(ICNNNetwork& network, const CNNLayerPtr& layer) {
    details::CNNNetworkImpl* networkImpl = dynamic_cast<details::CNNNetworkImpl*>(&network);
    if (networkImpl == nullptr) {
        THROW_IE_EXCEPTION << "Unexpected network type";
    }

    if (layer->outData.size() > 1) {
        THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has too many outputs " << layer->outData.size();
    }

    if (layer->insData.size() > 1) {
        do {
            DataPtr data = layer->insData[0].lock();
            if (data == nullptr) {
                THROW_IE_EXCEPTION << "Layer's inserted data is nullptr";
            }
            CNNLayerPtr parentLayer = data->getCreatorLayer().lock();
            if (parentLayer == nullptr) {
                THROW_IE_EXCEPTION << "Layer's parent layer is nullptr";
            }
            CNNNetworkHelper::removeLayer(network, parentLayer);
        } while (!layer->insData.empty());
    }

    DataPtr childData;
    std::vector<CNNLayerPtr> children;
    std::vector<size_t> childrenIndexes;
    if (layer->outData.size() > 0) {
        childData = layer->outData[0];
        auto inputTo = childData->getInputTo();
        if (inputTo.size() == 0) {
            std::vector<CNNLayerPtr> parents = getParents(*layer);
            if (parents.size() != 1) {
                THROW_IE_EXCEPTION << "not possible remove output layer with several parents";
            }
            networkImpl->addOutput(parents[0]->name);
            CNNNetworkImpl* networkImpl = dynamic_cast<CNNNetworkImpl*>(&network);
            networkImpl->removeOutput(layer->name);
        } else {
            for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
                children.push_back(it->second);
                childrenIndexes.push_back(getInputIndex(it->second, layer));
                disconnectLayers(networkImpl, layer, it->second);
            }
        }
    }

    if (layer->insData.size() > 1) {
        // TODO: implement
        THROW_IE_EXCEPTION << "not implemented";
    }

    DataPtr parentData;
    CNNLayerPtr parentLayer;
    if (layer->insData.size() > 0) {
        // remove connections with parent layers
        parentData = layer->insData[0].lock();
        if (parentData == nullptr) {
            THROW_IE_EXCEPTION << "Input data is absent";
        }
        parentLayer = parentData->getCreatorLayer().lock();
        if (parentLayer == nullptr) {
            THROW_IE_EXCEPTION << "Input layer for '" << layer->name << "' is absent";
        }

        const size_t ouputLayerOutDataIndex = disconnectLayers(networkImpl, parentLayer, layer);
        if (ouputLayerOutDataIndex >= parentLayer->outData.size()) {
            THROW_IE_EXCEPTION << "Index " << ouputLayerOutDataIndex << " out of range output ports count "
                               << parentLayer->outData.size() << " for layer " << parentLayer->name;
        }

        for (size_t index = 0; index < children.size(); ++index) {
            CNNLayerPtr childLayer = children[index];
            const size_t childInputIndex = childrenIndexes[index];

            DataPtr outData = parentLayer->outData[ouputLayerOutDataIndex];
            outData->getInputTo().emplace(childLayer->name, childLayer);
            childLayer->insData.insert(childLayer->insData.begin() + childInputIndex, outData);

            updateInput(networkImpl, parentLayer, outData);
        }
    }

    networkImpl->removeData(layer->name);
    networkImpl->removeLayer(layer->name);
}

bool CNNNetworkHelper::isWeightsSupported(const CNNLayer& layer) noexcept {
    if (layer.insData.size() > 1) {
        CNNLayerPtr weightsLayer = CNNNetworkHelper::getParent(layer, 1);
        if (weightsLayer == nullptr)
            return false;
        if ((weightsLayer->type == "Const") || (weightsLayer->type == "FakeQuantize")) {
            return true;
        }

        if (weightsLayer->type == "ScaleShift") {
            const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(*weightsLayer);
            if (parents.size() != 1ul) {
                return false;
            }

            return (parents[0]->type == "FakeQuantize") || (parents[0]->type == "Const");
        }

        return false;
    } else {
        return layer.blobs.find("weights") != layer.blobs.end();
    }
}

Blob::Ptr CNNNetworkHelper::getWeights(
        const CNNLayer& layer,
        const bool roundQuantizedValues) {
    if (layer.insData.size() > 1) {
        CNNLayerPtr weightsLayer = CNNNetworkHelper::getParent(layer, 1);
        if (weightsLayer == nullptr) {
            THROW_IE_EXCEPTION << "Convolution weights const layer are absent";
        }

        if (weightsLayer->type == "Const") {
            CNNNetworkHelper::checkConstWithBlobs(weightsLayer);
            return weightsLayer->blobs.find("custom")->second;
        } else if (weightsLayer->type == "FakeQuantize") {
            return CNNNetworkHelper::quantizeWeights(*weightsLayer, roundQuantizedValues, Precision::UNSPECIFIED);
        } else if (weightsLayer->type == "ScaleShift") {
            const CNNLayerPtr parent = CNNNetworkHelper::getParent(*weightsLayer);
            if (parent == nullptr)
                THROW_IE_EXCEPTION << "Layer '" << weightsLayer->name << "' does not have parent";
            if (parent->type == "FakeQuantize") {
                return CNNNetworkHelper::quantizeWeights(*parent, roundQuantizedValues, Precision::UNSPECIFIED);
            } else if (parent->type == "Const") {
                CNNNetworkHelper::checkConstWithBlobs(parent);
                return CNNNetworkHelper::getBlob(parent, "custom");
            } else {
                THROW_IE_EXCEPTION << "Unexpected weights layer " << parent->type << " " << parent->name << " for " << layer.type << " " << layer.name;
            }
        } else {
            THROW_IE_EXCEPTION << "Unexpected weights layer type " << weightsLayer->type;
        }
    } else {
        if (layer.blobs.find("weights") == layer.blobs.end()) {
            THROW_IE_EXCEPTION << "Convolution weights are absent";
        }
        return layer.blobs.find("weights")->second;
    }
}

Blob::Ptr CNNNetworkHelper::getBiases(const CNNLayer& layer) {
    if (layer.insData.size() > 1U) {
        if (layer.insData.size() > 2U) {
            CNNLayerPtr biasesLayer = CNNNetworkHelper::getParent(layer, 2U);
            if (biasesLayer == nullptr) {
                return nullptr;
            }

            CNNNetworkHelper::checkConstWithBlobs(biasesLayer);
            return biasesLayer->blobs.find("custom")->second;
        } else {
            return nullptr;
        }
    } else {
        const auto it = layer.blobs.find("biases");
        return (it != layer.blobs.end()) ? it->second : nullptr;
    }
}

Blob::Ptr CNNNetworkHelper::quantizeWeights(const CNNLayer& quantize, const bool roundValues, const Precision precision) {
    if (quantize.insData.size() != 5lu) {
        THROW_IE_EXCEPTION << "Unexpected inputs count: " << quantize.insData.size();
    }
    for (int i = 0; i < quantize.insData.size(); i++)
        if (quantize.insData[i].lock() == nullptr)
            THROW_IE_EXCEPTION << "Invalid input data for layer '" << quantize.name << "' with index " << i;

    const Blob::Ptr sourceBlob = getQuantizeLayerBlob(quantize);
    if (sourceBlob == nullptr) {
        THROW_IE_EXCEPTION << "weights blob is empty for " << quantize.type << " layer " << quantize.name;
    }

    const auto& sourceBlobTD = sourceBlob->getTensorDesc();
    const Precision blobPrecision = sourceBlobTD.getPrecision();

    auto targetBlobPrecision = precision == Precision::UNSPECIFIED ? blobPrecision : precision;
    if (targetBlobPrecision != Precision::FP32 && targetBlobPrecision != Precision::FP16 &&
        targetBlobPrecision != Precision::I8 && targetBlobPrecision != Precision::U8)
        THROW_IE_EXCEPTION << "Unexpected precision: " << precision;

    Blob::Ptr targetBlob = make_blob_with_precision(TensorDesc(targetBlobPrecision, sourceBlobTD.getDims(), sourceBlobTD.getLayout()));
    targetBlob->allocate();

    quantizeBlob(quantize, targetBlob, roundValues);

    return targetBlob;
}

bool CNNNetworkHelper::isQuantizedConstWeights(const CNNLayer& layer) {
    CNNLayerPtr quantize = CNNNetworkHelper::getParent(layer, 1);
    if (quantize == nullptr) {
        return false;
    }

    if (quantize->type == "Const") {
        return true;
    }

    if (quantize->type != "FakeQuantize") {
        return false;
    }

    if (quantize->insData.size() != 5ul) {
        THROW_IE_LPT_EXCEPTION(*quantize) << "unexpected inputs size";
    }

    return onConstWeightsPath(*quantize);
}

int CNNNetworkHelper::getConstParentBranchID(const CNNLayer& layer) {
    int constBranchID = -1;
    for (int i = 0; i < layer.insData.size(); i++) {
        bool allConst = true;

        const DataPtr insData = layer.insData[i].lock();
        if (insData == nullptr) {
            THROW_IE_LPT_EXCEPTION(layer) << "invalid input data with index " << i;
        }

        const CNNLayerPtr parent = insData->getCreatorLayer().lock();
        if (parent == nullptr) {
            THROW_IE_LPT_EXCEPTION(layer) << "parent layer is absent";
        }

        if (!CaselessEq<std::string>()(parent->type, "FakeQuantize")) continue;
        for (const auto& p : parent->insData) {
            const DataPtr parentConstInsData = p.lock();
            if (parentConstInsData == nullptr) {
                THROW_IE_LPT_EXCEPTION(*parent) << "input data is absent";
            }
            const CNNLayerPtr parentConst = parentConstInsData->getCreatorLayer().lock();
            if (parentConst == nullptr) {
                THROW_IE_LPT_EXCEPTION(*parent) << "input layer is absent";
            }
            if (!CaselessEq<std::string>()(parentConst->type, "Const")) {
                allConst = false;
                break;
            }
        }
        if (allConst) {
            constBranchID = i;
            break;
        }
    }

    return constBranchID;
}

Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer) {
    return getPrecisionParent(layer, 0ul, false);
}

Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex) {
    return getPrecisionParent(layer, parentIndex, true);
}

Precision CNNNetworkHelper::getPrecisionParent(const CNNLayer& layer, const size_t parentIndex, const bool useParentIndex) {
    const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
    if (parents.empty()) {
        THROW_IE_EXCEPTION << "parents for layer " << layer.type << " '" << layer.name << "' are absent";
    }

    if (useParentIndex) {
        DataPtr parentOutData = getOutData(*parents[parentIndex], layer);
        if (parentOutData == nullptr) {
            THROW_IE_EXCEPTION <<
                "parent layer " << parents[parentIndex]->type << " '" << parents[parentIndex]->name <<
                "' output data  was not found for child " << layer.type << " '" << layer.name << "'";
        }
        return parentOutData->getTensorDesc().getPrecision();
    }

    Precision parentOutDataPrecision = Precision::UNSPECIFIED;
    for (CNNLayerPtr parent : parents) {
        DataPtr parentOutData = getOutData(*parent, layer);
        if (parentOutData == nullptr) {
            THROW_IE_EXCEPTION <<
                "parent layer " << parent->type << " '" << parent->name <<
                "' output data  was not found for child " << layer.type << " '" << layer.name << "'";
        }

        if (parentOutDataPrecision == Precision::UNSPECIFIED) {
            parentOutDataPrecision = parentOutData->getTensorDesc().getPrecision();
        } else if (parentOutDataPrecision != parentOutData->getTensorDesc().getPrecision()) {
            THROW_IE_EXCEPTION <<
                "Parent layer " << parent->type << " '" << parent->name <<
                "' output port has unexpected precision " << parentOutData->getTensorDesc().getPrecision();
        }
    }

    return parentOutDataPrecision;
}

DataPtr CNNNetworkHelper::getOutData(const CNNLayer& parentLayer, const CNNLayer& childLayer) {
    DataPtr parentOutData;
    for (DataPtr outData : parentLayer.outData) {
        const std::map<std::string, CNNLayerPtr> inputTo = outData->getInputTo();
        for (auto childIt : inputTo) {
            if (childIt.second->name == childLayer.name) {
                parentOutData = outData;
                break;
            }
        }

        if (parentOutData != nullptr) {
            break;
        }
    }
    return parentOutData;
}

void CNNNetworkHelper::quantizeBlob(const CNNLayer& quantize, Blob::Ptr& targetBlob, bool roundValues) {
    const Blob::Ptr sourceBlob = getQuantizeLayerBlob(quantize);
    if (sourceBlob == nullptr) {
        THROW_IE_EXCEPTION << "quantized blob is empty for " << quantize.type << " layer " << quantize.name;
    }

    auto srcData = getFloatData(sourceBlob);
    const std::vector<size_t>& outDims = quantize.outData[0]->getDims();
    if (outDims.empty() || outDims.size() > 5lu) {
        THROW_IE_EXCEPTION << "Unexpected dimensions count " << outDims.size() << " for layer '" << quantize.name << "'";
    }

    // OIDHW
    const size_t OC = outDims[0];
    const size_t IC = outDims.size() > 1lu ? outDims[1] : 1;
    const size_t D  = outDims.size() > 4lu ? outDims[outDims.size() - 3] : 1;
    const size_t H  = outDims.size() > 2lu ? outDims.size() == 3lu ? outDims[2] : outDims[outDims.size() - 2] : 1;
    const size_t W  = outDims.size() > 3lu ? outDims[outDims.size() - 1] : 1;

    // Const layer blob shape (sourceBlob->getTensorDesc().getDims()) can be different from output port shape
    // CVS-27850: [IE COMMON] Align Const layer blob shape with output port shape
    if (sourceBlob->size() != OC * IC * D * H * W) {
        THROW_IE_EXCEPTION << "Unexpected weights size for layer '" << quantize.name << "'";
    }

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(quantize);

    const bool isInputLowBroadcasted = quantizationDetails.inputLowValues.size() != OC;
    if ((quantizationDetails.inputLowValues.size() != 1) && (quantizationDetails.inputLowValues.size() != OC)) {
        THROW_IE_EXCEPTION << "Unexpected input low values count " << quantizationDetails.inputLowValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    const bool isInputHighBroadcasted = quantizationDetails.inputHighValues.size() != OC;
    if ((quantizationDetails.inputHighValues.size() != 1) && (quantizationDetails.inputHighValues.size() != OC)) {
        THROW_IE_EXCEPTION << "Unexpected input high values count " << quantizationDetails.inputHighValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    const bool isOutputLowBroadcasted = quantizationDetails.outputLowValues.size() != OC;
    if ((quantizationDetails.outputLowValues.size() != 1) && (quantizationDetails.outputLowValues.size() != OC)) {
        THROW_IE_EXCEPTION << "Unexpected output low values count " << quantizationDetails.outputLowValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    const bool isOutputHighBroadcasted = quantizationDetails.outputHighValues.size() != OC;
    if ((quantizationDetails.outputHighValues.size() != 1) && (quantizationDetails.outputHighValues.size() != OC)) {
        THROW_IE_EXCEPTION << "Unexpected output high values count " << quantizationDetails.outputHighValues.size() <<
            " for " << OC << " channels, layer '" << quantize.name << "'";
    }

    auto levels_1 = static_cast<float>(quantize.GetParamAsUInt("levels")) - 1.f;

    const size_t DHW = D * H * W;
    const size_t IDHW = IC * DHW;

    std::vector<float> dstBuffer(targetBlob->size());

    auto srcPtr = srcData.get();
    auto dstPtr = &dstBuffer[0];

    parallel_for4d(OC, IC, D, H, [&](size_t oc, size_t ic, size_t d, size_t h) {
        const float inputLow = quantizationDetails.inputLowValues[isInputLowBroadcasted ? 0 : oc];
        const float inputHigh = quantizationDetails.inputHighValues[isInputHighBroadcasted ? 0 : oc];
        const float outputLow = quantizationDetails.outputLowValues[isOutputLowBroadcasted ? 0 : oc];
        const float outputHigh = quantizationDetails.outputHighValues[isOutputHighBroadcasted ? 0 : oc];

        for (size_t w = 0; w < W; w++) {
            const size_t idx = oc * IDHW + ic * DHW + d * H * W + h * W + w;

            if (srcPtr[idx] <= inputLow) {
                dstPtr[idx] = roundValues ? std::roundf(outputLow) : outputLow;
            } else if (srcPtr[idx] > inputHigh) {
                dstPtr[idx] = roundValues ? std::roundf(outputHigh) : outputHigh;
            } else {
                const float value = std::roundf((srcPtr[idx] - inputLow) / (inputHigh - inputLow) * levels_1) /
                                    levels_1 * (outputHigh - outputLow) + outputLow;
                dstPtr[idx] = roundValues ? std::roundf(value) : value;
            }
        }
    });

    fillBlobByFP32(targetBlob, dstPtr);
}
