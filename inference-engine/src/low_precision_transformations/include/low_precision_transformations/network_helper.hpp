// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include <legacy/ie_layers.h>
#include <legacy/cnn_network_impl.hpp>

#include "low_precision_transformations/common/dequantization_details.hpp"
#include "low_precision_transformations/transformation_context.hpp"
#include "low_precision_transformations/quantization_details.hpp"

namespace InferenceEngine {
namespace details {

IE_SUPPRESS_DEPRECATED_START

class INFERENCE_ENGINE_API_CLASS(Subgraph) {
public:
    bool fillSubgraphForConcat(const CNNLayerPtr& concat, std::unordered_set<std::string>& handledLayers);
    bool empty() const;

    std::vector<CNNLayerPtr> quantizationLayers;
    std::vector<CNNLayerPtr> concatLayers;
    std::unordered_map<std::string, CNNLayer*> layers;

private:
    bool fillSubgraphForQuantization(const CNNLayerPtr& fakeQuantize, std::unordered_set<std::string>& handledLayers);
    bool fillSubgraphForIntermediate(const CNNLayerPtr& intermediate, std::unordered_set<std::string>& handledLayers);
    bool fill(const CNNLayerPtr& concat, std::unordered_set<std::string>& handledLayers);
};

/**
    * @brief CNNNetworkHelper class encapsulates manipulations with CNN Network.
    */
class INFERENCE_ENGINE_API_CLASS(CNNNetworkHelper) {
public:
    static Subgraph getSubgraph(const CNNLayer& concat);

    static CNNLayerPtr getLayer(const ICNNNetwork& network, const std::string& layerName);

    static Blob::Ptr makeNewBlobPtr(const TensorDesc& desc);

    static void invertFakeQuantize(const CNNLayer& fakeQuantize);

    static void updateBlobs(CNNLayer& layer, const std::string& blobName, float value);

    static void updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, float value);

    static void updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, const std::vector<float>& values);

    static void updateBlobs(CNNLayer& layer, const std::string& blobName, const std::vector<float>& values);

    // return true if at least one child uses layer on weights
    static bool onWeights(const CNNLayer& layer);

    static bool onConstWeightsPath(const CNNLayer& quantize);

    static size_t getIndex(const CNNLayer& layer);

    static std::vector<CNNLayerPtr> transformFakeQuantizeToConst(
        TransformationContext& context,
        const CNNLayerPtr fakeQuantize,
        const Blob::Ptr weights,
        const std::string& constLayerName);

    static void setOutDataPrecision(const CNNLayer& layer, const Precision& precision);

    static void setOutDataPrecision(const std::vector<CNNLayerPtr>& layers, const Precision& precision);

    static void setOutDataPrecision(
        const CNNLayer& beginLayer,
        const size_t branchWithEndBeforeLayer,
        const CNNLayer& endBeforeLayer,
        const Precision& precision);

    static bool IsChild(
        const std::vector<CNNLayerPtr>& children,
        const std::unordered_set<std::string>& layerTypes,
        const std::unordered_set<std::string>& ignoreLayerTypes = {});

    static size_t getOutputChannelsCount(const CNNLayer& layer, bool isOnWeights = false);

    static std::vector<CNNLayerPtr> getLayers(const CNNLayer& parent, const CNNLayer& child);

    static Blob::Ptr getBlob(CNNLayerPtr layer, const std::string& blobName);

    static Blob::Ptr getBlob(const CNNLayer* layer, const std::string& blobName);

    static std::shared_ptr<float> getFloatData(const CNNLayerPtr& layer, const std::string& blobName);

    static std::shared_ptr<float> getFloatData(const Blob::Ptr& srcBlob);

    static bool isBlobPrecisionSupported(const Precision precision);

    static void fillBlobByFP32(Blob::Ptr& dstBlob, float value);

    static void fillBlobByFP32(Blob::Ptr& dstBlob, const float* srcData);

    static void fillBlobByFP32(const CNNLayerPtr& layer, const std::string& blobName, const float* srcData);

    static std::shared_ptr<float> convertFloatData(const float* srcData, const size_t dataSize, const Precision precision);

    static CNNLayerPtr getParent(
        const CNNLayer& layer,
        const size_t index = 0,
        const std::string& ignoreLayerType = "");

    static std::vector<CNNLayerPtr> getParents(
        const CNNLayer& layer,
        const std::string& exceptionLayerName = "");

    static std::vector<CNNLayerPtr> getParentsRecursivelyExceptTypes(
        const CNNLayer& layer,
        const std::unordered_set<std::string>& exceptionLayerTypes = {},
        const int portIndex = -1);

    static size_t getInputChannelsCount(const CNNLayer& layer);

    static size_t getParamOutput(const CNNLayer& layer);

    static size_t getKernelSize(const CNNLayer& layer);

    static void renameLayer(ICNNNetwork& net, const std::string& currentName, const std::string& newName);

    static CNNLayerPtr addLayer(
        TransformationContext& context,
        const CNNLayerPtr parent,
        const CNNLayerPtr child,
        const CNNLayerPtr newLayer);

    static void replaceLayer(TransformationContext& context, const CNNLayerPtr source, const CNNLayerPtr target);

    static CNNLayerPtr addScaleShiftBetween(
        TransformationContext& context,
        const CNNLayerPtr parent,
        const CNNLayerPtr child,
        const DequantizationDetails& dequantizationDetails,
        const std::string& name = "");

    static CNNLayerPtr addConstBetween(
        ICNNNetwork& net,
        const CNNLayerPtr layer1,
        const CNNLayerPtr layer2,
        const Blob::Ptr customBlob,
        const std::string& name);

    static void addLayerToCNNNetworkAfterData(
        DataPtr parentOutData,
        CNNLayer::Ptr layer,
        const std::string& nextLayerName,
        ICNNNetwork& net);

    IE_SUPPRESS_DEPRECATED_START
    static void fillInScaleShift(ScaleShiftLayer* layer, const size_t channels, const float* scales, const float* shifts);
    IE_SUPPRESS_DEPRECATED_END

    static std::vector<CNNLayerPtr> getChildren(const CNNLayer& layer, const std::string& exceptionLayerName = "");

    static std::vector<CNNLayerPtr> getChildrenRecursivelyExceptTypes(
        const CNNLayer& layer,
        const std::unordered_set<std::string>& exceptionLayerTypes = {});

    static void checkConstWithBlobs(const CNNLayerPtr layer);

    static void checkQuantizeOnWeights(const CNNLayerPtr layer);

    static void updateInput(details::CNNNetworkImpl* network, CNNLayerPtr& layer, DataPtr outData);

    static size_t disconnectLayers(
        CNNNetworkImpl* network,
        const CNNLayerPtr& parentLayer,
        const CNNLayerPtr& childLayer);

    static size_t getInputIndex(const CNNLayerPtr& childLayer, const CNNLayerPtr& parentLayer);

    static void removeLayer(ICNNNetwork& network, const CNNLayerPtr& layer);

    static bool isWeightsSupported(const CNNLayer& layer) noexcept;

    static Blob::Ptr getWeights(const CNNLayer& layer, const bool roundQuantizedValues);

    static Blob::Ptr getBiases(const CNNLayer& layer);

    static Blob::Ptr quantizeWeights(
        const CNNLayer& quantize,
        const bool roundValues,
        const Precision precision = Precision::UNSPECIFIED);

    static bool isQuantizedConstWeights(const CNNLayer& quantize);

    static int getConstParentBranchID(const CNNLayer& layer);

    static Precision getPrecisionParent(const CNNLayer& layer);

    static Precision getPrecisionParent(const CNNLayer& layer, const size_t parentIndex);

    static DataPtr getOutData(const CNNLayer& parentLayer, const CNNLayer& childLayer);

private:
    // 1  - on weights
    // 0  - weightable layer was not found
    // -1 - on activations
    static int onWeightsInDepth(const CNNLayer& layer);

    static Precision getPrecisionParent(const CNNLayer& layer, const size_t parentIndex, const bool useParentIndex);

    static Blob::Ptr getQuantizeLayerBlob(const CNNLayer& quantize) {
        if (quantize.insData.size() < 1) {
            THROW_IE_EXCEPTION << "unexpected parents count for " << quantize.type << " layer " << quantize.name;
        }

        const DataPtr data = quantize.insData[0].lock();
        if (data == nullptr) {
            THROW_IE_EXCEPTION << "parent data is absent for " << quantize.type << " layer " << quantize.name;
        }

        IE_SUPPRESS_DEPRECATED_START
        const CNNLayerPtr blobLayer = getCreatorLayer(data).lock();
        if (blobLayer == nullptr) {
            THROW_IE_EXCEPTION << "parent layer is absent for " << quantize.type << " layer " << quantize.name;
        }
        IE_SUPPRESS_DEPRECATED_END

        checkConstWithBlobs(blobLayer);

        return blobLayer->blobs.begin()->second;
    }

    static void quantizeBlob(const CNNLayer& quantize, Blob::Ptr& targetBlob, bool roundValues);
};

IE_SUPPRESS_DEPRECATED_END

}  // namespace details
}  // namespace InferenceEngine
