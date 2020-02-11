// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cmath>
#include <memory>
#include <string>
#include <vector>
#include <unordered_set>

#include "cnn_network_impl.hpp"

#include "low_precision_transformations/common/dequantization_details.hpp"
#include "low_precision_transformations/transformation_context.hpp"
#include "low_precision_transformations/quantization_details.hpp"

namespace InferenceEngine {
namespace details {

/**
    * @brief CNNNetworkHelper class encapsulates manipulations with CNN Network.
    */
class INFERENCE_ENGINE_API_CLASS(CNNNetworkHelper) {
public:
    static CNNLayerPtr getLayer(const ICNNNetwork& network, const std::string& layerName);

    static Blob::Ptr makeNewBlobPtr(const TensorDesc& desc);

    static void invertFakeQuantize(const CNNLayer& fakeQuantize);

    static void updateBlobs(CNNLayer& layer, const std::string& blobName, float value);

    static void updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, float value);

    static void updateBlobs(const CNNLayer& quantizeLayer, int constLayerIndex, const std::vector<float>& values);

    static void updateBlobs(CNNLayer& layer, const std::string& blobName, const std::vector<float>& values);

    // return true if at least one child uses layer on weights
    static bool onWeights(const CNNLayer& layer);

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

    static Blob::Ptr getBlob(CNNLayer* layer, const std::string& blobName);

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

    static void fillInScaleShift(ScaleShiftLayer* layer, const size_t channels, const float* scales, const float* shifts);

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

    static Blob::Ptr getWeights(const CNNLayer& layer, const bool roundQuantizedValues, const std::vector<float>& weightsShiftPerChannel = {});

    static Blob::Ptr getBiases(const CNNLayer& layer);

    static Blob::Ptr quantizeWeights(
        const CNNLayer& quantize,
        const bool roundValues,
        const Precision precision = Precision::UNSPECIFIED,
        const std::vector<float>& weightsShiftPerChannel = {});

    static int getConstParentBranchID(const CNNLayer& layer);

    static int getFakeQuantizeBranchWithOneChild(const CNNLayer& layer);

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

        DataPtr data = quantize.insData[0].lock();
        if (data == nullptr) {
            THROW_IE_EXCEPTION << "parent data is absent for " << quantize.type << " layer " << quantize.name;
        }

        CNNLayerPtr blobLayer = data->getCreatorLayer().lock();
        if (blobLayer == nullptr) {
            THROW_IE_EXCEPTION << "parent layer is absent for " << quantize.type << " layer " << quantize.name;
        }

        if (blobLayer->blobs.size() != 1) {
            THROW_IE_EXCEPTION << "unexpected blobs count for " << blobLayer->type << " layer " << blobLayer->name;
        }

        return blobLayer->blobs.begin()->second;;
    }

    // TODO: don't need to define type for weights quantization: separate to two methods
    template <class T>
    static Blob::Ptr quantizeBlob(
            const CNNLayer& quantize,
            const bool roundValues,
            const Precision precision,
            const std::vector<float>& shiftPerChannel = {}) {
        const Blob::Ptr sourceBlob = getQuantizeLayerBlob(quantize);
        if (sourceBlob == nullptr) {
            THROW_IE_EXCEPTION << "quantized blob is empty for " << quantize.type << " layer " << quantize.name;
        }

        auto srcData = getFloatData(sourceBlob);
        const std::vector<size_t>& originalDims = quantize.outData[0]->getDims();
        const std::vector<size_t>& dims =
                originalDims.size() == 2lu ? std::vector<size_t>({ originalDims[0], originalDims[1], 1lu, 1lu, 1lu }) :
                originalDims.size() == 4lu ? std::vector<size_t>({ originalDims[0], originalDims[1], 1lu, originalDims[2], originalDims[3] }) :
                originalDims;
        if (dims.size() != 5lu) {
            THROW_IE_EXCEPTION << "Unexpected dimensions count " << dims.size() << " for layer '" << quantize.name << "'";
        }

        // OIDHW
        const size_t outputsSize = dims[0];  // O
        const size_t inputsSize = dims[1];  // I
        const size_t D = dims[2];  // D
        const size_t H = dims[3];  // H
        const size_t W = dims[4];  // W

        const auto& sourceBlobTensorDesc = sourceBlob->getTensorDesc();
        Blob::Ptr targetBlob = make_shared_blob<T>(TensorDesc(
            precision != Precision::UNSPECIFIED ? precision : sourceBlobTensorDesc.getPrecision(),
            sourceBlobTensorDesc.getDims(),
            sourceBlobTensorDesc.getLayout()));
        targetBlob->allocate();

        const size_t sourceBlobSize = sourceBlob->size();
        if (sourceBlobSize != (inputsSize * outputsSize * D * H * W)) {
            THROW_IE_EXCEPTION << "Unexpected weights dimensions "
                << outputsSize << "x"
                << inputsSize << "x"
                << D << "x"
                << H << "x"
                << W << " for layer '" << quantize.name << "'";
        }

        auto dstBuffer = getFloatData(targetBlob);

        const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(quantize);

        const bool isInputLowBroadcasted = quantizationDetails.inputLowValues.size() != outputsSize;
        if ((quantizationDetails.inputLowValues.size() != 1) && (quantizationDetails.inputLowValues.size() != outputsSize)) {
            THROW_IE_EXCEPTION << "Unexpected input low values count " << quantizationDetails.inputLowValues.size() <<
                " for " << outputsSize << " channels, layer '" << quantize.name << "'";
        }

        const bool isInputHighBroadcasted = quantizationDetails.inputHighValues.size() != outputsSize;
        if ((quantizationDetails.inputHighValues.size() != 1) && (quantizationDetails.inputHighValues.size() != outputsSize)) {
            THROW_IE_EXCEPTION << "Unexpected input high values count " << quantizationDetails.inputHighValues.size() <<
                " for " << outputsSize << " channels, layer '" << quantize.name << "'";
        }

        const bool isOutputLowBroadcasted = quantizationDetails.outputLowValues.size() != outputsSize;
        if ((quantizationDetails.outputLowValues.size() != 1) && (quantizationDetails.outputLowValues.size() != outputsSize)) {
            THROW_IE_EXCEPTION << "Unexpected output low values count " << quantizationDetails.outputLowValues.size() <<
                " for " << outputsSize << " channels, layer '" << quantize.name << "'";
        }

        const bool isOutputHighBroadcasted = quantizationDetails.outputHighValues.size() != outputsSize;
        if ((quantizationDetails.outputHighValues.size() != 1) && (quantizationDetails.outputHighValues.size() != outputsSize)) {
            THROW_IE_EXCEPTION << "Unexpected output high values count " << quantizationDetails.outputHighValues.size() <<
                " for " << outputsSize << " channels, layer '" << quantize.name << "'";
        }

        const float levels_1 = static_cast<float>(quantize.GetParamAsUInt("levels")) - 1.f;

        const size_t DHW = D * H * W;
        const size_t IDHW = inputsSize * DHW;

        for (size_t outputIndex = 0; outputIndex < outputsSize; outputIndex++) {
            for (size_t inputIndex = 0; inputIndex < inputsSize; inputIndex++) {
                for (size_t d = 0; d < D; d++) {
                    for (size_t h = 0; h < H; h++) {
                        for (size_t w = 0; w < W; w++) {
                            const float inputLow = quantizationDetails.inputLowValues[isInputLowBroadcasted ? 0 : outputIndex];
                            const float inputHigh = quantizationDetails.inputHighValues[isInputHighBroadcasted ? 0 : outputIndex];
                            const float outputLow = quantizationDetails.outputLowValues[isOutputLowBroadcasted ? 0 : outputIndex];
                            const float outputHigh = quantizationDetails.outputHighValues[isOutputHighBroadcasted ? 0 : outputIndex];

                            const size_t idx = outputIndex * IDHW + inputIndex * DHW + d * H * W + h * W + w;

                            if (srcData.get()[idx] <= inputLow) {
                                dstBuffer.get()[idx] = roundValues ? std::roundf(outputLow) : outputLow;
                            } else if (srcData.get()[idx] > inputHigh) {
                                dstBuffer.get()[idx] = roundValues ? std::roundf(outputHigh) : outputHigh;
                            } else {
                                const float value = std::roundf((srcData.get()[idx] - inputLow) / (inputHigh - inputLow) * levels_1) /
                                                    levels_1 * (outputHigh - outputLow) + outputLow;
                                dstBuffer.get()[idx] = roundValues ? std::roundf(value) : value;
                            }
                        }
                    }
                }
            }
        }

        fillBlobByFP32(targetBlob, dstBuffer.get());

        return targetBlob;
    }
};

}  // namespace details
}  // namespace InferenceEngine
