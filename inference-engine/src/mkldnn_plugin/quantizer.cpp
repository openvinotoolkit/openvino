// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <string>
#include <fstream>
#include <utility>
#include <map>
#include <memory>
#include <limits>
#include <cmath>

#include <ie_common.h>
#include <blob_factory.hpp>
#include <details/caseless.hpp>
#include <details/ie_cnn_network_tools.h>
#include <data_stats.h>
#include "cnn_network_impl.hpp"
#include "network_serializer.h"
#include "quantizer.hpp"

#include "ie_util_internal.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

// #define QUANTIZATION_DUMP
#ifdef QUANTIZATION_DUMP
#   define QUANTIZATION_DUMP_DIR "C:\\Projects\\icv\\dump\\quantized"
#   define QUANTIZATION_ENABLE_DUMP(_x) { _x;}
#else
#   define QUANTIZATION_DUMP_DIR ""
#   define QUANTIZATION_ENABLE_DUMP(_x)
#endif

/**
 * @brief CNNNetworkHelper class encapsulates manipulations with CNN Network.
  */
class CNNNetworkHelper {
public:
    static void updateInput(details::CNNNetworkImpl* network, CNNLayerPtr& layer, DataPtr outData) {
        if (!CaselessEq<std::string>()(layer->type, "Input")) {
            return;
        }

        InputInfo::Ptr inputInfo = network->getInput(layer->name);
        if (inputInfo->name() == layer->name) {
            inputInfo->setInputData(outData);
        }
    }

    static void removeLayer(ICNNNetwork& network, CNNLayerPtr& layer) {
        details::CNNNetworkImpl* networkImpl = dynamic_cast<details::CNNNetworkImpl*>(&network);
        if (networkImpl == nullptr) {
            THROW_IE_EXCEPTION << "Unexpected network type";
        }

        if (layer->insData.size() > 1) {
            THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has too many inputs " << layer->insData.size();
        }
        if (layer->outData.size() > 1) {
            THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has too many outputs " << layer->outData.size();
        }

        DataPtr childData;
        std::vector<CNNLayerPtr> children;
        if (layer->outData.size() > 0) {
            childData = layer->outData[0];
            auto inputTo = childData->getInputTo();
            for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
                children.push_back(it->second);
                disconnectLayers(networkImpl, layer, it->second);
            }
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
                THROW_IE_EXCEPTION << "Index " << ouputLayerOutDataIndex <<
                    " out of range output ports count " << parentLayer->outData.size() <<
                    " for layer " << parentLayer->name;
            }

            for (auto it = children.begin(); it != children.end(); ++it) {
                auto childLayer = *it;
                DataPtr outData = parentLayer->outData[ouputLayerOutDataIndex];
                outData->getInputTo().emplace(childLayer->name, childLayer);
                childLayer->insData.push_back(outData);

                updateInput(networkImpl, parentLayer, outData);
            }
        }

        networkImpl->removeData(layer->name);
        networkImpl->removeLayer(layer->name);
    }

    static void insertClampBetween(
        ICNNNetwork& net,
        const CNNLayerPtr parentLayer,
        const CNNLayerPtr childLayer,
        const QuantizationDetails& quantizationDetails) {
        // Searching the connection between the layers
        int l1_out_i = 0;
        for (; l1_out_i < parentLayer->outData.size(); l1_out_i++) {
            if (parentLayer->outData[l1_out_i]->getInputTo().find(childLayer->name) != parentLayer->outData[l1_out_i]->getInputTo().end()) {
                break;
            }
        }
        if (l1_out_i == parentLayer->outData.size()) {
            THROW_IE_EXCEPTION << "Can't find layer " << childLayer->name << " among layer " << parentLayer->name << " outputs";
        }

        int l2_in_i = 0;
        for (; l2_in_i < childLayer->insData.size(); l2_in_i++) {
            if (childLayer->insData[l2_in_i].lock()->getCreatorLayer().lock() == parentLayer) {
                break;
            }
        }
        if (l2_in_i == childLayer->insData.size()) {
            THROW_IE_EXCEPTION << "Can't find layer " << childLayer->name << " among layer " << parentLayer->name << " inputs";
        }

        DataPtr parentOutData = parentLayer->outData[l1_out_i];

        {
            std::string layerName = parentLayer->name + "_Clamp_" + childLayer->name;
            LayerParams ssCnnLayerParams{ layerName, "Clamp", Precision::FP32 };

            CNNLayerPtr clampLayer = CNNLayerPtr(new ClampLayer(ssCnnLayerParams));
            if (quantizationDetails.outputLowValues.size() != 1) {
                THROW_IE_EXCEPTION << "Unexpected output low values count " << quantizationDetails.outputLowValues.size();
            }
            dynamic_cast<ClampLayer*>(clampLayer.get())->min_value = quantizationDetails.outputLowValues[0];
            if (quantizationDetails.outputHighValues.size() != 1) {
                THROW_IE_EXCEPTION << "Unexpected output low values count " << quantizationDetails.outputHighValues.size();
            }
            dynamic_cast<ClampLayer*>(clampLayer.get())->max_value = quantizationDetails.outputHighValues[0];

            addLayerToCNNNetworkAfterData(parentOutData, clampLayer, childLayer->name, net);

            Precision odPrecision = Precision::FP32;
            clampLayer->outData[0]->setPrecision(odPrecision);
        }
    }

    static void insertReLUBetween(
        ICNNNetwork& net,
        const CNNLayerPtr parentLayer,
        const CNNLayerPtr childLayer) {
        // Searching the connection between the layers
        int l1_out_i = 0;
        for (; l1_out_i < parentLayer->outData.size(); l1_out_i++) {
            if (parentLayer->outData[l1_out_i]->getInputTo().find(childLayer->name) != parentLayer->outData[l1_out_i]->getInputTo().end()) {
                break;
            }
        }
        if (l1_out_i == parentLayer->outData.size()) {
            THROW_IE_EXCEPTION << "Can't find layer " << childLayer->name << " among layer " << parentLayer->name << " outputs";
        }

        int l2_in_i = 0;
        for (; l2_in_i < childLayer->insData.size(); l2_in_i++) {
            if (childLayer->insData[l2_in_i].lock()->getCreatorLayer().lock() == parentLayer) {
                break;
            }
        }
        if (l2_in_i == childLayer->insData.size()) {
            THROW_IE_EXCEPTION << "Can't find layer " << childLayer->name << " among layer " << parentLayer->name << " inputs";
        }

        DataPtr parentOutData = parentLayer->outData[l1_out_i];

        {
            CNNLayerPtr newLayerPtr;
            {
                std::string layerName = parentLayer->name + "_ReLU_" + childLayer->name;
                LayerParams ssCnnLayerParams{ layerName, "ReLU", Precision::FP32 };

                ReLULayer* reluLayer = new ReLULayer(ssCnnLayerParams);
                newLayerPtr = CNNLayerPtr(reluLayer);
            }
            addLayerToCNNNetworkAfterData(parentOutData, newLayerPtr, childLayer->name, net);

            newLayerPtr->precision = Precision::I8;
            newLayerPtr->outData[0]->setPrecision(Precision::U8);
        }
    }

    static void addScaleShifts(
        ICNNNetwork& net,
        const std::map<std::string, const QuantizationDetails>& quantizationDetails) {
        const std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);
        std::vector<std::pair<CNNLayerPtr, CNNLayerPtr>> pairs;
        for (auto iter : sortedLayers) {
            for (int l1_out_i = 0; l1_out_i < iter->outData.size(); l1_out_i++) {
                for (auto nextIter : iter->outData[l1_out_i]->getInputTo()) {
                    CNNLayer::Ptr next = nextIter.second;

                    // Checking for an INT8 convolution or fully connected with FP32 output
                    if ((CaselessEq<std::string>()(iter->type, "Convolution") ||
                        CaselessEq<std::string>()(iter->type, "FullyConnected")) &&
                        (iter->precision == Precision::I8) &&
                        (next->precision == Precision::FP32) &&
                        (iter->outData[l1_out_i]->getPrecision() == Precision::FP32)) {
                        // Do nothing here only if iter provides data to fp32 layers
                        // MKLDNNPlugin will generate x8->f32 convolution

                    } else if ((iter->precision != Precision::FP32 && next->precision == Precision::FP32) ||
                        (iter->precision == Precision::FP32 && next->precision != Precision::FP32)) {
                        pairs.push_back(std::pair<CNNLayerPtr, CNNLayerPtr>(iter, next));
                    }
                }
            }
        }

        for (auto& pair : pairs) {
            addScaleShiftBetween(net, pair.first, pair.second, quantizationDetails);
        }
    }

    static void addScaleShiftBetween(
        ICNNNetwork& net,
        const CNNLayerPtr layer1,
        const CNNLayerPtr layer2,
        const std::map<std::string, const QuantizationDetails>& quantizationDetails) {
        if (CaselessEq<std::string>()(layer2->type, "priorbox") ||
            CaselessEq<std::string>()(layer2->type, "priorboxclustered")) {
            return;
        }

        // Searching the connection between the layers
        int l1_out_i = 0;
        for (; l1_out_i < layer1->outData.size(); l1_out_i++) {
            if (layer1->outData[l1_out_i]->getInputTo().find(layer2->name) != layer1->outData[l1_out_i]->getInputTo().end()) {
                break;
            }
        }
        if (l1_out_i == layer1->outData.size()) {
            THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " outputs";
        }

        int l2_in_i = 0;
        for (; l2_in_i < layer2->insData.size(); l2_in_i++) {
            if (layer2->insData[l2_in_i].lock()->getCreatorLayer().lock() == layer1) {
                break;
            }
        }
        if (l2_in_i == layer2->insData.size()) {
            THROW_IE_EXCEPTION << "Can't find layer " << layer2->name << " among layer " << layer1->name << " inputs";
        }

        DataPtr outData = layer1->outData[l1_out_i];

        Blob::Ptr oScaleBlob = nullptr;
        if (layer1->blobs.find("o-scale") != layer1->blobs.end()) {
            oScaleBlob = layer1->blobs["o-scale"];
        }

        Blob::Ptr iScaleBlob = nullptr;
        if (layer2->blobs.find("i-scale") != layer2->blobs.end()) {
            iScaleBlob = layer2->blobs["i-scale"];
            layer2->blobs.erase("i-scale");
        }

        if (iScaleBlob == nullptr && oScaleBlob == nullptr) {
            return;  // No multipliers found around this edge. We can't create a ScaleShift here;
        } else {
            // Creating a ScaleShiftLayer
            std::string prefix;
            float *iScaleBuffer = nullptr, *oScaleBuffer = nullptr;
            if (oScaleBlob != nullptr) {
                oScaleBuffer = static_cast<float*>(oScaleBlob->buffer());
                prefix += "o";
            }
            if (iScaleBlob != nullptr) {
                iScaleBuffer = static_cast<float*>(iScaleBlob->buffer());
                prefix += "i";
            }

            std::string layerName = layer1->name + "_" + prefix + "ScaleShift_" + layer2->name;
            LayerParams ssCnnLayerParams{ layerName, "ScaleShift", Precision::FP32 };
            CNNLayerPtr ssCnnLayer(new ScaleShiftLayer(ssCnnLayerParams));

            addLayerToCNNNetworkAfterData(outData, ssCnnLayer, layer2->name, net);

            size_t c = static_cast<size_t>(outData->getDims()[1]);

            {
                ScaleShiftLayer* scshLayer = dynamic_cast<ScaleShiftLayer*>(ssCnnLayer.get());
                if (scshLayer == nullptr) {
                    THROW_IE_EXCEPTION << "Layer " << ssCnnLayer->name << " is not instance of ScaleShiftLayer class";
                }
                fillInScaleShift(scshLayer, c, oScaleBuffer, iScaleBuffer);
            }

            Precision odPrecision = Precision::FP32;

            // TODO: can INT8 convolution be in Precision::U8?
            if ((layer2->precision == Precision::I8) || (layer2->precision == Precision::U8)) {
                const auto it = quantizationDetails.find(layer2->name);
                if (it == quantizationDetails.end()) {
                    THROW_IE_EXCEPTION << "Quantization details for layer '" << layer2->name << "' was not found";
                }
                odPrecision = it->second.hasNegativeOutput() ? Precision::I8 : Precision::U8;
            }
            ssCnnLayer->outData[0]->setPrecision(odPrecision);
        }
    }

    static void fillInScaleShift(ScaleShiftLayer* scshLayer, size_t c, float* weightsN, float* weightsD) {
        // Setting "scales"
        SizeVector weightsSize = { c };
        TensorDesc weightsDesc(Precision::FP32, weightsSize, InferenceEngine::C);
        scshLayer->_weights = InferenceEngine::make_shared_blob<float>(weightsDesc);
        scshLayer->_weights->allocate();
        float * weightsData = scshLayer->_weights->buffer();
        for (size_t i = 0; i < c; i++) {
            if (weightsN == nullptr && weightsD != nullptr) {
                weightsData[i] = 1.0 / weightsD[i];
            } else if (weightsD == nullptr && weightsN != nullptr) {
                weightsData[i] = weightsN[i];
            } else if (weightsN != nullptr && weightsD != nullptr) {
                weightsData[i] = weightsN[i] / weightsD[i];
            } else {
                weightsData[i] = 1.0;
            }
        }

        // Setting "shifts"
        SizeVector shiftsSize = { c };
        TensorDesc shiftsDesc(Precision::FP32, shiftsSize, InferenceEngine::C);
        scshLayer->_biases = InferenceEngine::make_shared_blob<float>(shiftsDesc);
        scshLayer->_biases->allocate();
        float * biasesData = scshLayer->_biases->buffer();
        for (size_t i = 0; i < c; i++) {
            biasesData[i] = 0.f;  // Setting to constant "0"
        }
    }

    static void addLayerToCNNNetworkAfterData(
        DataPtr parentOutData,
        CNNLayer::Ptr layer,
        const std::string& nextLayerName,
        ICNNNetwork& net) {
        CNNLayerPtr parentLayer = parentOutData->getCreatorLayer().lock();
        if (parentOutData && layer && parentLayer &&
            parentOutData->getInputTo().find(nextLayerName) != parentOutData->getInputTo().end()) {
            CNNLayerPtr nextLayer = parentOutData->getInputTo()[nextLayerName];

            DataPtr newEdgeAfterLayer(new Data(layer->name, parentOutData->getTensorDesc()));
            newEdgeAfterLayer->setName(layer->name);
            newEdgeAfterLayer->getCreatorLayer() = layer;
            newEdgeAfterLayer->getInputTo().clear();
            newEdgeAfterLayer->getInputTo()[nextLayerName] = nextLayer;
            newEdgeAfterLayer->setPrecision(Precision::FP32);

            CNNNetworkImpl* netImpl = dynamic_cast<CNNNetworkImpl*>(&net);
            if (netImpl == nullptr) {
                THROW_IE_EXCEPTION << "unexpected network type";
            }
            netImpl->addData(layer->name.c_str(), newEdgeAfterLayer);
            netImpl->addLayer(layer);

            parentOutData->getInputTo().erase(nextLayerName);
            parentOutData->getInputTo()[layer->name] = layer;

            layer->insData.push_back(parentOutData);
            layer->outData.push_back(newEdgeAfterLayer);

            for (size_t i = 0; i < nextLayer->insData.size(); i++) {
                if (nextLayer->insData[i].lock() == parentOutData) {
                    nextLayer->insData[i] = newEdgeAfterLayer;
                }
            }
        } else {
            THROW_IE_EXCEPTION << "Invalid argument";
        }
    }

    static size_t disconnectLayers(CNNNetworkImpl* network, CNNLayerPtr& parentLayer, CNNLayerPtr& childLayer) {
        bool wasFound = false;
        for (auto dataIt = parentLayer->outData.begin(); dataIt != parentLayer->outData.end(); ++dataIt) {
            auto data = *dataIt;
            for (auto inputIt = data->getInputTo().begin(); inputIt != data->getInputTo().end(); ++inputIt) {
                auto currentChildLayer = inputIt->second;
                if (currentChildLayer == nullptr) {
                    THROW_IE_EXCEPTION << "Output layer for '" << parentLayer->name << "'is absent";
                }
                if (currentChildLayer->name == childLayer->name) {
                    const DataPtr dataToRemove = network->getData(data->getName().c_str());
                    if (!dataToRemove) {
                        THROW_IE_EXCEPTION << "there is not data to remove";
                    }

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
            THROW_IE_EXCEPTION << "Output layer '" << childLayer->name << "' was not found for '" << parentLayer->name << "'";
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
            THROW_IE_EXCEPTION << "Input layer '" << parentLayer->name << "' was not found for '" << childLayer->name << "'";
        }
        return 0;
    }

    static std::vector<CNNLayerPtr> getChildren(CNNLayerPtr layer) {
        std::vector<CNNLayerPtr> children;
        for (DataPtr outData : layer->outData) {
            for (auto outDataIn : outData->getInputTo()) {
                children.push_back(outDataIn.second);
            }
        }

        return children;
    }

    /**
     * Analyses layer children and returns true if all children have expected types and precision.
     */
    static bool all(
        const std::vector<CNNLayerPtr>& layers,
        const std::string& expectedType,
        const Precision expectedPrecision = Precision::UNSPECIFIED) {
        for (const CNNLayerPtr layer : layers) {
            if (!CaselessEq<std::string>()(layer->type, expectedType)) {
                return false;
            }

            if ((expectedPrecision != Precision::UNSPECIFIED) && (layer->precision != expectedPrecision)) {
                return false;
            }
        }

        return true;
    }

    static bool all(
        const std::map<std::string, CNNLayerPtr>& layers,
        const std::string& expectedType,
        const Precision expectedPrecision = Precision::UNSPECIFIED) {
        for (const auto layer : layers) {
            if (!CaselessEq<std::string>()(layer.second->type, expectedType)) {
                return false;
            }

            if ((expectedPrecision != Precision::UNSPECIFIED) && (layer.second->precision != expectedPrecision)) {
                return false;
            }
        }

        return true;
    }

    static void removeQuantize(ICNNNetwork& network, CNNLayerPtr& quantize) {
        details::CNNNetworkImpl* networkImpl = dynamic_cast<details::CNNNetworkImpl*>(&network);
        if (networkImpl == nullptr) {
            THROW_IE_EXCEPTION << "Unexpected network type";
        }

        if (!CaselessEq<std::string>()(quantize->type, "quantize")) {
            THROW_IE_EXCEPTION << "Unexpected layer type '" << quantize->type << "' for layer '" << quantize->name << "'";
        }
        if (quantize->insData.size() != 5) {
            THROW_IE_EXCEPTION << "Unexpected inputs count " << quantize->insData.size() << " for quantize layer '" << quantize->name << "'";
        }
        if (quantize->outData.size() != 1) {
            THROW_IE_EXCEPTION << "Unexpected outputs count " << quantize->outData.size() << " for quantize layer '" << quantize->name << "'";
        }

        // first layer consists activations
        while (quantize->insData.size() != 1) {
            DataWeakPtr insData = *(quantize->insData.begin() + 1);

            DataPtr data = insData.lock();
            if (data == nullptr) {
                THROW_IE_EXCEPTION << "Input data for quantize layer '" << quantize->name << "' absent";
            }

            // TODO: reuse from ConstTransformer::validateQuantizeConst
            CNNLayerPtr constLayer = data->getCreatorLayer().lock();
            if (constLayer == nullptr) {
                THROW_IE_EXCEPTION << "Input layer for quantize layer '" << quantize->name << "' absent";
            }
            if (!CaselessEq<std::string>()(constLayer->type, "const")) {
                THROW_IE_EXCEPTION << "Unexpected input layer type '" << constLayer->type << "' for quantize layer '" << quantize->name << "'";
            }
            if (constLayer->insData.size() != 0) {
                THROW_IE_EXCEPTION << "Unexpected outputs count " << constLayer->insData.size() << " for constant layer '" << constLayer->name << "'";
            }
            if (constLayer->outData.size() != 1) {
                THROW_IE_EXCEPTION << "Unexpected outputs count " << constLayer->outData.size() << " for constant layer '" << constLayer->name << "'";
            }

            CNNNetworkHelper::removeLayer(network, constLayer);
        }

        CNNNetworkHelper::removeLayer(network, quantize);
    }

    static CNNLayerPtr getNextNot(const CNNLayerPtr layer, const std::string& excludeType) {
        CNNLayer::Ptr nextLayer = layer;
        do {
            if (nextLayer->outData.empty()) {
                return CNNLayer::Ptr();
            }

            if (nextLayer->outData.size() != 1) {
                THROW_IE_EXCEPTION << "Outputs of layer '" << layer->name << "' " << layer->outData.size() << " more then 1";
            }

            const auto children = nextLayer->outData[0]->getInputTo();
            if (children.empty()) {
                return CNNLayer::Ptr();
            }

            if (children.size() != 1) {
                THROW_IE_EXCEPTION << "Output layers of layer '" << layer->name << "' " << layer->outData.size() << " more then 1";
            }

            nextLayer = children.begin()->second;
        } while (CaselessEq<std::string>()(nextLayer->type, excludeType));

        return nextLayer;
    }
};

/**
 * Base class for quantization pattern implementation.
 */
class QuantizationPattern {
public:
    virtual ~QuantizationPattern() = default;
    virtual void definesExecutionPrecision(
        const std::map<std::string, const QuantizationDetails>& quantizationDetailsMap,
        CNNLayerPtr layer) = 0;
};

typedef std::shared_ptr<QuantizationPattern> QuantizationPatternPtr;

/**
 * ReLU layer applyed quantization pattern:
 * Conv(I8) --I8--> ReLU(?) --?--> Conv(I8)
 * ==<fuse>==
 * Conv(I8) --U8-->                Conv(I8)
 */
class ReLUQuantizationPattern : public QuantizationPattern {
public:
    void definesExecutionPrecision(
        const std::map<std::string, const QuantizationDetails>& quantizationDetailsMap,
        CNNLayerPtr layer) override {
        if (!CaselessEq<std::string>()(layer->type, "ReLU")) {
            THROW_IE_EXCEPTION << "Not supported layer type " << layer->type;
        }

        if (layer->insData.size() > 1) {
#ifndef NDEBUG
            std::cout << "not implemented: layer '" << layer->name << "' (" << layer->type << ") has several inputs" << std::endl;
#endif
        }
        DataPtr parentInsData = layer->insData[0].lock();
        if (!parentInsData) {
            THROW_IE_EXCEPTION << "Input data is not valid";
        }
        CNNLayerPtr parentLayer = parentInsData->getCreatorLayer().lock();
        if (!CaselessEq<std::string>()(parentLayer->type, "Convolution")) {
            return;
        }
        if (parentLayer->outData.size() != 1) {
            THROW_IE_EXCEPTION << "Convolution " << parentLayer->name << " has several output ports";
        }

        auto children = CNNNetworkHelper::getChildren(layer);
        if (CNNNetworkHelper::all(children, "Convolution", Precision::I8) &&
            (parentLayer->outData[0]->getPrecision() != Precision::FP32)) {
            layer->precision = Precision::I8;

            parentInsData->setPrecision(Precision::U8);

            Precision precision;
            for (auto layerData : layer->outData) {
                std::map<std::string, CNNLayerPtr> inputTo = layerData->getInputTo();
                for (auto it = inputTo.begin(); it != inputTo.end(); ++it) {
                    const auto quantizationDetails = quantizationDetailsMap.find(it->second->name)->second;
                    precision = quantizationDetails.hasNegativeOutput() ? Precision::I8 : Precision::U8;
                }
                layerData->setPrecision(precision);
            }
        } else {
            parentInsData->setPrecision(Precision::FP32);
            layer->precision = Precision::FP32;
        }
    }
};

/**
 * Convolution layer applyed quantization pattern:
 * Conv(I8) --(?)--> Conv(I8)
 * Conv(I8) --U8-->  Conv(I8)
 */
class ConvolutionQuantizationPattern : public QuantizationPattern {
public:
    void definesExecutionPrecision(
        const std::map<std::string, const QuantizationDetails>& quantizationDetailsMap,
        CNNLayerPtr layer) override {
        if (!CaselessEq<std::string>()(layer->type, "Convolution")) {
            THROW_IE_EXCEPTION << "Not supported layer type " << layer->type;
        }

        if (layer->precision != Precision::I8) {
            return;
        }

        if (layer->outData.size() > 1) {
#ifndef NDEBUG
            std::cout << "not implemented: layer '" << layer->name << "' (" << layer->type << ") has several outputs" << std::endl;
#endif
        }

        for (DataPtr outData : layer->outData) {
#ifndef NDEBUG
            if (outData->getInputTo().size() > 1) {
                std::cout << "not implemented: layer '" << layer->name << "' (" << layer->type << ") has several children layers" << std::endl;
            }
#endif

            for (auto outDataIn : outData->getInputTo()) {
                CNNLayerPtr childLayer = outDataIn.second;
                if (CaselessEq<std::string>()(childLayer->type, "Convolution")) {
                    const auto quantizationDetails = quantizationDetailsMap.find(childLayer->name)->second;
                    if (quantizationDetails.hasNegativeOutput()) {
                        outData->setPrecision(Precision::I8);
                    } else {
                        outData->setPrecision(Precision::U8);
                    }
                }
            }
        }
    }
};

// TODO: quantization patterns are hard coded, only for CPU
const std::map<std::string, QuantizationPatternPtr> cpuQuantizationPatterns = {
    { "Convolution", QuantizationPatternPtr(new ConvolutionQuantizationPattern()) },
    { "ReLU", QuantizationPatternPtr(new ReLUQuantizationPattern()) }
};

#ifndef NDEBUG
void printQuantizationDetails(const std::string& layerName, const QuantizationDetails& quantizationDetails) {
    std::cout << "Quantize layer '" << layerName << "':" << std::endl;
    std::cout << "\tquantizeDetails.inputLowValues[0] (" << quantizationDetails.inputLowValues.size() << "): " <<
        quantizationDetails.inputLowValues[0] << std::endl;
    std::cout << "\tquantizeDetails.inputHighValues[0] (" << quantizationDetails.inputHighValues.size() << "):" <<
        quantizationDetails.inputHighValues[0] << std::endl;
    std::cout << "\tquantizeDetails.outputLowValues[0] (" << quantizationDetails.outputLowValues.size() << "): " <<
        quantizationDetails.outputLowValues[0] << std::endl;
    std::cout << "\tquantizeDetails.outputHighValues[0] (" << quantizationDetails.outputHighValues.size() << "): " <<
        quantizationDetails.outputHighValues[0] << std::endl;
}
#endif

bool isReLULikeClamp(CNNLayer::Ptr layer) {
    if (CaselessEq<std::string>()(layer->type, "Clamp")) {
        ClampLayer *clamp = dynamic_cast<ClampLayer *>(layer.get());
        if (clamp == nullptr) {
            THROW_IE_EXCEPTION << "Int8 Normalizer error: cannot cast layer '" << layer->name << "' to Clamp";
        }
        return clamp->min_value == 0;
    }
    return false;
}

bool isClampLikeQuantizer(const QuantizationDetails& quantizationDetails) {
    return (quantizationDetails.inputChannels() == 1) &&
        (quantizationDetails.outputChannels() == 1) &&
        (quantizationDetails.outputLowValues[0] == quantizationDetails.inputLowValues[0]) &&
        (quantizationDetails.outputHighValues[0] == quantizationDetails.inputHighValues[0]) &&
        (quantizationDetails.inputLowValues[0] < quantizationDetails.inputHighValues[0]);
}

std::string getBlobDimention2(const Blob::Ptr blob) {
    size_t idx = blob->getTensorDesc().getDims().size();

    std::stringstream blobDimention;
    blobDimention << "[";
    for (auto &dim : blob->getTensorDesc().getDims()) {
        blobDimention << dim << ((--idx) != 0u ? ", " : "");
    }
    blobDimention << "]";

    return blobDimention.str();
}

void precisionColoring2(const CNNLayerPtr layer,
    ordered_properties &printed_properties,
    ordered_properties &node_properties) {
    // looking for the w-scale
    if (layer->blobs.find("w-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("w-scale", getBlobDimention2(layer->blobs.find("w-scale")->second)));
    }

    // looking for the oi-scale
    if (layer->blobs.find("oi-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("oi-scale", getBlobDimention2(layer->blobs.find("oi-scale")->second)));
    }

    // looking for the o-scale
    if (layer->blobs.find("o-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("o-scale", getBlobDimention2(layer->blobs.find("o-scale")->second)));
    }
    // looking for the i-scale
    if (layer->blobs.find("i-scale") != layer->blobs.end()) {
        printed_properties.insert(printed_properties.begin(),
            std::pair<std::string, std::string>("i-scale", getBlobDimention2(layer->blobs.find("i-scale")->second)));
    }

    printed_properties.insert(printed_properties.begin(),
        std::pair<std::string, std::string>("Precision", layer->precision == Precision::FP32 ? "FP32" : "I8"));

    if (layer->precision == Precision::FP32) {
        node_properties.emplace_back("fillcolor", "#5A5DF0");
    } else {
        node_properties.emplace_back("fillcolor", "#20F608");
    }
}


class ConstTransformerDumper {
public:
    static void dumpOriginalBlob(const std::string& layerName, Blob::Ptr originalBlob) {
        dumpBlob(getFileName(layerName) + "_original", originalBlob);
    }

    static void dumpQuantizedBlob(const std::string& layerName, Blob::Ptr quantizedBlob) {
        dumpBlob(getFileName(layerName) + "_quantized", quantizedBlob);
    }

    static void serialize(const std::string& irFilePath, const ICNNNetwork& network) {
        ResponseDesc response;
        const auto status = network.serialize(
            irFilePath + ".xml",
            irFilePath + ".bin",
            &response);

        if (status != StatusCode::OK) {
            THROW_IE_EXCEPTION << "ConstTransformerDumper: serialization failed: " << response.msg << ": " << irFilePath;
        }
    }

private:
    static std::string getFileName(const std::string& layerName) {
        std::string tmpLayerName = layerName;
        std::replace(tmpLayerName.begin(), tmpLayerName.end(), '\\', '_');
        std::replace(tmpLayerName.begin(), tmpLayerName.end(), '/', '_');
        std::replace(tmpLayerName.begin(), tmpLayerName.end(), ' ', '_');
        std::replace(tmpLayerName.begin(), tmpLayerName.end(), ':', '_');
        return tmpLayerName;
    }

    static void dumpBlob(const std::string& prefixFileName, Blob::Ptr dumpBlob) {
        const auto buffer = dumpBlob->buffer().as<float*>();

        auto dumpFilePath = QUANTIZATION_DUMP_DIR + std::string("/") + prefixFileName + ".txt";

        std::ofstream dumpFile;
        dumpFile.open(dumpFilePath);
        if (!dumpFile.is_open()) {
            THROW_IE_EXCEPTION << "ConstTransformerDumper: cannot create dump file by path: " << dumpFilePath;
        }

        const size_t blobSize = dumpBlob->size();
        switch (dumpBlob->getTensorDesc().getPrecision()) {
        case Precision::FP32:
        {
            auto *buffer = dumpBlob->buffer().as<float*>();
            for (size_t i = 0; i < blobSize; i++)
                dumpFile << buffer[i] << std::endl;
            break;
        }
        default:
        {
            THROW_IE_EXCEPTION << "ConstTransformerDumper: unexpected precision";
        }
        }
    }
};

void checkConstWithBlobs(const CNNLayerPtr& layer) {
    if (layer->type != "Const") {
        THROW_IE_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 0) {
        THROW_IE_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name << "'";
    }
    if (layer->outData.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name << "'";
    }
}

void checkQuantizeOnWeights(const CNNLayerPtr& layer) {
    if (layer->type != "Quantize") {
        THROW_IE_EXCEPTION << "Unexpected layer type '" << layer->name << "'";
    }
    if (layer->blobs.size() != 0) {
        THROW_IE_EXCEPTION << "Unexpected blobs count " << layer->blobs.size() << " for layer '" << layer->name << "'";
    }
    if (layer->insData.size() != 5) {
        THROW_IE_EXCEPTION << "Unexpected inputs count " << layer->insData.size() << " for layer '" << layer->name << "'";
    }
    if (layer->outData.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected outputs count " << layer->outData.size() << " for layer '" << layer->name << "'";
    }
}

QuantizationDetails::QuantizationDetails() :
    levels(),
    inputLowValues({}),
    inputHighValues({}),
    outputLowValues({}),
    outputHighValues({}) {}

QuantizationDetails::QuantizationDetails(const QuantizationDetails& quantizationDetails) :
    levels(quantizationDetails.levels),
    inputLowValues(quantizationDetails.inputLowValues),
    inputHighValues(quantizationDetails.inputHighValues),
    outputLowValues(quantizationDetails.outputLowValues),
    outputHighValues(quantizationDetails.outputHighValues) {}

QuantizationDetails::QuantizationDetails(
    const size_t levels,
    const std::vector<float>& inputLowValues,
    const std::vector<float>& inputHighValues,
    const std::vector<float>& outputLowValues,
    const std::vector<float>& outputHighValues) :
    levels(levels),
    inputLowValues(inputLowValues),
    inputHighValues(inputHighValues),
    outputLowValues(outputLowValues),
    outputHighValues(outputHighValues) {}

QuantizationDetails QuantizationDetails::getDetails(const CNNLayerPtr quantize) {
    if (quantize->insData.size() != 5) {
        THROW_IE_EXCEPTION << "Unexpected inputs size " << quantize->insData.size() << " for Quantize layer '" << quantize->name;
    }

    if (!quantize->CheckParamPresence("levels")) {
        THROW_IE_EXCEPTION << "Parameter 'levels' is absent for Quantize layer '" << quantize->name << "'";
    }

    const auto levels = quantize->GetParamAsInt("levels");

    const CNNLayerPtr inputLowLayer = quantize->insData[1].lock()->getCreatorLayer().lock();
    validate(inputLowLayer);
    const std::vector<float> inputLowValues = getBlobValue(inputLowLayer);

    const CNNLayerPtr inputHighLayer = quantize->insData[2].lock()->getCreatorLayer().lock();
    validate(inputHighLayer);
    const std::vector<float> inputHighValues = getBlobValue(inputHighLayer);

    const CNNLayerPtr outputLowLayer = quantize->insData[3].lock()->getCreatorLayer().lock();
    validate(outputLowLayer);
    const std::vector<float> outputLowValues = getBlobValue(outputLowLayer);

    const CNNLayerPtr outputHighLayer = quantize->insData[4].lock()->getCreatorLayer().lock();
    validate(outputHighLayer);
    const std::vector<float> outputHighValues = getBlobValue(outputHighLayer);

    if (inputLowValues.size() != inputHighValues.size()) {
        THROW_IE_EXCEPTION << "Quantize input values sizes are not equal for layer " << quantize->name;
    }

    if (outputLowValues.size() != outputHighValues.size()) {
        THROW_IE_EXCEPTION << "Quantize output values sizes are not equal for layer " << quantize->name;
    }

    return QuantizationDetails(levels, inputLowValues, inputHighValues, outputLowValues, outputHighValues);
}

bool QuantizationDetails::hasNegativeOutput() const {
    for (const float value : outputLowValues) {
        if (value < 0.f) {
            return true;
        }
    }

    for (const float value : outputHighValues) {
        if (value < 0.f) {
            return true;
        }
    }

    return false;
}

float QuantizationDetails::maxOutput(const size_t channel) const {
    const auto value = fmax(
        fabs(outputLowValues[outputLowValues.size() == 1 ? 0 : channel]),
        fabs(outputHighValues[outputHighValues.size() == 1 ? 0 : channel]));
    return value;
}

float QuantizationDetails::maxInput(const size_t channel) const {
    const auto value = fmax(
        fabs(outputLowValues[inputLowValues.size() == 1 ? 0 : channel]),
        fabs(outputHighValues[inputHighValues.size() == 1 ? 0 : channel]));
    return value;
}

size_t QuantizationDetails::inputChannels() const {
    return inputLowValues.size();
}

size_t QuantizationDetails::outputChannels() const {
    return outputLowValues.size();
}

void QuantizationDetails::validate(const CNNLayerPtr& constantLayer) {
    if (constantLayer == nullptr) {
        THROW_IE_EXCEPTION << "Quantize layer input is absent";
    }

    if (constantLayer->blobs.size() == 0) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' doesn't have blobs";
    }

    if (constantLayer->blobs.size() > 1) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' has too much blobs";
    }

    const auto blob = constantLayer->blobs.begin()->second;
    const auto byteSize = blob->byteSize();

    if ((blob->getTensorDesc().getDims().size() != 1) && (blob->getTensorDesc().getDims().size() != 4)) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimentions are not correct";
    }

    const auto tensorDesc = blob->getTensorDesc();
    const auto dims = tensorDesc.getDims();
    if ((tensorDesc.getDims().size() != 1) && (tensorDesc.getDims().size() != 4)) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' blob dimentions size " << dims.size() << " not correct";
    }

    if ((tensorDesc.getLayout() != Layout::C) && ((tensorDesc.getLayout() != Layout::NCHW))) {
        THROW_IE_EXCEPTION << "Quantize layer input '" << constantLayer->name << "' layout " << dims[0] << " not correct";
    }
}

std::vector<float> QuantizationDetails::getBlobValue(const CNNLayerPtr& constantLayer) {
    const auto blob = constantLayer->blobs.begin()->second;
    const auto size = blob->getTensorDesc().getDims()[0];
    auto buffer = blob->buffer().as<float*>();
    return std::vector<float>(buffer, buffer + size);
}

bool Quantizer::isNetworkSupported(const ICNNNetwork& network) {
    ICNNNetworkStats* statistics = nullptr;
    ResponseDesc response;
    const StatusCode sts = network.getStats(&statistics, &response);
    if (sts != StatusCode::OK) {
        THROW_IE_EXCEPTION << "Statistics were not received: " << response.msg;
    }

    if (!statistics->isEmpty()) {
        return false;
    }

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    for (auto layer : sortedLayers) {
        if (CaselessEq<std::string>()(layer->type, "quantize") &&
            // INT8 is supported only
            (QuantizationDetails::getDetails(layer).levels == 256)) {
            return true;
        }
    }

    return false;
}

static void updateExistingOrInsertClampBetween(
    ICNNNetwork& net,
    const CNNLayerPtr parentLayer,
    const bool parentWillBeRemoved,
    const CNNLayerPtr childLayer,
    const QuantizationDetails& quantizationDetails) {
    ClampLayer* clampLayer;
    if (parentWillBeRemoved) {
        CNNLayer::Ptr parent = parentLayer->insData[0].lock()->getCreatorLayer().lock();
        if (parent->type != "Clamp") {
            CNNNetworkHelper::insertClampBetween(net, parentLayer, childLayer, quantizationDetails);
            return;
        }
        clampLayer = dynamic_cast<ClampLayer*>(parent.get());
    } else {
        if (parentLayer->type != "Clamp") {
            CNNNetworkHelper::insertClampBetween(net, parentLayer, childLayer, quantizationDetails);
            return;
        }
        clampLayer = dynamic_cast<ClampLayer*>(parentLayer.get());
    }

    if (clampLayer == nullptr) {
        THROW_IE_EXCEPTION << "Unexpected Clamp layer instance type";
    }

    if (quantizationDetails.outputLowValues.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected output low values count " << quantizationDetails.outputLowValues.size();
    }
    if (clampLayer->min_value < quantizationDetails.outputLowValues[0]) {
        clampLayer->min_value = quantizationDetails.outputLowValues[0];
    }

    if (quantizationDetails.outputHighValues.size() != 1) {
        THROW_IE_EXCEPTION << "Unexpected output low values count " << quantizationDetails.outputHighValues.size();
    }
    if (clampLayer->max_value > quantizationDetails.outputHighValues[0]) {
        clampLayer->max_value = quantizationDetails.outputHighValues[0];
    }
}

void Quantizer::definesExecutionPrecision(
    ICNNNetwork& net,
    const std::map<std::string, const QuantizationDetails>& quantizationDetailsMap) {
    const auto quantizationPatterns = cpuQuantizationPatterns;

    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);
    for (CNNLayerPtr layer : sortedLayers) {
        const auto quantizationPatternIt = quantizationPatterns.find(layer->type);
        if (quantizationPatternIt == quantizationPatterns.end()) {
            continue;
        }
        const QuantizationPatternPtr quantizationPattern = quantizationPatternIt->second;
        quantizationPattern->definesExecutionPrecision(quantizationDetailsMap, layer);
    }
}

void Quantizer::propagateScaleFactors(ICNNNetwork& net) {}

void Quantizer::quantize(
    ICNNNetwork& net,
    const QuantizationPrecision precision,
    const bool transformQuantizeOnDataPath,
    const std::string& dotFilePath,
    const std::string& irFilePath) {
    CNNNetworkImpl* network = dynamic_cast<CNNNetworkImpl*>(&net);

    // Step #1: Move weights & remove Quantize on weights
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(net);
    for (CNNLayerPtr layer : sortedLayers) {
        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
        if ((weightableLayer != nullptr) &&
            (CaselessEq<std::string>()(weightableLayer->type, "convolution") ||
            CaselessEq<std::string>()(weightableLayer->type, "fullyconnected")) &&
            (weightableLayer->insData.size() > 1)) {
            transformQuantizeOnWeightsPath(net, layer);
        }
    }

    if (!irFilePath.empty()) {
        ConstTransformerDumper::serialize(irFilePath, net);
    }

    std::map<std::string, const QuantizationDetails> quantizationDetailsMap;

    // Step #2: Collect quantization details
    sortedLayers = CNNNetSortTopologically(net);
    for (CNNLayerPtr layer : sortedLayers) {
        if (CaselessEq<std::string>()(layer->type, "quantize")) {
            const auto quantizationDetails = QuantizationDetails::getDetails(layer);
            quantizationDetailsMap.emplace(layer->name, quantizationDetails);

            const auto children = layer->outData[0]->getInputTo();
            for (auto childIt = children.begin(); childIt != children.end(); ++childIt) {
                quantizationDetailsMap.emplace(childIt->second->name, quantizationDetails);
            }
        }
    }

    // Step #3: Analyze Clamp layers
    for (CNNLayerPtr layer : sortedLayers) {
        if (CaselessEq<std::string>()(layer->type, "clamp")) {
            ClampLayer* clampLayer = dynamic_cast<ClampLayer*>(layer.get());
            if (clampLayer == nullptr) {
                THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has not expected type";
            }

            CNNLayer::Ptr child = clampLayer->outData[0]->getInputTo().begin()->second;
            if (CaselessEq<std::string>()(child->type, "quantize")) {
                auto it = quantizationDetailsMap.find(child->name);
                const QuantizationDetails quantizationDetails = it->second;
                if ((clampLayer->min_value > quantizationDetails.inputLowValues[0]) ||
                    (clampLayer->max_value < quantizationDetails.inputHighValues[0])) {
                    // std::cout << "Quantizer::quantize: " << child->name << std::endl;
                    THROW_IE_EXCEPTION << "Quantizer layer '" << child->name << "' has unexpected input range";
                }
            }
        }
    }

    // Step #4: Quantize & remove Quantize on activations
    for (CNNLayerPtr quantizeLayer : sortedLayers) {
        if (CaselessEq<std::string>()(quantizeLayer->type, "quantize")) {
            if (quantizeLayer->insData.size() != 5) {
                THROW_IE_EXCEPTION << "Quantize layer '" << quantizeLayer->name <<
                    "' has unexpected inputs count (" << quantizeLayer->insData.size() << ")";
            }

            if (quantizeLayer->outData.size() != 1) {
                THROW_IE_EXCEPTION << "Quantize layer '" << quantizeLayer->name <<
                    "' has unexpected outputs count (" << quantizeLayer->outData.size() << ")";
            }

            if (quantizeLayer->outData[0]->getInputTo().empty()) {
#ifndef NDEBUG
                std::cout << "Quantize layer '" << quantizeLayer->name <<
                    "' is skipped, no input layers for first output data (" << quantizeLayer->outData.size() << ")";
#endif
                continue;
            }

            // TODO: reserach only, implemented for INT8 only
            // Quantize => 1) Convolution 2) Eltwise
            // move to some another place later
            // isClampLikeQuantizer()

            const auto children = quantizeLayer->outData[0]->getInputTo();

            if (!CNNNetworkHelper::all(children, "Convolution")) {
                const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(quantizeLayer);
                if (!isClampLikeQuantizer(quantizationDetails)) {
                    THROW_IE_EXCEPTION << "unsupported children";
                }

                for (auto childIt = children.begin(); childIt != children.end(); ++childIt) {
                    CNNLayerPtr childLayer = childIt->second;
                    if (CaselessEq<std::string>()(childLayer->type, "eltwise")) {
                        // CNNLayerPtr parentLayer = quantizeLayer->insData[0].lock()->getCreatorLayer().lock();
                        updateExistingOrInsertClampBetween(net, quantizeLayer, false, childLayer, quantizationDetails);
                    }
                }
            }

            bool quantizableLayerWasFound = false;
            for (auto childIt = children.begin(); childIt != children.end(); ++childIt) {
                if (CaselessEq<std::string>()(childIt->second->type, "convolution") ||
                    CaselessEq<std::string>()(childIt->second->type, "fullyconnected")) {
                    quantizableLayerWasFound = true;
                    CNNLayerPtr quantizableLayer = childIt->second;

                    quantizeConvolutionOrFullyConnected(
                        net,
                        quantizeLayer,
                        quantizableLayer,
                        precision,
                        quantizationDetailsMap,
                        transformQuantizeOnDataPath);
                }
            }

            if (!quantizableLayerWasFound) {
                const CNNLayerPtr quantizableLayer = quantizeLayer->outData[0]->getInputTo().begin()->second;

                auto it = quantizationDetailsMap.find(quantizableLayer->name);
                if (it != quantizationDetailsMap.end()) {
                    updateExistingOrInsertClampBetween(net, quantizeLayer, true, quantizableLayer, it->second);
                }

                if (transformQuantizeOnDataPath) {
                    CNNNetworkHelper::removeQuantize(net, quantizeLayer);
                }
            }
        }
    }

    if (precision == QuantizationPrecision::INT8) {
        definesExecutionPrecision(net, quantizationDetailsMap);
        propagateScaleFactors(net);
        CNNNetworkHelper::addScaleShifts(net, quantizationDetailsMap);
    }

    // we should align layer properties and values in params map
    sortedLayers = CNNNetSortTopologically(net);
    for (CNNLayerPtr layer : sortedLayers) {
        InferenceEngine::details::NetworkSerializer::updateStdLayerParams(layer);
    }

    if (!dotFilePath.empty()) {
        std::ofstream file(dotFilePath);
        saveGraphToDot(net, file, precisionColoring2);
    }
}

void Quantizer::transformQuantizeOnWeightsPath(ICNNNetwork& network, const std::string& irFilePath) {
    std::vector<CNNLayerPtr> sortedLayers = CNNNetSortTopologically(network);
    for (CNNLayerPtr layer : sortedLayers) {
        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
        if ((weightableLayer != nullptr) &&
            (CaselessEq<std::string>()(weightableLayer->type, "convolution") ||
            CaselessEq<std::string>()(weightableLayer->type, "fullyconnected")) &&
            (weightableLayer->insData.size() > 1)) {
            transformQuantizeOnWeightsPath(network, layer);
        }
    }

    if (!irFilePath.empty()) {
        ConstTransformerDumper::serialize(irFilePath, network);
    }
}

void Quantizer::transformQuantizeOnWeightsPath(ICNNNetwork& network, CNNLayerPtr layer) {
    details::CNNNetworkImpl* networkImpl = dynamic_cast<details::CNNNetworkImpl*>(&network);
    if (networkImpl == nullptr) {
        THROW_IE_EXCEPTION << "Unexpected network type";
    }

    WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
    if ((weightableLayer != nullptr) &&
        (CaselessEq<std::string>()(weightableLayer->type, "convolution") ||
        CaselessEq<std::string>()(weightableLayer->type, "fullyconnected")) &&
        (weightableLayer->insData.size() > 1)) {
        if (weightableLayer->insData.size() > 3) {
            THROW_IE_EXCEPTION << "Unexpected inputs count for " << weightableLayer->name;
        }

        InferenceEngine::Blob::Ptr weightsBlob;
        const CNNLayerPtr weightsLayer = weightableLayer->insData[1].lock()->getCreatorLayer().lock();
        if (!weightsLayer) {
            THROW_IE_EXCEPTION << "Weights layer absent for layer " << weightableLayer->name;
        }

        if (weightsLayer->type == "Quantize") {
            checkQuantizeOnWeights(weightsLayer);

            QUANTIZATION_ENABLE_DUMP(ConstTransformerDumper::dumpOriginalBlob(
                weightsLayer->name,
                weightsLayer->insData[0].lock()->getCreatorLayer().lock()->blobs.begin()->second));
            weightsBlob = quantizeWeights(weightsLayer);
            QUANTIZATION_ENABLE_DUMP(ConstTransformerDumper::dumpQuantizedBlob(weightsLayer->name, weightsBlob));

            for (const auto constBlobDataWeakPtr : weightsLayer->insData) {
                const auto constBlobDataPtr = constBlobDataWeakPtr.lock();
                if (constBlobDataPtr == nullptr) {
                    THROW_IE_EXCEPTION << "Data for const layer is absent for convolution layer '" << weightableLayer->name << "'";
                }

                auto const constBlobLayer = constBlobDataPtr->getCreatorLayer().lock();
                if (constBlobLayer == nullptr) {
                    THROW_IE_EXCEPTION << "Const layer is absent for convolution layer '" << weightableLayer->name << "'";
                }

                networkImpl->removeData(constBlobLayer->name);
                networkImpl->removeLayer(constBlobLayer->name);
            }

            networkImpl->removeData(weightsLayer->name);
            networkImpl->removeLayer(weightsLayer->name);

        } else if (weightsLayer->type == "Const") {
            checkConstWithBlobs(weightsLayer);
            weightsBlob = weightsLayer->blobs.begin()->second;
            networkImpl->removeData(weightsLayer->name);
            networkImpl->removeLayer(weightsLayer->name);
        } else {
            THROW_IE_EXCEPTION << "Unexpected weightable layer '" << weightableLayer->name <<
                "' ('" << weightableLayer->type << "') input '" << weightsLayer->name << "' ('" <<
                weightsLayer->type << "')";
        }

        weightableLayer->_weights = weightsBlob;
        weightableLayer->blobs["weights"] = weightsBlob;

        if (weightableLayer->insData.size() > 2) {
            const CNNLayerPtr biasesLayer = weightableLayer->insData[2].lock()->getCreatorLayer().lock();
            if (!biasesLayer) {
                THROW_IE_EXCEPTION << "Biases layer absent for layer " << weightableLayer->name;
            }
            checkConstWithBlobs(biasesLayer);

            weightableLayer->_biases = biasesLayer->blobs.begin()->second;
            weightableLayer->blobs["biases"] = biasesLayer->blobs.begin()->second;
            networkImpl->removeData(biasesLayer->name);
            networkImpl->removeLayer(biasesLayer->name);
        }

        weightableLayer->insData.erase(weightableLayer->insData.begin() + 1, weightableLayer->insData.end());
    }
}

CNNLayer::Ptr getLatestInFuse(
    const CNNLayer::Ptr layer,
    const std::map<std::string, const QuantizationDetails>& quantizationDetailsMap,
    const bool transformQuantizeOnDataPath) {
    if (layer->outData.size() != 1) {
        return CNNLayer::Ptr();
    }

    if (layer->outData[0]->getInputTo().size() != 1) {
        return CNNLayer::Ptr();
    }

    CNNLayer::Ptr nextLayer = layer;
    do {
        nextLayer = nextLayer->outData[0]->getInputTo().begin()->second;
        if (nextLayer->outData[0]->getInputTo().size() != 1) {
            return CNNLayer::Ptr();
        }

        if ((!transformQuantizeOnDataPath) && CaselessEq<std::string>()(nextLayer->type, "quantize")) {
            return CNNLayer::Ptr();
        }
    } while (CaselessEq<std::string>()(nextLayer->type, "quantize"));

    if (quantizationDetailsMap.find(nextLayer->name) != quantizationDetailsMap.end()) {
        return nextLayer;
    }

    if (CaselessEq<std::string>()(nextLayer->type, "relu")) {
        if (nextLayer->outData[0]->getInputTo().size() != 1) {
            return CNNLayer::Ptr();
        }

        nextLayer = CNNNetworkHelper::getNextNot(nextLayer, "quantize");
        if (quantizationDetailsMap.find(nextLayer->name) != quantizationDetailsMap.end()) {
            return nextLayer;
        }
    }

    return CNNLayer::Ptr();
}

void Quantizer::quantizeConvolutionOrFullyConnected(
    ICNNNetwork& network,
    CNNLayerPtr& quantize,
    CNNLayerPtr& layer,
    QuantizationPrecision mode,
    std::map<std::string, const QuantizationDetails>& quantizationDetailsMap,
    const bool transformQuantizeOnDataPath) {
    if ((mode == QuantizationPrecision::FP32) || (mode == QuantizationPrecision::INT8)) {
        if (layer->insData.size() < 1) {
            THROW_IE_EXCEPTION << "No inputs for layer '" << layer->name << "'";
        }

        const DataPtr data = layer->insData[0].lock();
        if (data == nullptr) {
            THROW_IE_EXCEPTION << "No input data for layer '" << layer->name << "'";
        }

        if (data->getTensorDesc().getLayout() != Layout::NCHW) {
            THROW_IE_EXCEPTION << "Unexpected layout for layer '" << layer->name << "'";
        }

        const auto quantizationDetails = QuantizationDetails::getDetails(quantize);
        if (quantizationDetails.levels != 256) {
            THROW_IE_EXCEPTION << "Unexpected 'levels' param value " << quantizationDetails.levels <<
                " for Quantize layer '" << quantize->name << "'";
        }

        // Quantize on data path can have one scalar only
        if ((quantizationDetails.inputChannels() != 1) || (quantizationDetails.outputChannels() != 1)) {
            THROW_IE_EXCEPTION << "Unexpected Quantize values size for Quantize layer '" << quantize->name;
        }

        if (quantizationDetails.inputLowValues[0] > quantizationDetails.inputHighValues[0]) {
            std::cout << "Quantizer::quantizeConvolutionOrFullyConnected: inputLowValue VS inputHighValue for layer: " << quantize->name << std::endl;
        }

        if (mode == QuantizationPrecision::FP32) {
            if ((quantize->insData.size() != 1) && (quantize->insData.size() != 5)) {
                THROW_IE_EXCEPTION << "Not implemented: enexpected inputs size " << quantize->outData.size() << " for layer '" << layer->name << "'";
            }

            updateExistingOrInsertClampBetween(network, quantize, true, layer, quantizationDetails);
        }

        if (transformQuantizeOnDataPath) {
            CNNNetworkHelper::removeQuantize(network, quantize);

            const size_t valuesSize = quantizationDetails.inputLowValues.size() > quantizationDetails.outputLowValues.size() ?
                quantizationDetails.inputLowValues.size() :
                quantizationDetails.outputLowValues.size();

            const bool isInputValuesBroadcasted = quantizationDetails.inputLowValues.size() < valuesSize;
            const bool isOutputValuesBroadcasted = quantizationDetails.outputLowValues.size() < valuesSize;

            std::vector<float> weightsQuantizationCoeff(valuesSize);
            for (size_t i = 0; i < weightsQuantizationCoeff.size(); ++i) {
                const size_t outIndex = isOutputValuesBroadcasted ? 0 : i;
                const size_t inIndex = isInputValuesBroadcasted ? 0 : i;
                weightsQuantizationCoeff[i] =
                    (quantizationDetails.outputHighValues[outIndex] - quantizationDetails.outputLowValues[outIndex]) /
                    (quantizationDetails.inputHighValues[inIndex] - quantizationDetails.inputLowValues[inIndex]);
            }

            const size_t inputChannels = data->getTensorDesc().getDims()[1];
            const size_t outputChannels = layer->outData[0]->getTensorDesc().getDims()[1];

            Blob::Ptr fp32Weights = nullptr;
            if (layer->blobs.find("weights") != layer->blobs.end()) {
                fp32Weights = layer->blobs["weights"];
                WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
                if (weightableLayer == nullptr) {
                    THROW_IE_EXCEPTION << "Convolution '" << layer->name << "' is not weightable";
                }
                weightableLayer->_weights = layer->blobs["weights"];

                float* fp32WeightsBuffer = fp32Weights->buffer().as<float*>();

                WeightableLayer *pConv = dynamic_cast<WeightableLayer *>(layer.get());
                ConvolutionLayer *pConv1 = dynamic_cast<ConvolutionLayer *>(layer.get());

                if (pConv1 != nullptr && pConv1->_group == 0) {
                    THROW_IE_EXCEPTION << "Convolution '" << layer->name << "' has wrong groups number == 0";
                }
                int group = 1;
                if (pConv1 != nullptr && pConv1->_group != 1) {
                    group = pConv1->_group;
                }

                std::vector<float> newWeights;  // "new" weights are weights multiplied by i-scale

                const size_t W_CO = outputChannels / group;
                const size_t W_CI = inputChannels / group;
                const size_t W_HW = fp32Weights->size() / W_CI / W_CO / group;

                {
                    if ((weightsQuantizationCoeff.size() != 1) && (weightsQuantizationCoeff.size() != (group * W_CI))) {
                        THROW_IE_EXCEPTION << "Unexpected weights quantization values count '" <<
                            weightsQuantizationCoeff.size() << " for layer " << quantize->name;
                    }

                    // const bool isQuantizeWeightsCoeffBroadcasted = (weightsQuantizationCoeff.size() == (group * W_CI + W_CI)) ? false : true;
                    const bool isQuantizeWeightsCoeffBroadcasted = (weightsQuantizationCoeff.size() == (group * W_CI)) ? false : true;
                    if (weightsQuantizationCoeff.size() != 1) {
                        std::cout << "quantizeWeightsCoeff.size() != 1" << std::endl;
                    }

                    for (size_t g = 0; g < group; g++) {
                        for (size_t co = 0; co < W_CO; co++) {
                            for (size_t ci = 0; ci < W_CI; ci++) {
                                size_t kernelBase = g * W_CO * W_CI * W_HW + co * W_CI * W_HW + ci * W_HW;
                                for (size_t hw = 0; hw < W_HW; hw++) {
                                    const size_t index = isQuantizeWeightsCoeffBroadcasted ? 0 : ci;
                                    // newWeights.push_back(weight[kernelBase + hw] * weightsQuantizationCoeff[index]);
                                    fp32WeightsBuffer[kernelBase + hw] = fp32WeightsBuffer[kernelBase + hw] * weightsQuantizationCoeff[index];
                                }
                            }
                        }
                    }
                }
                size_t outChannelSize = fp32Weights->getTensorDesc().getDims().back() / W_CO / group;
            }

            const float biasesQuantizationCoeff = -(
                ((quantizationDetails.outputHighValues[0] - quantizationDetails.outputLowValues[0]) * quantizationDetails.inputLowValues[0]) /
                (quantizationDetails.inputHighValues[0] - quantizationDetails.inputLowValues[0])) + quantizationDetails.outputLowValues[0];

            if (layer->blobs.find("biases") != layer->blobs.end()) {
                Blob::Ptr fp32Biases = layer->blobs["biases"];
                WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
                if (weightableLayer == nullptr) {
                    THROW_IE_EXCEPTION << "Convolution '" << layer->name << "' is not weightable";
                }
                weightableLayer->_biases = layer->blobs["biases"];

                float* fp32BiasesBuffer = fp32Biases->buffer().as<float*>();
                for (size_t i = 0; i < fp32Biases->byteSize(); ++i) {
                    // TODO: have to extend: W - weights:
                    // fp32BiasesBuffer[i] = fp32BiasesBuffer[i] + biasesQuantizationCoeff * W;
                    fp32BiasesBuffer[i] = fp32BiasesBuffer[i] + biasesQuantizationCoeff;
                }
            }
        }

        auto layerOutputs = layer->outData[0]->getInputTo();

        if (mode == QuantizationPrecision::INT8) {
            if (layerOutputs.size() == 1) {
                size_t inputChannels = layer->insData[0].lock()->getTensorDesc().getDims()[1];
                size_t outputChannels = layer->outData[0]->getTensorDesc().getDims()[1];

                auto previousLayer = layer->insData[0].lock()->getCreatorLayer().lock();
                std::string inputLayerName = previousLayer->name;

                Blob::Ptr iScale = calculateScale(quantizationDetails, inputChannels);
                layer->blobs["i-scale"] = iScale;

                Blob::Ptr fp32Weights = nullptr;
                Blob::Ptr fp32Biases = nullptr;

                Blob::Ptr int8weights = nullptr;
                Blob::Ptr int32biases = nullptr;

                WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layer.get());
                if (weightableLayer == nullptr) {
                    THROW_IE_EXCEPTION << "Convolution '" << layer->name << "' is not weightable";
                }

                if (layer->blobs.find("weights") != layer->blobs.end()) {
                    fp32Weights = layer->blobs["weights"];
                    auto const dimentions = fp32Weights->getTensorDesc().getDims();
                    if ((dimentions.size() != 1) && (dimentions.size() != 4)) {
                        THROW_IE_EXCEPTION << "FP32 weights dimensions are not correct";
                    }

                    // Creating int8 weights blob
                    std::shared_ptr<Data> int8WeightsData = std::shared_ptr<Data>(new Data(
                        "weights",
                        TensorDesc(Precision::I8, fp32Weights->getTensorDesc().getDims(), fp32Weights->getTensorDesc().getLayout())));
                    int8weights = CreateBlobFromData(int8WeightsData);
                    int8weights->allocate();
                    layer->blobs["weights"] = int8weights;
                    weightableLayer->_weights = int8weights;
                }

                if (layer->blobs.find("biases") != layer->blobs.end()) {
                    fp32Biases = layer->blobs["biases"];

                    // Creating int8 biases blob
                    std::shared_ptr<Data> int32BiasesData = std::shared_ptr<Data>(new Data(
                        "biases",
                        TensorDesc(Precision::I32, fp32Biases->getTensorDesc().getDims(), fp32Biases->getTensorDesc().getLayout())));
                    int32biases = CreateBlobFromData(int32BiasesData);
                    int32biases->allocate();
                    layer->blobs["biases"] = int32biases;
                    weightableLayer->_biases = int32biases;
                }

                std::vector<float> weightScalers;


                // Creating w-scale blob
                if (fp32Weights) {
                    const float *weight = static_cast<const float *>(fp32Weights->buffer());

                    WeightableLayer *pConv = dynamic_cast<WeightableLayer *>(layer.get());
                    ConvolutionLayer *pConv1 = dynamic_cast<ConvolutionLayer *>(layer.get());

                    if (pConv1 != nullptr && pConv1->_group == 0) {
                        THROW_IE_EXCEPTION << "Convolution '" << layer->name << "'has wrong groups number == 0";
                    }
                    int group = 1;
                    if (pConv1 != nullptr && pConv1->_group != 1) {
                        group = pConv1->_group;
                    }


                    std::vector<float> newWeights;  // "new" weights are weights multiplied by i-scale

                    size_t W_CO = outputChannels / group,
                        W_CI = inputChannels / group,
                        W_HW = fp32Weights->size() / W_CI / W_CO / group;

                    {
                        float *iScaleMemory = static_cast<float *>(iScale->buffer());
                        for (size_t g = 0; g < group; g++) {
                            for (size_t co = 0; co < W_CO; co++) {
                                for (size_t ci = 0; ci < W_CI; ci++) {
                                    size_t kernelBase = g * W_CO * W_CI * W_HW + co * W_CI * W_HW + ci * W_HW;
                                    for (size_t hw = 0; hw < W_HW; hw++) {
                                        const size_t iScaleMemoryIndex = g * W_CI + ci;
                                        if (iScaleMemoryIndex >= iScale->size()) {
                                            THROW_IE_EXCEPTION << "incorrect i-sclae memory index";
                                        }
                                        newWeights.push_back(weight[kernelBase + hw] * iScaleMemory[g * W_CI + ci]);
                                    }
                                }
                            }
                        }
                    }
                    if (newWeights.empty()) {
                        THROW_IE_EXCEPTION << "Could not quantize layer '" << layer->name << "'. Invalid layer parameters.";
                    }
                    const size_t outChannelSize = int8weights->byteSize() / W_CO / group;
                    if (outChannelSize == 0) {
                        THROW_IE_EXCEPTION << "output channels size is not correct";
                    }

                    // Calculating weights normalization scale factor (w-scale)
                    float *weight_convolution;
                    size_t co;
                    for (co = 0, weight_convolution = &newWeights[0]; co < outputChannels; co++, weight_convolution += outChannelSize) {
                        float max = FLT_MIN;
                        DataStats::GetDataAbsMax(weight_convolution, outChannelSize, max);

                        float scaler = static_cast<float>(getMaxSignValue(quantizationDetails.levels)) / max;
                        weightScalers.push_back(scaler);
                    }

                    std::shared_ptr<Data> wScaleData = std::shared_ptr<Data>(new Data("w-scale", { Precision::FP32, { outputChannels }, Layout::C }));
                    auto wScale = CreateBlobFromData(wScaleData);
                    wScale->allocate();

                    float *wScaleMemory = static_cast<float *>(wScale->buffer());

                    for (size_t i = 0; i < outputChannels; i++) {
                        wScaleMemory[i] = 1.0 / weightScalers[i];
                    }
                    layer->blobs["w-scale"] = wScale;

                    // Normalizing the weights
                    ScaleDataToInt(&newWeights[0], fp32Weights->size(), int8weights, weightScalers);
                }

                const CNNLayerPtr latestInFuse = getLatestInFuse(layer, quantizationDetailsMap, transformQuantizeOnDataPath);
                Blob::Ptr oScale = latestInFuse ?
                    calculateScale(
                        quantizationDetailsMap.find(latestInFuse->name)->second,
                        layer->outData[0]->getTensorDesc().getDims()[1]) :
                    Blob::Ptr();

                if (oScale) {
                    layer->blobs["oi-scale"] = oScale;
                    layer->outData[0]->setPrecision(Precision::U8);
                } else {
                    layer->outData[0]->setPrecision(Precision::FP32);
                }

                // Normalizing the biases
                if (fp32Biases) {
                    const float *bias = static_cast<const float *>(fp32Biases->buffer());
                    ScaleDataToInt(bias, fp32Biases->size(), int32biases, weightScalers);
                }

                // execute Convolution/FullyConnected in INT8 after Quantize
                layer->precision = Precision::I8;
            } else {
                if (layer->insData.size() != 1) {
                    THROW_IE_EXCEPTION << "Layer '" << layer->name << "' has several parents, not implemented.";
                }
                CNNLayer::Ptr parent = layer->insData[0].lock()->getCreatorLayer().lock();
                updateExistingOrInsertClampBetween(network, parent, false, layer, quantizationDetails);
            }
        }
    }
}

int Quantizer::getMaxSignValue(const size_t quantizationLevels) {
    if (quantizationLevels == 256) {
        return 127;
    }
    THROW_IE_EXCEPTION << "Not supported quantization levels " << quantizationLevels;
}

int Quantizer::getMaxUnsignValue(const size_t quantizationLevels) {
    if (quantizationLevels == 256) {
        return 255;
    }
    THROW_IE_EXCEPTION << "Not supported quantization levels " << quantizationLevels;
}

Blob::Ptr Quantizer::quantizeWeights(const CNNLayerPtr quantize) {
    const CNNLayerPtr blobLayer = quantize->insData[0].lock()->getCreatorLayer().lock();
    InferenceEngine::Blob::Ptr sourceBlob = blobLayer->blobs.begin()->second;

    auto srcData = sourceBlob->buffer().as<float*>();
    const auto dims = quantize->outData[0]->getDims();
    if (dims.size() != 4) {
        THROW_IE_EXCEPTION << "Unexpected dimensions count " << dims.size() << " for layer '" << quantize->name << "'";
    }

    // OIHW
    size_t C = dims[0];  // O
    size_t N = dims[1];  // I
    size_t H = dims[2];  // H
    size_t W = dims[3];  // W

    InferenceEngine::Blob::Ptr targetBlob = sourceBlob;

    const size_t sourceBlobSize = sourceBlob->size();
    if (sourceBlobSize != (N * C * H * W)) {
        THROW_IE_EXCEPTION << "Unexpected weights dimention " << C << "x" << N << "x" << H << "x" << W << " for layer '" << quantize->name << "'";
    }

    auto dstData = targetBlob->buffer().as<float*>();

    const QuantizationDetails quantizationDetails = QuantizationDetails::getDetails(quantize);

    bool isInputLowBroadcasted = quantizationDetails.inputLowValues.size() != C;
    if ((quantizationDetails.inputLowValues.size() != 1) && (quantizationDetails.inputLowValues.size() != C)) {
        THROW_IE_EXCEPTION << "Unexpected input low values count " << quantizationDetails.inputLowValues.size() <<
            " for " << C << " channels, layer '" << quantize->name << "'";
    }

    bool isInputHighBroadcasted = quantizationDetails.inputHighValues.size() != C;
    if ((quantizationDetails.inputHighValues.size() != 1) && (quantizationDetails.inputHighValues.size() != C)) {
        THROW_IE_EXCEPTION << "Unexpected input high values count " << quantizationDetails.inputHighValues.size() <<
            " for " << C << " channels, layer '" << quantize->name << "'";
    }

    bool isOutputLowBroadcasted = quantizationDetails.outputLowValues.size() != C;
    if ((quantizationDetails.outputLowValues.size() != 1) && (quantizationDetails.outputLowValues.size() != C)) {
        THROW_IE_EXCEPTION << "Unexpected ouput low values count " << quantizationDetails.outputLowValues.size() <<
            " for " << C << " channels, layer '" << quantize->name << "'";
    }

    bool isOutputHighBroadcasted = quantizationDetails.outputHighValues.size() != C;
    if ((quantizationDetails.outputHighValues.size() != 1) && (quantizationDetails.outputHighValues.size() != C)) {
        THROW_IE_EXCEPTION << "Unexpected ouput high values count " << quantizationDetails.outputHighValues.size() <<
            " for " << C << " channels, layer '" << quantize->name << "'";
    }

    const auto levels = quantize->GetParamAsInt("levels");

    for (int c = 0; c < C; c++) {
        float inputLow = quantizationDetails.inputLowValues[isInputLowBroadcasted ? 0 : c];
        float inputHigh = quantizationDetails.inputHighValues[isInputHighBroadcasted ? 0 : c];
        float outputLow = quantizationDetails.outputLowValues[isOutputLowBroadcasted ? 0 : c];
        float outputHigh = quantizationDetails.outputHighValues[isOutputHighBroadcasted ? 0 : c];

        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int n = 0; n < N; n++) {
                    size_t idx = c * H * W * N + h * W * N + w * N + n;

                    if (srcData[idx] <= inputLow)
                        dstData[idx] = outputLow;
                    else if (srcData[idx] > inputHigh)
                        dstData[idx] = outputHigh;
                    else
                        dstData[idx] = roundf((srcData[idx] - inputLow) / (inputHigh - inputLow) * (levels - 1)) /
                        (levels - 1) * (outputHigh - outputLow) + outputLow;
                }
            }
        }
    }

    return targetBlob;
}

Blob::Ptr Quantizer::calculateScale(const QuantizationDetails& quantizationDetails, const size_t channels) {
    const int maxInt = quantizationDetails.hasNegativeOutput() ?
        getMaxSignValue(quantizationDetails.levels) :
        getMaxUnsignValue(quantizationDetails.levels);
    std::shared_ptr<Data> scaleData = std::shared_ptr<Data>(new Data("scale", { Precision::FP32, { channels }, Layout::C }));
    Blob::Ptr scale = CreateBlobFromData(scaleData);
    scale->allocate();
    float* scaleMemory = static_cast<float*>(scale->buffer());

    for (size_t channel = 0; channel < channels; ++channel) {
        scaleMemory[channel] = quantizationDetails.maxOutput(channel) / static_cast<float>(maxInt);
        if (fabs(scaleMemory[channel]) < 1e-7) {
            scaleMemory[channel] = 1.0f;
        }
    }

    return scale;
}

void Quantizer::ScaleDataToInt(
    const float* srcData,
    const size_t srcSize,
    Blob::Ptr int8blob,
    const std::vector<float>& scales) {
    if (scales.size() == 0 || /*srcblob->size()*/srcSize % scales.size() != 0) {
        THROW_IE_EXCEPTION << "Wrong number of scale factors";
    }

    size_t channels = scales.size();
    size_t channelSize = /*srcblob->size()*/srcSize / channels;

    const float* data = srcData;
    if (int8blob->getTensorDesc().getPrecision() == Precision::I8) {
        int8_t* int8data = static_cast<int8_t*>(int8blob->buffer());
        int minValue = std::numeric_limits<int8_t>::min();
        int maxValue = std::numeric_limits<int8_t>::max();

        size_t offset;

        float val;

        for (size_t ch = 0; ch < channels; ch++) {
            offset = channelSize * ch;

            for (size_t i = 0; i < channelSize; i++) {
                val = data[offset + i] * scales[ch];

                if (val > maxValue) {
                    val = maxValue;
                } else if (val < minValue) {
                    val = minValue;
                }

                int8data[offset + i] = std::round(val);
            }
        }
    } else if (int8blob->getTensorDesc().getPrecision() == Precision::I32) {
        int32_t* int32data = static_cast<int32_t*>(int8blob->buffer());
        int maxValue = std::numeric_limits<int32_t>::max();
        int minValue = std::numeric_limits<int32_t>::min();

        size_t offset;

        float val;

        for (size_t ch = 0; ch < channels; ch++) {
            offset = channelSize * ch;

            for (size_t i = 0; i < channelSize; i++) {
                val = data[offset + i] * scales[ch];

                if (val > maxValue) {
                    val = maxValue;
                } else if (val < minValue) {
                    val = minValue;
                }

                int32data[offset + i] = std::round(val);
            }
        }
    }
}
