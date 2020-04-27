// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "details/ie_cnn_network_tools.h"
#include <details/caseless.hpp>
#include "low_precision_transformations/network_helper.hpp"
#include "low_precision_transformations/layer_transformation.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

class LowPrecisionChainValidation {
public:
    class Chain : public std::unordered_set<std::string> {
    public:
        Chain(const Precision precision) : precision(precision) {}
        const Precision precision;
        bool exist(const std::vector<std::string> layerNames) {
            for (const std::string& layerName : layerNames) {
                if (find(layerName) == end()) {
                    return false;
                }
            }
            return true;
        }
    };

    using ChainsVector = std::vector<std::shared_ptr<Chain>>;

    static ChainsVector validate(
        const CNNNetwork& network,
        const CNNLayerPtr layer,
        const CNNLayerPtr endLayer) {
        std::unordered_map<std::string, Precision> precisionByPort;
        analyse(network, precisionByPort);

        std::unordered_map<std::string, std::shared_ptr<InternalChain>> handledLayers;

        InternalChainsMap chains;
        const std::shared_ptr<InternalChain> chain = std::make_shared<InternalChain>(handledLayers.size(), layer->outData[0]->getTensorDesc().getPrecision());
        chains.emplace(chain->id, chain);

        std::unordered_map<size_t, std::unordered_set<size_t>> hasToBeMerged;

        validate(
            layer,
            endLayer,
            precisionByPort,
            handledLayers,
            chains,
            chains[0],
            layer->outData[0]->getTensorDesc().getPrecision(),
            hasToBeMerged);

        auto it = hasToBeMerged.begin();
        while (it != hasToBeMerged.end()) {
            const size_t destinationChainId = it->first;
            const auto destinationChainIt = chains.find(destinationChainId);
            if (destinationChainIt == chains.end()) {
                THROW_IE_EXCEPTION << "chain with id was not found " << destinationChainId;
            }

            const std::shared_ptr<InternalChain> destinationChain = destinationChainIt->second;

            for (auto const sourceChainId : it->second) {
                const auto sourceChainIt = chains.find(sourceChainId);
                if (sourceChainIt == chains.end()) {
                    THROW_IE_EXCEPTION << "chain with id was not found " << sourceChainId;
                }

                std::shared_ptr<InternalChain> sourceChain = sourceChainIt->second;
                for (auto sourceIt = sourceChain->begin(); sourceIt != sourceChain->end(); ++sourceIt) {
                    destinationChain->emplace(*sourceIt);
                }

                chains.erase(sourceChainIt);
            }

            hasToBeMerged.erase(it);
            it = hasToBeMerged.begin();
        }

        ChainsVector resultChains;
        for (auto internalChainIt : chains) {
            auto internalChain = internalChainIt.second;
            std::shared_ptr<Chain> chain = std::make_shared<Chain>(internalChain->precision);
            resultChains.push_back(chain);
            for (auto layerNameIt = internalChain->begin(); layerNameIt != internalChain->end(); ++layerNameIt) {
                chain->insert(*layerNameIt);
            }
        }
        return resultChains;
    }

private:
    class InternalChain : public std::unordered_set<std::string> {
    public:
        InternalChain(const size_t id, const Precision precision) : id(id), precision(precision) {}
        const size_t id;
        const Precision precision;
    };

    using InternalChainsMap = std::map<size_t, std::shared_ptr<InternalChain>>;

    static void validate(
        const CNNLayerPtr layer,
        const CNNLayerPtr endLayer,
        const std::unordered_map<std::string, Precision>& precisionByPort,
        std::unordered_map<std::string, std::shared_ptr<InternalChain>>& handledLayers,
        InternalChainsMap& chains,
        std::shared_ptr<InternalChain> chain,
        const Precision chainPrecision,
        std::unordered_map<std::size_t, std::unordered_set<size_t>>& hasToBeMerged) {
        const auto handledLayerIt = handledLayers.find(layer->name);
        if (handledLayerIt != handledLayers.end())
        {
            if (chain->precision == handledLayerIt->second->precision) {
                const auto it = hasToBeMerged.find(handledLayerIt->second->id);
                std::unordered_set<size_t>& fused = it == hasToBeMerged.end() ?
                    hasToBeMerged.emplace(handledLayerIt->second->id, std::unordered_set<size_t>()).first->second :
                    it->second;
                fused.insert(chain->id);
            }
            return;
        }

        handledLayers.emplace(layer->name, chain);

        chain->insert(layer->name);

        if ((endLayer != nullptr) && (layer->name == endLayer->name)) {
            return;
        }

        for (size_t outDataIndex = 0; outDataIndex < layer->outData.size(); ++outDataIndex) {
            DataPtr outData = layer->outData[outDataIndex];
            const std::map<std::string, CNNLayerPtr> inputTo = outData->getInputTo();
            const Precision parentOutPrecision = getDataPrecision(precisionByPort, *layer, outDataIndex);

            for (auto it = inputTo.begin(); it != inputTo.end(); it++) {
                const CNNLayerPtr child = it->second;

                for (size_t childOutDataIndex = 0ul; childOutDataIndex < child->outData.size(); ++childOutDataIndex) {
                    const Precision childOutPrecision = getDataPrecision(precisionByPort, *child, childOutDataIndex);
                    if (parentOutPrecision == childOutPrecision) {
                        validate(child, endLayer, precisionByPort, handledLayers, chains, chain, chainPrecision, hasToBeMerged);
                    } else {
                        std::shared_ptr<InternalChain> childChain = std::make_shared<InternalChain>(handledLayers.size(), childOutPrecision);
                        chains.emplace(childChain->id, childChain);
                        validate(child, endLayer, precisionByPort, handledLayers, chains, childChain, childOutPrecision, hasToBeMerged);
                    }
                }
            }
        }
    }

    static void analyse(const CNNNetwork& network, std::unordered_map<std::string, Precision>& precisionByPort) {
        std::unordered_set<std::string> handledLayers;

        const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);
        for (const CNNLayerPtr layer : layers) {
            if (handledLayers.find(layer->name) != handledLayers.end()) {
                continue;
            }

            if (analyseAsymmetricQuantizationPattern(*layer, precisionByPort, handledLayers) != Precision::UNSPECIFIED) {
                continue;
            }

            if (analyseSymmetricQuantizationPattern(*layer, precisionByPort, handledLayers) != Precision::UNSPECIFIED) {
                continue;
            }

            fillPrecisionByPort(*layer, Precision::UNSPECIFIED, precisionByPort);
            handledLayers.emplace(layer->name);
        }
    }

    static void fillPrecisionByPort(
        const CNNLayer& layer,
        const Precision precision,
        std::unordered_map<std::string, Precision>& precisionByPort) {
        for (size_t outDataIndex = 0; outDataIndex < layer.outData.size(); ++outDataIndex) {
            DataPtr outData = layer.outData[outDataIndex];
            const std::string outDataId = getDataId(layer, outDataIndex);
            if (precisionByPort.find(outDataId) != precisionByPort.end()) {
                continue;
            }

            precisionByPort.emplace(outDataId, precision == Precision::UNSPECIFIED ? outData->getTensorDesc().getPrecision() : precision);
        }
    }

    static std::string getDataId(const CNNLayer& layer, const size_t dataIndex) {
        return layer.name + ".outputPort" + std::to_string(dataIndex);
    }

    static Precision getDataPrecision(const std::unordered_map<std::string, Precision>& precisionByPort, const CNNLayer& layer, const size_t dataIndex) {
        const auto precisionIt = precisionByPort.find(getDataId(layer, dataIndex));
        if (precisionIt == precisionByPort.end()) {
            THROW_IE_EXCEPTION <<
                "Precision for data '" << getDataId(layer, dataIndex) <<
                "' was not found for layer " << layer.type << " " << layer.name;
        }
        return precisionIt->second;
    }

    static Precision analyseAsymmetricQuantizationPattern(
        const CNNLayer& layer,
        std::unordered_map<std::string, Precision>& precisionByPort,
        std::unordered_set<std::string>& handledLayers) {
        if (!CaselessEq<std::string>()(layer.type, "Eltwise")) {
            return Precision::UNSPECIFIED;
        }

        const std::vector<CNNLayerPtr> parents = CNNNetworkHelper::getParents(layer);
        if ((parents.size() != 2ul) ||
            (!CaselessEq<std::string>()(parents[0]->type, "FakeQuantize")) ||
            (!CaselessEq<std::string>()(parents[1]->type, "Const")) ||
            CNNNetworkHelper::getParents(*parents[1]).size() != 0) {
            return Precision::UNSPECIFIED;
        }

        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
        if ((children.size() != 1ul) || (!CaselessEq<std::string>()(children[0]->type, "Convolution"))) {
            return Precision::UNSPECIFIED;
        }

        const std::vector<CNNLayerPtr> convolutionChildren = CNNNetworkHelper::getChildren(*children[0]);
        if ((convolutionChildren.size() != 1ul) || (!CaselessEq<std::string>()(convolutionChildren[0]->type, "FakeQuantize"))) {
            return Precision::UNSPECIFIED;
        }

        const Precision precisionBefore = CNNNetworkHelper::getPrecisionParent(layer);
        const Precision precisionAfterFakeQuantize = convolutionChildren[0]->outData[0]->getTensorDesc().getPrecision();
        const Precision precision = (precisionBefore == precisionAfterFakeQuantize) ? precisionAfterFakeQuantize : layer.outData[0]->getTensorDesc().getPrecision();

        fillPrecisionByPort(layer, precision, precisionByPort);
        handledLayers.emplace(layer.name);
        handledLayers.emplace(children[0]->name);

        return precision;
    }

    static Precision analyseSymmetricQuantizationPattern(
        const CNNLayer& layer,
        std::unordered_map<std::string, Precision>& precisionByPort,
        std::unordered_set<std::string>& handledLayers) {
        if ((!CaselessEq<std::string>()(layer.type, "Convolution")) &&
            (!CaselessEq<std::string>()(layer.type, "FullyConnected")) &&
            (!CaselessEq<std::string>()(layer.type, "GEMM"))) {
            return Precision::UNSPECIFIED;
        }

        const std::vector<CNNLayerPtr> children = CNNNetworkHelper::getChildren(layer);
        if ((children.size() != 1ul) || (!CaselessEq<std::string>()(children[0]->type, "FakeQuantize"))) {
            return Precision::UNSPECIFIED;
        }

        const Precision precisionBefore = CNNNetworkHelper::getPrecisionParent(layer, 0ul);
        const Precision precisionAfterFakeQuantize = children[0]->outData[0]->getTensorDesc().getPrecision();
        const Precision precision = (precisionBefore == precisionAfterFakeQuantize) ? precisionAfterFakeQuantize : layer.outData[0]->getTensorDesc().getPrecision();

        // TODO: convolution weights and biases layers are skipped
        fillPrecisionByPort(layer, precision, precisionByPort);
        handledLayers.emplace(layer.name);

        return precision;
    }
};

class LowPrecisionTransformationValidation {
public:
    static void validate(
            InferenceEngine::CNNNetwork& network,
            // TODO: not correct, quantization parameters are defined per transformation
            const InferenceEngine::details::LayerTransformation::Params& params,
            const std::unordered_set<std::string>& notTransformedLayers = {},
            const std::vector<std::pair<std::string, std::string>>& originalLayersInfo = {});

    static std::vector<std::pair<std::string, std::string>> getLayers(const InferenceEngine::CNNNetwork& network);

    static void validateIntervalsAndLevel(
            const InferenceEngine::CNNNetwork& network,
            const InferenceEngine::details::LayerTransformation::Params& params,
            const std::unordered_set<std::string>& notTransformedLayers);

    static void validateWeightsToConst(
            const InferenceEngine::CNNNetwork& network,
            const InferenceEngine::details::LayerTransformation::Params& params,
            const std::unordered_set<std::string>& notTransformedLayers);

    // TODO: refactor (I8/U8 is used)
    static void validatePrecision(
            const InferenceEngine::CNNNetwork& network,
            const InferenceEngine::details::LayerTransformation::Params& params,
            const std::unordered_set<std::string>& notTransformedLayers);

    static void validateActivations(
        const InferenceEngine::CNNNetwork& network,
        const InferenceEngine::details::LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers);

    static void validateScaleShifts(
        const InferenceEngine::CNNNetwork& network,
        const InferenceEngine::details::LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers);

    static void validateConvolutions(
        const InferenceEngine::CNNNetwork& network,
        const InferenceEngine::details::LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers);

    static void validateWithReference(
        InferenceEngine::CNNNetwork& network,
        const std::vector<std::pair<std::string, std::string>>& originalLayersInfo);

    static void validateCustomLayerHandling(
        const InferenceEngine::CNNNetwork& network,
        const std::unordered_set<std::string>& notTransformedLayers);

private:
    static InferenceEngine::details::DataPrecision getDataPrecision(
        const InferenceEngine::CNNLayer& layer,
        const InferenceEngine::details::LayerTransformation::Params& params);

    // TODO: quantizedTensorAlignmentOnActivations is used
    static void validateFakeQuantize(
        const InferenceEngine::CNNLayerPtr& layer,
        const InferenceEngine::details::LayerTransformation::Params& params,
        const bool multiBranch);

    static bool isFakeQuantizeBeforeEltwiseOnConvolutionBranch(const InferenceEngine::CNNLayer& fakeQuantize);

    static bool isFakeQuantizeBeforeConcat(const InferenceEngine::CNNLayer& fakeQuantize);

    static inline bool equals(const float value1, const float value2, const float max_diff = 0.0001f);

    static void validateEltwise(
        InferenceEngine::CNNNetwork& network,
        const InferenceEngine::details::LayerTransformation::Params& params,
        const InferenceEngine::CNNLayer& eltwise);

    static void validateAsymmetricPattern(
        const InferenceEngine::CNNNetwork& network,
        const InferenceEngine::details::LayerTransformation::Params& params,
        const std::unordered_set<std::string>& notTransformedLayers);

    static void validateAsymmetricPattern(const InferenceEngine::CNNLayer& layer, const InferenceEngine::details::LayerTransformation::Params& params);

    static void validateAsymmetricPatternEltwise(const InferenceEngine::CNNLayer& eltwise, const InferenceEngine::details::LayerTransformation::Params& params);

    static void validateEmptyConst(const InferenceEngine::CNNLayer& layer, const InferenceEngine::details::LayerTransformation::Params& params);
};
