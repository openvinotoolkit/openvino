// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "graph_transformer.h"

#include <vector>
#include <string>
#include <iterator>
#include <map>
#include <memory>

#include <cpp/ie_cnn_network.h>
#include <details/ie_cnn_network_tools.h>
#include <details/caseless.hpp>
#include "network_serializer.h"
#include "cnn_network_impl.hpp"
#include "blob_factory.hpp"
#include "graph_tools.hpp"
#include <shape_infer/const_infer/ie_const_infer_holder.hpp>

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace InferenceEngine {

void checkConstWithBlobs(const CNNLayerPtr& layer) {
    if (layer == nullptr) {
        THROW_IE_EXCEPTION << "Invalid argument: layer is nullable";
    }
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

ConstTransformer::ConstTransformer(details::CNNNetworkImpl* _network) {
    if (!_network)
        THROW_IE_EXCEPTION << "[ERROR]: Failed to init ConstTransformer with null pointer of network";

    network = _network;
}

std::vector<std::string>
ConstTransformer::foldConstSubgraphsInternal(const std::map<std::string, bool>& constLayers, const BlobMap& constData,
                                               const std::vector<CNNLayerPtr>& sortedLayers) {
    std::vector<std::string> remainingConstLayers;
    for (const auto& layer : sortedLayers) {
        if (constLayers.find(layer->name) != constLayers.end()) {
            // const layer doesn't need parent connections -> erase them
            for (const auto& insData : layer->insData) {
                auto& inputTo = insData.lock()->getInputTo();
                inputTo.erase(layer->name);
                // Note: to resolve corner case above layers can be marked as const with const data, just to be removed properly..
                // and maybe this logic wouldn't be needed
                if (inputTo.empty()) {
                    auto creator = insData.lock()->getCreatorLayer().lock();
                    auto it = std::find(creator->outData.begin(), creator->outData.end(), insData.lock());
                    if (it != creator->outData.end()) {
                        network->removeData((*it)->getName());
                        creator->outData.erase(it);
                    }
                }
            }
            layer->insData.clear();

            if (constLayers.at(layer->name)) {
                for (const auto& outData : layer->outData) {
                    for (const auto& inputTo : outData->getInputTo()) {
                        CNNLayerPtr inputToLayer;
                        std::string inputToName;
                        std::tie(inputToName, inputToLayer) = inputTo;
                        auto& insData = inputToLayer->insData;
                        auto insDataIt = std::find_if(insData.begin(), insData.end(),
                                                      [&outData](const DataWeakPtr& current) {
                                                          return current.lock()->getName() == outData->getName();
                                                      });
                        // remove connection with const data, because for const child it's not needed, for dynamic - new one will be created
                        if (insDataIt != insData.end()) {
                            insDataIt = inputToLayer->insData.erase(insDataIt);
                        }
                    }
                    network->removeData(outData->getName());
                }
                network->removeLayer(layer->name);
            } else {
                // if only one output data is not const - do nothing, otherwise - run procedure below
                // note: multiple const output data requires multiple layers with blob["custom"] to keep const data
                bool keepConstData = layer->outData.size() == 1;
                if (keepConstData) {
                    auto outData = layer->outData[0];
                    for (const auto& inputTo : outData->getInputTo()) {
                        if (constLayers.find(inputTo.first) != constLayers.end()) {
                            keepConstData = false;
                        }
                    }
                }
                if (keepConstData) {
                    if (!constLayers.at(layer->name)) {
                        auto outData = layer->outData[0];
                        if (layer->blobs.find("custom") == layer->blobs.end()) {
                            // if there's no const data - set it
                            const auto it = constData.find(outData->getName());
                            if (it != constData.end()) {
                                layer->blobs["custom"] = it->second;
                            }
                        }
                        if (layer->type != "Const") {
                            // layer was calculated during the Const Propagation, need to hide its semantic (type, params)
                            LayerParams layerParams{layer->name + "__" + outData->getName() + "__Const", "Const",
                                                    layer->precision};
                            auto newLayer = std::make_shared<CNNLayer>(layerParams);
                            for (const auto& data : layer->outData) {
                                data->getCreatorLayer() = newLayer;
                            }
                            newLayer->outData = layer->outData;
                            newLayer->blobs["custom"] = layer->blobs["custom"];
                            network->removeLayer(layer->name);
                            network->addLayer(newLayer);
                            remainingConstLayers.push_back(newLayer->name);
                        } else {
                            // Layer with `Const` type should be also considered on trimming shape inputs
                            remainingConstLayers.push_back(layer->name);
                        }
                    }
                } else {
                    for (const auto& outData : layer->outData) {
                        for (const auto& inputTo : outData->getInputTo()) {
                            CNNLayerPtr inputToLayer;
                            std::string inputToName;
                            std::tie(inputToName, inputToLayer) = inputTo;
                            auto& insData = inputToLayer->insData;
                            auto insDataIt = std::find_if(insData.begin(), insData.end(),
                                                          [&outData](const DataWeakPtr& current) {
                                                              return current.lock()->getName() == outData->getName();
                                                          });
                            // remove connection with const data, because for const child it's not needed, for dynamic - new one will be created
                            if (insDataIt != insData.end()) {
                                insDataIt = inputToLayer->insData.erase(insDataIt);
                            }
                            if (constLayers.find(inputToName) == constLayers.end()) {
                                // next layer is not const, need to attach const data to it via blobs["custom"] of new Const layer
                                LayerParams layerParams{layer->name + "__" + outData->getName() + "__Const", "Const",
                                                        layer->precision};
                                auto newLayer = std::make_shared<CNNLayer>(layerParams);
                                remainingConstLayers.push_back(newLayer->name);
                                const auto it = constData.find(outData->getName());
                                if (it != constData.end()) {
                                    newLayer->blobs["custom"] = it->second;
                                }
                                auto newData = std::make_shared<Data>(outData->getName() + "__" + inputToName,
                                                                      outData->getTensorDesc());
                                newData->getCreatorLayer() = newLayer;
                                newData->getInputTo()[inputToName] = inputToLayer;
                                newLayer->outData = {newData};
                                network->addLayer(newLayer);
                                network->getData(newData->getName()) = newData;
                                inputToLayer->insData.insert(insDataIt, newData);
                            }
                        }
                    }
                    for (const auto& data : layer->outData) {
                        network->removeData(data->getName());
                    }
                    network->removeLayer(layer->name);
                }
            }
        }
    }
    return remainingConstLayers;
}

const std::map<std::string, bool> ConstTransformer::getConstLayers(const std::vector<CNNLayerPtr>& sortedLayers) {
    std::map<std::string, bool> mapConstLayers;
    // collect all const layers, which inputs are const layers.
    for (const auto& layer : sortedLayers) {
        // Layers with "Shape" and "Const" type are Const by definition
        if (layer->type == "Shape" || layer->type == "Const") {
            mapConstLayers[layer->name] = false;
        } else if (layer->type != "Quantize") {
            bool isAllInputsConst = true;
            for (auto const& data : layer->insData) {
                auto creatorName = data.lock()->getCreatorLayer().lock()->name;
                if (mapConstLayers.find(creatorName) == mapConstLayers.end()) {
                    isAllInputsConst = false;
                }
            }
            if (isAllInputsConst && !layer->insData.empty()) mapConstLayers[layer->name] = false;
        }
    }
    // Add mark for const layers, if it's used for shape taking layers as second input
    // true - is used and can be deleted from graph, as no influence on data, false - opposite
    std::map<std::string, bool> mapVisitedLayers = mapConstLayers;
    for (auto rit = sortedLayers.rbegin(); rit != sortedLayers.rend(); rit++) {
        auto currentLayer = (*rit);
        std::string currentLayerName = currentLayer->name;
        bool isCurrentConst = mapConstLayers.find(currentLayerName) != mapConstLayers.end();
        for (int i = 0; i < currentLayer->insData.size(); i++) {
            std::string creatorName;
            if (currentLayer->insData[i].lock()) {
                auto creator = currentLayer->insData[i].lock()->getCreatorLayer().lock();
                if (creator) {
                    creatorName = creator->name;
                }
            }
            bool isCreatorConst = mapConstLayers.find(creatorName) != mapConstLayers.end();
            if (isCreatorConst) {
                // mark second const input of shape taking layers (Reshape, Interp..), if they wasn't visited before
                if ((i == 1) && (shapeTaking.find(currentLayer->type)) != shapeTaking.end()) {
                    if (!mapConstLayers[creatorName]) {
                        if (!mapVisitedLayers.at(creatorName)) {
                            mapConstLayers[creatorName] = true;
                        }
                    }
                } else {
                    if (isCurrentConst) {
                        if (mapConstLayers.at(currentLayerName)) {
                            if (!mapConstLayers[creatorName]) {
                                if (!mapVisitedLayers.at(creatorName)) {
                                    mapConstLayers[creatorName] = true;
                                }
                            }
                        } else {
                            mapConstLayers[creatorName] = false;
                        }
                    } else {
                        mapConstLayers[creatorName] = false;
                    }
                }
            }
            mapVisitedLayers[creatorName] = true;
        }
        mapVisitedLayers[currentLayerName] = true;
    }
    return mapConstLayers;
}

const BlobMap ConstTransformer::getConstData(const std::map<std::string, bool>& constLayers, const std::vector<CNNLayerPtr>& sortedLayers) {
    ShapeInfer::ConstInferHolder holder;
    BlobMap constData;
    auto getInputBlobs = [&constData](const std::vector<DataWeakPtr>& insData,
                                      bool isForShape) -> std::vector<Blob::CPtr> {
        std::vector<Blob::CPtr> inputBlobs;
        // special case of Const layers: no inputs, no input blobs
        if (insData.empty()) {
            return {};
        }
        for (const auto& data : insData) {
            std::string dataName = data.lock()->getName();
            if (constData.find(dataName) != constData.end()) {
                // get blobs, inferred before
                inputBlobs.push_back(constData.at(dataName));
            } else {
                // special case of Shape layer: no input data, but blob contains info about dimensions, layout and etc...
                auto blob = make_blob_with_precision(data.lock()->getTensorDesc());
                inputBlobs.push_back(blob);
            }
        }
        return inputBlobs;
    };

    auto getOutputBlobs = [](const std::vector<DataPtr>& outData) -> std::vector<Blob::Ptr> {
        std::vector<Blob::Ptr> outputBlobs;
        for (const auto& data : outData) {
            auto blob = make_blob_with_precision(data->getTensorDesc());
            blob->allocate();
            outputBlobs.push_back(blob);
        }
        return outputBlobs;
    };

    for (const auto& layer : sortedLayers) {
        if (constLayers.find(layer->name) != constLayers.end()) {
            std::string layerName = layer->name;
            bool isForShape = constLayers.at(layerName);
            CNNLayerPtr layer;
            ResponseDesc resp;
            IE_ASSERT(StatusCode::OK == network->getLayerByName(layerName.c_str(), layer, &resp));

            auto implPtr = holder.getConstInferImpl(layer->type);
            if (!implPtr && !isForShape)
                THROW_IE_EXCEPTION << "Failed to find reference implementation for `"
                                      + layer->name + "` Layer with `" + layer->type + "` Type on constant propagation";
            if (!isForShape) {
                auto outputBlobs = getOutputBlobs(layer->outData);
                implPtr->infer(getInputBlobs(layer->insData, isForShape), layer->params, layer->blobs, outputBlobs);
                for (int i = 0; i < layer->outData.size(); i++) {
                    std::string dataName = layer->outData[i]->getName();
                    auto shapes = layer->outData[i]->getTensorDesc().getDims();
                    outputBlobs[i]->getTensorDesc().reshape(shapes, TensorDesc::getLayoutByDims(shapes));
                    constData[dataName] = outputBlobs[i];
                }
            }
        }
    }
    return constData;
}

void ConstTransformer::trimShapeInputs(const std::vector<std::string>& constLayers) {
    for (const auto& layerName : constLayers) {
        CNNLayerPtr layer;
        ResponseDesc resp;
        IE_ASSERT(StatusCode::OK == network->getLayerByName(layerName.c_str(), layer, &resp));

        if (layer->outData.size() == 1 && layer->type == "Const" && layer->insData.empty()) {
            auto constData = layer->outData[0];
            std::map<std::string, CNNLayerPtr> inputToMap = constData->getInputTo();
            for (const auto& inputTo : inputToMap) {
                CNNLayerPtr inputToLayer = inputTo.second;
                if (shapeTaking.find(inputToLayer->type) != shapeTaking.end()) {
                    auto& insData = inputToLayer->insData;
                    auto it = std::find_if(insData.begin(), insData.end(),
                                           [&constData](const DataWeakPtr& current) {
                                               return current.lock()->getName() == constData->getName();
                                           });
                    if (it != insData.end() && std::distance(insData.begin(), it) == 1) {
                        inputToLayer->insData.erase(it);
                        constData->getInputTo().erase(inputTo.first);
                    }
                }
            }
            if (constData->getInputTo().empty()) {
                network->removeData(constData->getName());
                network->removeLayer(layer->name);
            }
        }
    }
}

void ConstTransformer::foldConstSubgraphs() {
    auto sortedLayers = details::CNNNetSortTopologically(*network);
    auto constLayers = getConstLayers(sortedLayers);
    auto constData = getConstData(constLayers, sortedLayers);
    foldConstSubgraphsInternal(constLayers, constData, sortedLayers);
}

void ConstTransformer::fullTrim() {
    auto sortedLayers = details::CNNNetSortTopologically(*network);
    auto constMapLayers = getConstLayers(sortedLayers);
    auto constData = getConstData(constMapLayers, sortedLayers);
    auto constLayers = foldConstSubgraphsInternal(constMapLayers, constData, sortedLayers);
    trimShapeInputs(constLayers);
}

void ConstTransformer::moveWeights() {
    for (const auto& layerIt : network->allLayers()) {
        WeightableLayer* weightableLayer = dynamic_cast<WeightableLayer*>(layerIt.second.get());
        if ((weightableLayer != nullptr) &&
            (CaselessEq<std::string>()(weightableLayer->type, "Convolution") || CaselessEq<std::string>()(weightableLayer->type, "FullyConnected")) &&
            (weightableLayer->insData.size() > 1)) {
            if (weightableLayer->insData.size() > 3) {
                THROW_IE_EXCEPTION << "Unexpected inputs count for " << weightableLayer->name;
            }

            const DataPtr insData = weightableLayer->insData[1].lock();
            if (!insData) {
                THROW_IE_EXCEPTION << "Weights input is absent for layer " << weightableLayer->name;
            }

            InferenceEngine::Blob::Ptr weightsBlob;
            const CNNLayerPtr weightsLayer = insData->getCreatorLayer().lock();
            if (!weightsLayer) {
                THROW_IE_EXCEPTION << "Weights layer absent for layer " << weightableLayer->name;
            }

            bool removePathOnWeights = false;
            if (CaselessEq<std::string>()(weightsLayer->type, "Const")) {
                checkConstWithBlobs(weightsLayer);

                weightsBlob = weightsLayer->blobs.begin()->second;
                network->removeData(weightsLayer->name);
                network->removeLayer(weightsLayer->name);

                weightableLayer->_weights = weightsBlob;
                weightableLayer->blobs["weights"] = weightsBlob;
                removePathOnWeights = true;
            }

            bool removePathOnBiases = false;
            if (weightableLayer->insData.size() > 2) {
                const DataPtr insData = weightableLayer->insData[2].lock();
                if (!insData) {
                    THROW_IE_EXCEPTION << "Biases input is absent for layer " << weightableLayer->name;
                }

                const CNNLayerPtr biasesLayer = insData->getCreatorLayer().lock();
                if (!biasesLayer) {
                    THROW_IE_EXCEPTION << "Biases layer absent for layer " << weightableLayer->name;
                }

                checkConstWithBlobs(biasesLayer);

                weightableLayer->_biases = biasesLayer->blobs.begin()->second;
                weightableLayer->blobs["biases"] = weightableLayer->_biases;
                network->removeData(biasesLayer->name);
                network->removeLayer(biasesLayer->name);
                removePathOnBiases = true;
            }

            if (removePathOnWeights && removePathOnBiases) {
                weightableLayer->insData.erase(weightableLayer->insData.begin() + 1, weightableLayer->insData.end());
            } else if (removePathOnWeights) {
                weightableLayer->insData.erase(weightableLayer->insData.begin() + 1, weightableLayer->insData.begin() + 2);
            } else if (removePathOnBiases) {
                weightableLayer->insData.erase(weightableLayer->insData.begin() + 2, weightableLayer->insData.end());
            }
        }
    }
}
}  // namespace InferenceEngine
