// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <builders/ie_network_builder.hpp>
#include <builders/ie_const_layer.hpp>
#include <builders/ie_input_layer.hpp>

#include "graph_tools.hpp"

#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <memory>
#include <vector>
#include <string>
#include <limits>
#include <map>

#include <shape_infer/ie_reshaper.hpp>
#include "blob_factory.hpp"
#include <details/caseless.hpp>

#include "ie_cnn_layer_builder.h"
#include "ie_profiling.hpp"

using namespace InferenceEngine;

/******************************************************************************
 Network builder
 ******************************************************************************/
Builder::Network::Network(const std::string &name): Builder::Network(Context(), name) {}
Builder::Network::Network(const INetwork &network): Builder::Network(Context(), network) {}
Builder::Network::Network(const ICNNNetwork &network): Builder::Network(Context(), network) {}

Builder::Network::Network(const Context& ieContext, const std::string &name) {
    parameters["name"] = name;
    parameters["context"] = ieContext;
    parameters["version"] = 3;
    parameters["layers"] = std::vector<Layer::Ptr>();
    parameters["connections"] = std::vector<Connection>();
}

Builder::Network::Network(const Context& ieContext, const INetwork &network): Network(ieContext, network.getName()) {
    for (const auto& layer : network) {
        parameters["layers"].as<std::vector<Layer::Ptr>>().push_back(std::make_shared<Layer>(layer));
        const auto layerConnections = network.getLayerConnections(layer->getId());
        for (const auto& connection : layerConnections) {
            bool found = false;
            for (const auto& con : parameters["connections"].as<std::vector<Connection>>()) {
                if (con == connection) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                parameters["connections"].as<std::vector<Connection>>().push_back(connection);
            }
        }
    }
}

Builder::Network::Network(const Context& ieContext, const ICNNNetwork &network): Network(ieContext, network.getName()) {
    parameters["version"] = 0;
    auto allInputs = CNNNetGetAllInputLayers(network);
    InputsDataMap inputs;
    network.getInputsInfo(inputs);
    if (inputs.empty() && allInputs.empty())
        THROW_IE_EXCEPTION << "Cannot create graph! No inputs for the topology " << network.getName();

    std::unordered_map<std::string, idx_t> name2id;
    std::unordered_set<Data*> dataPtrs;
    std::vector<CNNLayerPtr> queueLayers;

    auto createGenericFromCNNLayer = [&](const CNNLayerPtr& cnnLayer) {
        for (const auto& data : cnnLayer->insData) {
            auto lockedData = data.lock();
            if (!lockedData)
                continue;
            if (dataPtrs.find(lockedData.get()) == dataPtrs.end()) {
                dataPtrs.insert(lockedData.get());
            }
        }
        for (const auto& data : cnnLayer->outData) {
            if (dataPtrs.find(data.get()) == dataPtrs.end()) {
                dataPtrs.insert(data.get());
            }
        }
        std::map<std::string, Blob::Ptr> blobs = cnnLayer->blobs;
        size_t inputsCount(0);
        for (const auto& data : cnnLayer->insData) {
            auto lockedData = data.lock();
            if (!lockedData)
                continue;
            inputsCount++;
        }
        const auto layer = builderFromCNNLayer(cnnLayer);
        idx_t layerId = addLayer(layer);

        if (blobs.find("weights") != blobs.end()) {
            idx_t constLayerId = addLayer(ConstLayer("weights").setData(blobs["weights"]));
            connect({constLayerId}, {layerId, inputsCount++});
        }
        if (blobs.find("biases") != blobs.end()) {
            if (blobs.find("weights") == blobs.end()) ++inputsCount;

            idx_t constLayerId = addLayer(ConstLayer("biases").setData(blobs["biases"]));
            connect({constLayerId}, {layerId, inputsCount++});
        }
        for (const auto& it : blobs) {
            if (it.first == "weights" || it.first == "biases")
                continue;
            idx_t constLayerId = addLayer(ConstLayer(it.first).setData(it.second));
            connect({constLayerId}, {layerId, inputsCount++});
        }
        name2id[layer.getName()] = layerId;
        return layerId;
    };

    auto addPreProcessFor = [&](const InputInfo::Ptr& inputInfo) {
        auto inputLayer = getLayer(name2id[inputInfo->name()]);
        if (inputLayer->getType().empty() && inputLayer->getName().empty())
            return;

        inputLayer->getParameters()["preProcess"] = inputInfo->getPreProcess();
    };

    for (auto input : inputs) {
        auto inputLayer = input.second->getInputData()->getCreatorLayer().lock();

        if (dataPtrs.find(input.second->getInputData().get()) == dataPtrs.end()) {
            dataPtrs.insert(input.second->getInputData().get());
        }

        if (!inputLayer) {
            // For v1 parser
            inputLayer.reset(new CNNLayer({input.second->getInputData()->getName(),
                                           "Input",
                                           input.second->getInputData()->getPrecision()}));

            inputLayer->outData.push_back(input.second->getInputData());
        }
        const auto layer = InputLayer(inputLayer->name).setPort(Port(inputLayer->outData[0]->getTensorDesc().getDims()));
        name2id[layer.getName()] = addLayer(layer);

        for (const auto &nlayer : input.second->getInputData()->getInputTo()) {
            queueLayers.push_back(nlayer.second);
        }
    }
    for (auto input : allInputs) {
        auto isRealInput = std::find_if(std::begin(inputs), std::end(inputs),
                                        [&](InputsDataMap::value_type &inputInfo) {
                                            return inputInfo.second->getInputData()->getName() == input->name;
                                        });
        if (isRealInput != std::end(inputs)) {
            continue;
        }

        details::CaselessEq<std::string> eq;
        CNNLayerPtr cnnLayer = input;

        if (eq(input->type, "Memory")) {
            auto memoryId = input->GetParamAsString("id");
            cnnLayer.reset(new CNNLayer({input->name + "/id=" + memoryId, "MemoryInput", input->precision}));
            cnnLayer->params = input->params;
            cnnLayer->outData = input->outData;
        }

        createGenericFromCNNLayer(cnnLayer);

        size_t count_out = 0;
        for (auto &&outData : input->outData) {
            for (auto &&nlayer : outData->getInputTo()) {
                queueLayers.push_back(nlayer.second);
            }
            count_out++;
        }
    }
    while (!queueLayers.empty()) {
        auto cnnLayerPtr = *queueLayers.begin();

        if (name2id.find(cnnLayerPtr->name) == name2id.end()) {
            createGenericFromCNNLayer(cnnLayerPtr);

            for (auto &&outData : cnnLayerPtr->outData) {
                for (auto &&nlayer : outData->getInputTo()) {
                    queueLayers.push_back(nlayer.second);
                }
            }
        }

        queueLayers.erase(queueLayers.begin());
    }
    std::map<std::string, DataPtr> output;
    network.getOutputsInfo(output);

    for (auto it = output.begin(); it != output.end(); it++) {
        CNNLayerPtr creator = (*it).second->getCreatorLayer().lock();
        if (name2id.find(creator->name) == name2id.end())
            THROW_IE_EXCEPTION << "Cannot find output layer " << creator->name;

        auto lastLayer = getLayer(name2id[creator->name]);
        if (lastLayer->getName() == "" && lastLayer->getType().empty())
            THROW_IE_EXCEPTION << "Cannot find output layer " << creator->name;

        std::string name = "out_" + lastLayer->getName();

        CNNLayerPtr cnnOutLayer(new CNNLayer({name, "Output", creator->outData[0]->getPrecision()}));
        cnnOutLayer->insData.push_back((*it).second);

        idx_t outLayerId = createGenericFromCNNLayer(cnnOutLayer);

        idx_t inIdx(0);
        for (size_t i = 0; i < creator->outData.size(); i++) {
            if (creator->outData[i] == (*it).second) {
                inIdx = i;
                break;
            }
        }

        parameters["connections"].as<std::vector<Connection>>().push_back(Connection({lastLayer->getId(), inIdx}, {outLayerId}));
    }

    for (const auto dataPtr : dataPtrs) {
        auto cnnInputLayer = dataPtr->getCreatorLayer().lock();
        idx_t inIdx(0);
        if (!cnnInputLayer) {
            // For v1 parser
            cnnInputLayer.reset(new CNNLayer({dataPtr->getName(),
                                              "Input",
                                              dataPtr->getPrecision()}));
        } else {
            for (size_t i = 0; i < cnnInputLayer->outData.size(); i++) {
                if (cnnInputLayer->outData[i].get() == dataPtr) {
                    inIdx = i;
                    break;
                }
            }
        }
        for (const auto& it : dataPtr->getInputTo()) {
            if (name2id.find(cnnInputLayer->name) == name2id.end() || name2id.find(it.second->name) == name2id.end())
                THROW_IE_EXCEPTION << "Cannot create connections between nodes: " << cnnInputLayer->name << " -> " << it.second->name;
            idx_t outIdx(0);

            for (size_t i = 0; i < it.second->insData.size(); i++) {
                const auto lockedData = it.second->insData[i].lock();
                if (lockedData && lockedData.get() == dataPtr) {
                    outIdx = i;
                    break;
                }
            }
            parameters["connections"].as<std::vector<Connection>>()
                .push_back(Connection({name2id[cnnInputLayer->name], inIdx}, {name2id[it.second->name], outIdx}));
        }
    }

    for (const auto &input : inputs) {
        addPreProcessFor(input.second);
    }
}

const std::vector<Builder::Layer::Ptr>& Builder::Network::getLayers() const {
    return parameters.at("layers").as<std::vector<Layer::Ptr>>();
}
std::vector<Builder::Layer::Ptr>& Builder::Network::getLayers() {
    return parameters["layers"].as<std::vector<Layer::Ptr>>();
}

idx_t Builder::Network::addLayer(const std::vector<PortInfo> &inputs,
                                 const Layer& layer) {
    IE_PROFILING_AUTO_SCOPE(Builder::Network::addLayer)
    auto layer_id = addLayer(layer);
    for (size_t i = 0; i < inputs.size(); i++) {
        connect({inputs[i].layerId(), inputs[i].portId()}, {layer_id, i});
    }
    return layer_id;
}

idx_t Builder::Network::addLayer(const Layer& layer) {
    auto getAvailableId = [&](idx_t defaultId) {
        if (defaultId == (std::numeric_limits<idx_t>::max)())
            defaultId = 0;

        auto it = parameters["layers"].as<std::vector<Layer::Ptr>>().begin();
        while (it != parameters["layers"].as<std::vector<Layer::Ptr>>().end()) {
            for (it = parameters["layers"].as<std::vector<Layer::Ptr>>().begin();
                    it != parameters["layers"].as<std::vector<Layer::Ptr>>().end(); it++) {
                if ((*it)->getId() == defaultId) {
                    defaultId++;
                    break;
                }
            }
        }
        return defaultId;
    };
    auto generateAvailableName = [&](const std::string& name, idx_t id) {
        const std::string idName = "id" + std::to_string(id);
        std::string generatedName(name);
        if (generatedName.empty())
            generatedName = idName;
        bool nameIsUnique(false);
        while (!nameIsUnique) {
            nameIsUnique = true;
            for (const auto& layer : parameters["layers"].as<std::vector<Layer::Ptr>>()) {
                if (generatedName == layer->getName()) {
                    nameIsUnique = false;
                    generatedName += "_" + idName;
                }
            }
        }
        return generatedName;
    };
    idx_t generatedId = getAvailableId(layer.getId());
    const auto name = generateAvailableName(layer.getName(), generatedId);
    parameters["layers"].as<std::vector<Layer::Ptr>>().emplace_back(std::make_shared<Layer>(generatedId, layer));
    parameters["layers"].as<std::vector<Layer::Ptr>>()[parameters["layers"].as<std::vector<Layer::Ptr>>().size() - 1]->setName(name);
    return generatedId;
}

void Builder::Network::connect(const PortInfo& input, const PortInfo& output) {
    const auto mergePortData = [&]() -> bool {
        const auto blobEqualOrEmpty = [](const Blob::Ptr& ref, const Blob::Ptr& test) -> bool {
            return (ref->size() == test->size() || test->size() == 0) &&
                   (!memcmp(ref->cbuffer(), test->cbuffer(), test->byteSize())) &&
                   (ref->getTensorDesc().getPrecision() == test->getTensorDesc().getPrecision() ||
                    test->getTensorDesc().getPrecision() == Precision::UNSPECIFIED) &&
                   (ref->getTensorDesc().getLayout() == test->getTensorDesc().getLayout() ||
                    test->getTensorDesc().getLayout() == Layout::ANY) &&
                   (ref->getTensorDesc().getDims() == test->getTensorDesc().getDims() ||
                    test->getTensorDesc().getDims().empty()) &&
                   (ref->cbuffer().as<char *>() == test->cbuffer().as<char *>() ||
                    test->cbuffer() == nullptr);
        };

        const auto srcPortData = getLayer(input.layerId())->getOutputPorts()[input.portId()].getData();
        const auto dstPortData = getLayer(output.layerId())->getInputPorts()[output.portId()].getData();
        if (srcPortData == dstPortData)
            return true;

        if (srcPortData->getParameters() != dstPortData->getParameters() &&
                !srcPortData->getParameters().empty() &&
                !dstPortData->getParameters().empty())
            return false;

        size_t srcDataCount(0), dstDataCount(0);
        if (!srcPortData->getParameters().empty()) srcDataCount++;
        if (!dstPortData->getParameters().empty()) dstDataCount++;

        const auto srcBlb = srcPortData->getData();
        const auto dstBlb = dstPortData->getData();
        if (srcBlb == dstBlb || (srcBlb->size() == dstBlb->size() &&
                srcBlb->getTensorDesc() == dstBlb->getTensorDesc() &&
                ((srcBlb->cbuffer().as<char *>() == dstBlb->cbuffer().as<char *>()) ||
                    (srcBlb->cbuffer() != nullptr && dstBlb->cbuffer() != nullptr &&
                    !memcmp(srcBlb->cbuffer(), dstBlb->cbuffer(), dstBlb->byteSize()))))) {
            srcDataCount++;
            dstDataCount++;
        } else if (blobEqualOrEmpty(srcBlb, dstBlb)) {
            srcDataCount++;
        } else if (blobEqualOrEmpty(dstBlb, srcBlb)) {
            dstDataCount++;
        } else {
            return false;
        }

        if (dstDataCount > srcDataCount) {
            // Change source and all src destination data
            for (const auto& connection : getLayerConnections(input.layerId())) {
                if (connection.from() != input)
                    continue;
                getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()].setData(dstPortData);
            }
            getLayer(input.layerId())->getOutputPorts()[input.portId()].setData(dstPortData);
        } else {
            // Change destination data
            getLayer(output.layerId())->getInputPorts()[output.portId()].setData(srcPortData);
        }

        return true;
    };

    if (!mergePortData())
        THROW_IE_EXCEPTION << "Cannot connect two ports with different data!";

    parameters["connections"].as<std::vector<Connection>>().emplace_back(input, output);
}

void Builder::Network::removeLayer(idx_t layerId) {
    auto it = parameters["layers"].as<std::vector<Layer::Ptr>>().begin();
    for (; it != parameters["layers"].as<std::vector<Layer::Ptr>>().end(); it++) {
        if ((*it)->getId() == layerId) {
            break;
        }
    }
    if (it != parameters["layers"].as<std::vector<Layer::Ptr>>().end())
        parameters["layers"].as<std::vector<Layer::Ptr>>().erase(it);
}

void Builder::Network::disconnect(const Connection& connection) {
    auto it = parameters["connections"].as<std::vector<Connection>>().begin();
    for (; it != parameters["connections"].as<std::vector<Connection>>().end(); it++) {
        if (connection == *it)
            break;
    }
    if (it != parameters["connections"].as<std::vector<Connection>>().end())
        parameters["connections"].as<std::vector<Connection>>().erase(it);

    try {
        auto layer = getLayer(connection.to().layerId());
        layer->getInputPorts()[connection.to().portId()].setData(std::make_shared<PortData>());
    } catch (InferenceEngine::details::InferenceEngineException& ex) {}
}

const INetwork::CPtr Builder::Network::build() {
    validate();
    InferenceEngine::Builder::Network::Ptr network =
            std::make_shared<InferenceEngine::Builder::Network>(static_cast<const INetwork&>(*this));
    return network;
}

void Builder::Network::validate() {
    // Check that all ports are connected
    for (const auto& layer : getLayers()) {
        std::vector<bool> existInCon(layer->getInputPorts().size());
        for (size_t i = 0; i < layer->getInputPorts().size(); i++) {
            if (layer->getInputPorts()[i].getParameters().find("type") != layer->getInputPorts()[i].getParameters().end())
                existInCon[i] = true;
        }
        std::vector<bool> existOutCon(layer->getOutputPorts().size());

        const auto layerConnections = getLayerConnections(layer->getId());
        for (const auto& connection : layerConnections) {
            if (connection.from().layerId() == layer->getId()) {
                existOutCon[connection.from().portId()] = true;
                getLayer(connection.to().layerId());
            }
            if (connection.to().layerId() == layer->getId()) {
                existInCon[connection.to().portId()] = true;
                getLayer(connection.from().layerId());
            }
        }
        bool allPortsConnected = true;
        for (const auto& cons : {existInCon, existOutCon}) {
            for (const auto &existCon : cons) {
                allPortsConnected = allPortsConnected && existCon;
            }
        }
        if (!allPortsConnected)
            THROW_IE_EXCEPTION << "Not all ports of layer " << layer->getName() << " were connected!";
    }

    // Check all layers
    for (const auto& connection : getConnections()) {
        if (!getLayer(connection.to().layerId()))
            THROW_IE_EXCEPTION << "Cannot find layer with id: " << connection.to().layerId();
        if (!getLayer(connection.from().layerId()))
            THROW_IE_EXCEPTION << "Cannot find layer with id: " << connection.from().layerId();
    }

    std::map<std::string, SizeVector> inputShapes;
    for (const auto& input : getInputs())
        inputShapes[input->getName()] = input->getOutputPorts()[0].shape();

    ShapeInfer::Reshaper reshaper(this);
    ResponseDesc resp;
    StatusCode sts = reshaper.run(inputShapes, &resp);
    // Not all implementations may be registered if all shapes were read from IR.
    if (sts == NOT_FOUND) {
        bool allShapesLooksGood = true;
        for (const auto& connection : getConnections()) {
            if (getLayer(connection.from().layerId())->getOutputPorts()[connection.from().portId()].shape() !=
                getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()].shape() ||
                getLayer(connection.to().layerId())->getInputPorts()[connection.to().portId()].shape().empty()) {
                allShapesLooksGood = false;
                break;
            }
        }
        if (allShapesLooksGood)
            sts = OK;
    }

    if (sts != OK)
        THROW_IE_EXCEPTION << resp.msg;

    // Check all parameters
    for (const auto& layer : getLayers()) {
        try {
            layer->build();
        } catch(InferenceEngine::details::InferenceEngineException& ex) {
            THROW_IE_EXCEPTION << "Cannot build layer " << layer->getName() << ": " << ex.what();
        } catch(std::bad_cast& ex) {
            THROW_IE_EXCEPTION << "Cannot build layer " << layer->getName() << ": " << ex.what();
        }
    }
}

Builder::Network::operator const INetwork::CPtr() {
    return build();
}

const ILayer::CPtr Builder::Network::getLayer(idx_t layerId) const noexcept {
    try {
        for (auto& layer : getLayers()) {
            if (layer->getId() == layerId)
                return layer->build();
        }
    } catch(...) {}

    return nullptr;
}

Builder::Layer::Ptr Builder::Network::getLayer(idx_t layerId) {
    for (auto& layer : getLayers()) {
        if (layer->getId() == layerId)
            return layer;
    }
    THROW_IE_EXCEPTION << "Cannot find layer with id: " << layerId;
}

const std::string& Builder::Network::getName() const noexcept {
    static std::string errName;
    try {
        return parameters.at("name");
    } catch (...) {
        return errName;
    }
}

const Context& Builder::Network::getContext() const noexcept {
    static Context errCtx;
    try {
        return parameters.at("context");
    } catch (...) {
        return errCtx;
    }
}

Context& Builder::Network::getContext() noexcept {
    static Context errCtx;
    try {
        return parameters.at("context");
    } catch (...) {
        return errCtx;
    }
}

Builder::Network::const_iterator Builder::Network::begin() const noexcept {
    try {
        return Network::const_iterator(this);
    } catch (...) {
        return Network::const_iterator(this, true);
    }
}


Builder::Network::const_iterator Builder::Network::end() const noexcept {
    return Network::const_iterator(this, true);
}

size_t Builder::Network::size() const noexcept {
    return static_cast<size_t>(std::distance(std::begin(*this), std::end(*this)));
}

Builder::Network::iterator Builder::Network::begin() {
    return Network::iterator(this);
}

Builder::Network::iterator Builder::Network::end() {
    return Network::iterator(this, true);
}

const std::vector<ILayer::CPtr> Builder::Network::getInputs() const noexcept {
    std::vector<ILayer::CPtr> inputs;
    try {
        for (const auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
            bool isInputLayer = true;
            for (const auto& connection : getLayerConnections(layer->getId())) {
                if (connection.to().layerId() == layer->getId()) {
                    isInputLayer = false;
                    break;
                }
            }
            if (isInputLayer) {
                inputs.push_back(layer->build());
            }
        }
    } catch (...) {}
    return inputs;
}

std::vector<Builder::Layer::Ptr> Builder::Network::getInputs() {
    std::vector<Builder::Layer::Ptr> inputs;
    for (auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
        bool isInputLayer = true;
        for (const auto& connection : getLayerConnections(layer->getId())) {
            if (connection.to().layerId() == layer->getId()) {
                isInputLayer = false;
                break;
            }
        }
        if (isInputLayer) {
            inputs.push_back(layer);
        }
    }
    return inputs;
}

const std::vector<ILayer::CPtr> Builder::Network::getOutputs() const noexcept {
    std::vector<ILayer::CPtr> outputs;
    try {
        for (const auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
            bool isOutputLayer = true;
            for (const auto& connection : getLayerConnections(layer->getId())) {
                if (connection.from().layerId() == layer->getId()) {
                    isOutputLayer = false;
                    break;
                }
            }
            if (isOutputLayer) {
                outputs.push_back(layer->build());
            }
        }
    } catch (...) {}
    return outputs;
}

std::vector<Builder::Layer::Ptr> Builder::Network::getOutputs() {
    std::vector<Builder::Layer::Ptr> outputs;
    for (auto& layer : parameters.at("layers").as<std::vector<Layer::Ptr>>()) {
        bool isOutputLayer = true;
        for (const auto& connection : getLayerConnections(layer->getId())) {
            if (connection.from().layerId() == layer->getId()) {
                isOutputLayer = false;
                break;
            }
        }
        if (isOutputLayer) {
            outputs.push_back(layer);
        }
    }
    return outputs;
}

const std::vector<Connection>& Builder::Network::getConnections() const {
    return parameters.at("connections").as<std::vector<Connection>>();
}

const std::vector<Connection> Builder::Network::getLayerConnections(idx_t layerId) const noexcept {
    std::vector<Connection> layerConnections;
    try {
        for (const auto connection : parameters.at("connections").as<std::vector<Connection>>()) {
            if (connection.from().layerId() == layerId || connection.to().layerId() == layerId)
                layerConnections.push_back(connection);
        }
    } catch (...) {}
    return layerConnections;
}
