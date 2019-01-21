// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_builders.hpp>
#include <ie_network.hpp>
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
#include "ie_format_parser.h"
#include "ie_layer_parsers.h"
#include "blob_factory.hpp"
#include <details/caseless.hpp>

#include "ie_cnn_layer_builder.h"
#include "ie_memcpy.h"

using namespace InferenceEngine;

/******************************************************************************
 Network builder
 ******************************************************************************/
Builder::Network::Network(const std::string &name): Builder::Network(Context(), name) {}
Builder::Network::Network(const INetwork &network): Builder::Network(Context(), network) {}
Builder::Network::Network(const ICNNNetwork &network): Builder::Network(Context(), network) {}

Builder::Network::Network(const Context& ieContext, const std::string &name): ctx(ieContext), name(name), version(3) {}

Builder::Network::Network(const Context& ieContext, const INetwork &network): ctx(ieContext), name(network.getName()), version(3) {
    for (const auto& layer : network) {
        layers.push_back(Layer(layer));
        const auto layerConnections = network.getLayerConnections(layer->getId());
        for (const auto& connection : layerConnections) {
            bool found = false;
            for (const auto& con : connections) {
                if (con == connection) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                connections.push_back(connection);
            }
        }
    }
}

Builder::Network::Network(const Context& ieContext, const ICNNNetwork &network): ctx(ieContext), name(network.getName()), version(0) {
    auto allInputs = CNNNetGetAllInputLayers(network);
    InputsDataMap inputs;
    network.getInputsInfo(inputs);
    if (inputs.empty() && allInputs.empty())
        THROW_IE_EXCEPTION << "Cannot create graph! No inputs for the topology " << network.getName();

    std::unordered_map<std::string, idx_t> name2id;
    std::unordered_set<Data*> dataPtrs;
    std::vector<CNNLayerPtr> queueLayers;

    auto createGenericFromCNNLayer = [&](const CNNLayerPtr& cnnLayer) {
        std::vector<Port> inputPorts;
        for (const auto& data : cnnLayer->insData) {
            auto lockedData = data.lock();
            if (!lockedData)
                continue;
            if (dataPtrs.find(lockedData.get()) == dataPtrs.end()) {
                dataPtrs.insert(lockedData.get());
            }
            inputPorts.emplace_back(lockedData->getTensorDesc().getDims());
        }
        std::vector<Port> outputPorts;
        for (const auto& data : cnnLayer->outData) {
            if (dataPtrs.find(data.get()) == dataPtrs.end()) {
                dataPtrs.insert(data.get());
            }
            outputPorts.push_back(Port(data->getTensorDesc().getDims()));
        }

        std::map<std::string, Parameter> params;
        for (const auto& it : cnnLayer->params) {
            params[it.first] = it.second;
        }
        const auto layer = Layer(cnnLayer->type, cnnLayer->name)
                .setInputPorts(inputPorts).setOutputPorts(outputPorts)
                .setParameters(params).setConstantData(cnnLayer->blobs);
        idx_t layerId = addLayer(layer);
        name2id[layer.getName()] = layerId;
        return layerId;
    };

    auto addPreProcessFor = [&](const InputInfo::Ptr& inputInfo) {
        auto inputLayer = getLayer(name2id[inputInfo->name()]);
        if (inputLayer.getType().empty() && inputLayer.getName().empty())
            return;

        ResizeAlgorithm alg = inputInfo->getPreProcess().getResizeAlgorithm();
        std::string algStr;
        switch (alg) {
            case RESIZE_BILINEAR:
                algStr = "RESIZE_BILINEAR";
                break;
            case RESIZE_AREA:
                algStr = "RESIZE_AREA";
                break;
            default:
                break;
        }

        if (!algStr.empty())
            inputLayer.getParameters()["resize_alg"] = algStr;

        switch (inputInfo->getPreProcess().getMeanVariant()) {
            case MEAN_IMAGE: {
                auto meanWidth = inputInfo->getPreProcess()[0]->meanData->dims()[0];
                auto meanHeight = inputInfo->getPreProcess()[0]->meanData->dims()[1];

                TensorDesc desc(Precision::FP32, inputLayer.getOutputPorts()[0].shape(), Layout::NCHW);
                Blob::Ptr meanBuffer = make_blob_with_precision(desc);
                meanBuffer->allocate();
                auto *meanData = meanBuffer->buffer().as<float *>();
                for (unsigned channel = 0; channel < inputInfo->getPreProcess().getNumberOfChannels(); channel++) {
                    Blob::Ptr meanBlob = inputInfo->getPreProcess()[channel]->meanData;
                    if (!meanBlob || meanBlob->precision() != Precision::FP32)
                        THROW_IE_EXCEPTION << "mean image not provided or not in Float 32";
                    if (meanBlob->size() != meanHeight*meanWidth) {
                        THROW_IE_EXCEPTION << "mean image size does not match expected network input, expecting " << meanWidth << " x " << meanHeight;
                    }
                    ie_memcpy(meanData + channel*meanBlob->size(),
                            meanBuffer->byteSize() - channel*meanBlob->size() * sizeof(float),
                            meanBlob->buffer(),
                            meanBlob->byteSize());
                }

                // WA for batch != 1
                // Reshape for new batch is not supported for models with mean image
                size_t noBatchSize = desc.getBlockingDesc().getStrides()[0];
                for (size_t b = 1; b < inputLayer.getOutputPorts()[0].shape()[0]; b++) {
                    ie_memcpy(meanData + noBatchSize*b,
                              meanBuffer->byteSize() - noBatchSize * b * sizeof(float),
                              meanData,
                              noBatchSize * sizeof(float));
                }

                std::vector<PortInfo> outPorts;
                std::vector<Connection> inputConnections = getLayerConnections(inputLayer.getId());
                for (const auto& connection : inputConnections) {
                    outPorts.push_back(connection.to());
                    disconnect(connection);
                }

                idx_t constId = addLayer(Builder::ConstLayer(inputLayer.getName() + "_mean_image")
                                                 .setPort(inputLayer.getOutputPorts()[0]).setData(meanBuffer));
                idx_t constNegId = addLayer({{constId}}, Builder::PowerLayer(inputLayer.getName() + "_mean_image_neg")
                                                 .setPort(inputLayer.getOutputPorts()[0]).setScale(-1));

                idx_t eltwiseId = addLayer({{inputLayer.getId()}, {constNegId}},
                        Builder::EltwiseLayer(inputLayer.getName() + "_mean_image_elt")
                             .setInputPorts({inputLayer.getOutputPorts()[0], inputLayer.getOutputPorts()[0]})
                             .setOutputPort(inputLayer.getOutputPorts()[0])
                             .setEltwiseType(Builder::EltwiseLayer::EltwiseType::SUM));

                for (const auto& port : outPorts) {
                    connect({eltwiseId}, port);
                }
            }
                break;
            case MEAN_VALUE: {
                TensorDesc desc(Precision::FP32, {inputInfo->getPreProcess().getNumberOfChannels()}, Layout::C);
                Blob::Ptr mean = make_blob_with_precision(desc);
                mean->allocate();
                Blob::Ptr scale = make_blob_with_precision(desc);
                scale->allocate();
                Blob::Ptr emptyScale = make_blob_with_precision(desc);
                emptyScale->allocate();
                auto *meanData = mean->buffer().as<float *>();
                auto *scaleData = scale->buffer().as<float *>();
                auto *emptyScaleData = emptyScale->buffer().as<float *>();
                bool noMean = true;
                bool noScale = true;
                for (size_t i = 0; i < inputInfo->getPreProcess().getNumberOfChannels(); i++) {
                    meanData[i] = -inputInfo->getPreProcess()[i]->meanValue;
                    noMean = noMean && (meanData[i] == 0);
                    scaleData[i] = inputInfo->getPreProcess()[i]->stdScale;
                    emptyScaleData[i] = 1;
                    noScale = noScale && (scaleData[i] == 1);
                }
                std::vector<PortInfo> outPorts;
                std::vector<Connection> inputConnections = getLayerConnections(inputLayer.getId());
                for (const auto& connection : inputConnections) {
                    outPorts.push_back(connection.to());
                    disconnect(connection);
                }

                idx_t meanId = inputLayer.getId();
                if (!noMean) {
                    meanId = addLayer({{inputLayer.getId()}},
                                            Builder::ScaleShiftLayer(inputLayer.getName() + "_mean_value")
                                                    .setPort(inputLayer.getOutputPorts()[0])
                                                    .setBiases(mean).setWeights(emptyScale));
                }

                idx_t scaleId = meanId;
                if (!noScale) {
                    scaleId = addLayer({{meanId}},
                                             Builder::ScaleShiftLayer(inputLayer.getName() + "_scale_value")
                                                     .setPort(inputLayer.getOutputPorts()[0])
                                                     .setWeights(scale));
                }

                for (const auto& port : outPorts) {
                    connect({scaleId}, port);
                }
            }
                break;
            default:
                break;
        }
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
        if (lastLayer.getName() == "" && lastLayer.getType().empty())
            THROW_IE_EXCEPTION << "Cannot find output layer " << creator->name;

        std::string name = "out_" + lastLayer.getName();

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

        connections.push_back(Connection({lastLayer.getId(), inIdx}, {outLayerId}));
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
        for (const auto& it : dataPtr->inputTo) {
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
            connections.push_back(Connection({name2id[cnnInputLayer->name], inIdx}, {name2id[it.second->name], outIdx}));
        }
    }

    for (auto input : inputs) {
        addPreProcessFor(input.second);
    }
}

std::vector<Builder::Layer>& Builder::Network::getLayers() {
    return layers;
}

const std::vector<Builder::Layer>& Builder::Network::getLayers() const {
    return layers;
}

idx_t Builder::Network::addLayer(const std::vector<PortInfo> &inputs,
                                 const Layer& layer) {
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

        auto it = layers.begin();
        while (it != layers.end()) {
            for (it = layers.begin(); it != layers.end(); it++) {
                if (it->getId() == defaultId) {
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
            for (const auto& layer : layers) {
                if (generatedName == layer.getName()) {
                    nameIsUnique = false;
                    generatedName += "_" + idName;
                }
            }
        }
        return generatedName;
    };
    idx_t generatedId = getAvailableId(layer.getId());
    const auto name = generateAvailableName(layer.getName(), generatedId);
    layers.emplace_back(generatedId, layer);
    layers[layers.size() - 1].getName() = name;
    return generatedId;
}

void Builder::Network::connect(const PortInfo& input, const PortInfo& output) {
    connections.emplace_back(input, output);
}

void Builder::Network::removeLayer(idx_t layerId) {
    auto it = layers.begin();
    for (; it != layers.end(); it++) {
        if (it->getId() == layerId) {
            break;
        }
    }
    if (it != layers.end())
        layers.erase(it);
}

void Builder::Network::disconnect(const Connection& connection) {
    auto it = connections.begin();
    for (; it != connections.end(); it++) {
        if (connection == *it)
            break;
    }
    if (it != connections.end())
        connections.erase(it);
}

const INetwork::Ptr Builder::Network::build() const {
    // Check that all ports are connected
    for (const auto& layer : layers) {
        std::vector<bool> existInCon(layer.getInputPorts().size());
        std::vector<bool> existOutCon(layer.getOutputPorts().size());

        const auto layerConnections = getLayerConnections(layer.getId());
        for (const auto& connection : layerConnections) {
            if (connection.from().layerId() == layer.getId()) {
                existOutCon[connection.from().portId()] = true;
                getLayer(connection.to().layerId());
            }
            if (connection.to().layerId() == layer.getId()) {
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
            THROW_IE_EXCEPTION << "Not all ports of layer " << layer.getName() << " were connected!";
    }

    InferenceEngine::details::Network::Ptr network = std::make_shared<InferenceEngine::details::Network>(ctx, name);
    for (const auto& layer : layers) {
        network->addLayer(layer.build());
    }
    for (const auto& connection : connections) {
        network->addConnection(connection);
    }

    // Check that all ports are connected
    for (const auto& layer : *network) {
        std::vector<bool> existInCon(layer->getInputPorts().size());
        std::vector<bool> existOutCon(layer->getOutputPorts().size());

        const auto layerConnections = network->getLayerConnections(layer->getId());
        for (const auto& connection : layerConnections) {
            if (connection.from().layerId() == layer->getId()) {
                existOutCon[connection.from().portId()] = true;
            }
            if (connection.to().layerId() == layer->getId()) {
                existInCon[connection.to().portId()] = true;
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

    std::map<std::string, SizeVector> inputShapes;
    for (const auto& input : network->getInputs())
        inputShapes[input->getName()] = input->getOutputPorts()[0].shape();

    if (version) {
        details::BaseCreator::version_ = version;
    }

    ShapeInfer::Reshaper reshaper(ctx, network);
    ResponseDesc resp;
    StatusCode sts = reshaper.run(inputShapes, &resp);
    // Not all implementations may be registered if all shapes were read from IR.
    if (sts == NOT_FOUND) {
        bool allShapesLooksGood = true;
        for (const auto& connection : network->getConnections()) {
            if (network->getLayer(connection.from().layerId())->
                    getOutputPorts()[connection.from().portId()].shape() !=
                network->getLayer(connection.to().layerId())->
                        getInputPorts()[connection.to().portId()].shape()) {
                allShapesLooksGood = false;
                break;
            }
        }
        if (allShapesLooksGood)
            sts = OK;
    }

    if (sts != OK)
        THROW_IE_EXCEPTION << resp.msg;

    return std::static_pointer_cast<INetwork>(network);
}

const std::shared_ptr<ICNNNetwork> Builder::convertToICNNNetwork(const INetwork::Ptr& network) {
    std::unique_ptr<details::CNNNetworkImpl> cnnNetworkImpl(new details::CNNNetworkImpl());

    Precision detectedPrecision = Precision::FP32;
    for (const auto& layer : *network) {
        const auto& params = layer->getParameters();
        if (!params)
            continue;
        Precision prc = Precision::UNSPECIFIED;
        for (const auto& blobIterator : params->getConstantData()) {
            if (blobIterator.second) {
                prc = blobIterator.second->precision();
                break;
            }
        }
        if (prc != Precision::UNSPECIFIED) {
            detectedPrecision = prc;
            break;
        }
    }

    auto createCNNLayer = [](const std::shared_ptr<const ILayer>& layer, Precision precision) {
        static std::vector<std::shared_ptr<BaseConverter>> convertors = {
                std::make_shared<LayerConverter<InferenceEngine::PowerLayer>>("Power"),
                std::make_shared<LayerConverter<InferenceEngine::ConvolutionLayer>>("Convolution"),
                std::make_shared<LayerConverter<InferenceEngine::DeconvolutionLayer>>("Deconvolution"),
                std::make_shared<LayerConverter<InferenceEngine::PoolingLayer>>("Pooling"),
                std::make_shared<LayerConverter<InferenceEngine::FullyConnectedLayer>>("InnerProduct"),
                std::make_shared<LayerConverter<InferenceEngine::FullyConnectedLayer>>("FullyConnected"),
                std::make_shared<LayerConverter<InferenceEngine::NormLayer>>("LRN"),
                std::make_shared<LayerConverter<InferenceEngine::NormLayer>>("Norm"),
                std::make_shared<LayerConverter<InferenceEngine::SoftMaxLayer>>("Softmax"),
                std::make_shared<LayerConverter<InferenceEngine::GRNLayer>>("GRN"),
                std::make_shared<LayerConverter<InferenceEngine::MVNLayer>>("MVN"),
                std::make_shared<LayerConverter<InferenceEngine::ReLULayer>>("ReLU"),
                std::make_shared<LayerConverter<InferenceEngine::ClampLayer>>("Clamp"),
                std::make_shared<LayerConverter<InferenceEngine::SplitLayer>>("Split"),
                std::make_shared<LayerConverter<InferenceEngine::SplitLayer>>("Slice"),
                std::make_shared<LayerConverter<InferenceEngine::ConcatLayer>>("Concat"),
                std::make_shared<LayerConverter<InferenceEngine::EltwiseLayer>>("Eltwise"),
                std::make_shared<LayerConverter<InferenceEngine::ScaleShiftLayer>>("ScaleShift"),
                std::make_shared<LayerConverter<InferenceEngine::PReLULayer>>("PReLU"),
                std::make_shared<LayerConverter<InferenceEngine::CropLayer>>("Crop"),
                std::make_shared<LayerConverter<InferenceEngine::ReshapeLayer>>("Reshape"),
                std::make_shared<LayerConverter<InferenceEngine::ReshapeLayer>>("Flatten"),
                std::make_shared<LayerConverter<InferenceEngine::TileLayer>>("Tile"),
                std::make_shared<ActivationConverter>(),
                std::make_shared<LayerConverter<InferenceEngine::BatchNormalizationLayer>>("BatchNormalization"),
        };
        for (auto &convertor : convertors) {
            if (!convertor->canCreate(layer->getType()))
                continue;
            return convertor->createLayer(layer, precision);
        }
        static LayerConverter<CNNLayer> genericCreator("");
        return genericCreator.createLayer(layer, precision);
    };

    cnnNetworkImpl->setName(network->getName());
    cnnNetworkImpl->setPrecision(Precision::UNSPECIFIED);
    for (const auto& layer : *network) {
        if (details::CaselessEq<std::string>()(layer->getType(), "Output"))
            continue;
        CNNLayerPtr cnnLayer = createCNNLayer(layer, detectedPrecision);
        if (cnnNetworkImpl->getPrecision() == Precision::UNSPECIFIED) {
            cnnNetworkImpl->setPrecision(cnnLayer->precision);
        } else if (cnnNetworkImpl->getPrecision() == Precision::MIXED &&
                   cnnNetworkImpl->getPrecision() != cnnLayer->precision) {
            cnnNetworkImpl->setPrecision(Precision::MIXED);
        }

        auto connections = network->getLayerConnections(layer->getId());
        std::unordered_set<idx_t> inputNum, outputNum;
        for (const auto& connection : connections) {
            if (connection.from().layerId() != layer->getId())
                inputNum.insert(connection.to().portId());
            else
                outputNum.insert(connection.from().portId());
        }
        cnnLayer->insData.resize(inputNum.size());
        cnnLayer->outData.resize(outputNum.size());
        cnnNetworkImpl->addLayer(cnnLayer);
    }

    for (const auto& layer : *network) {
        auto connections = network->getLayerConnections(layer->getId());
        CNNLayerPtr cnnLayer;
        StatusCode sts = cnnNetworkImpl->getLayerByName(layer->getName().c_str(), cnnLayer, nullptr);
        details::CaselessEq<std::string> eq;
        if (sts != OK && eq(layer->getType(), "Output"))
            continue;
        else if (sts != OK)
            THROW_IE_EXCEPTION << "Cannot find CNNLayer by name " << layer->getName();

        for (const auto& connection : connections) {
            if (connection.from().layerId() != layer->getId())
                continue;

            const auto& outLayer = network->getLayer(connection.to().layerId());

            CNNLayerPtr cnnOutLayer;
            sts = cnnNetworkImpl->getLayerByName(outLayer->getName().c_str(), cnnOutLayer, nullptr);
            if (sts != OK && !eq(outLayer->getType(), "Output"))
                THROW_IE_EXCEPTION << "Cannot find CNNLayer by name " << outLayer->getName();

            std::string dataName = layer->getName();
            if (cnnLayer->outData.size() > 1) {
                dataName += "_" + std::to_string(connection.from().portId());
            }
            DataPtr& data = cnnNetworkImpl->getData(dataName);
            if (!data) {
                TensorDesc dataDesc(detectedPrecision, layer->getOutputPorts()[connection.from().portId()].shape(),
                                    TensorDesc::getLayoutByDims(layer->getOutputPorts()[connection.from().portId()].shape()));
                data = std::make_shared<Data>(layer->getName(), dataDesc);
                data->creatorLayer = cnnLayer;
            }
            cnnLayer->outData[connection.from().portId()] = data;
            if (cnnOutLayer) {
                data->inputTo[outLayer->getName()] = cnnOutLayer;
                cnnOutLayer->insData[connection.to().portId()] = data;
            } else {
                cnnNetworkImpl->addOutput(data->getName());
            }
        }

        cnnLayer->validateLayer();
        if (eq(cnnLayer->type, "Input")) {
            InputInfo::Ptr inputInfo(new InputInfo());
            inputInfo->setInputData(*cnnLayer->outData.begin());
            cnnNetworkImpl->setInputInfo(inputInfo);
        }
    }

    return std::shared_ptr<ICNNNetwork>(cnnNetworkImpl.release());
}

Builder::Network::operator const INetwork::Ptr() const {
    return build();
}

const Builder::Layer &Builder::Network::getLayer(idx_t layerId) const {
    for (auto& layer : getLayers()) {
        if (layer.getId() == layerId)
            return layer;
    }
    THROW_IE_EXCEPTION << "Cannot find layer with id: " << layerId;
}

Builder::Layer &Builder::Network::getLayer(idx_t layerId) {
    for (auto& layer : getLayers()) {
        if (layer.getId() == layerId)
            return layer;
    }
    THROW_IE_EXCEPTION << "Cannot find layer with id: " << layerId;
}

const std::vector<Connection> Builder::Network::getLayerConnections(idx_t layerId) const noexcept {
    std::vector<Connection> layerConnections;
    for (const auto connection : connections) {
        if (connection.from().layerId() == layerId || connection.to().layerId() == layerId)
            layerConnections.push_back(connection);
    }
    return layerConnections;
}
