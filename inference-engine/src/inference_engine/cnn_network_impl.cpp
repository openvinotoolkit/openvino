// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ie_common.h>
#include "cnn_network_impl.hpp"
#include <memory>
#include <map>
#include <set>
#include <string>
#include <cassert>
#include <shape_infer/ie_reshaper.hpp>
#include "debug.h"
#include "graph_tools.hpp"
#include <vector>
#include <math.h>
#include "network_serializer.h"
#include "ie_profiling.hpp"

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

CNNNetworkImpl::CNNNetworkImpl(): _stats(new CNNNetworkStatsImpl()) {
}

CNNNetworkImpl::~CNNNetworkImpl() {
    for (auto& data : _data) {
        if (!data.second)
            continue;
        for (auto& input : data.second->getInputTo()) {
            if (!input.second)
                continue;
            input.second.reset();
        }
    }
}

void CNNNetworkImpl::getOutputsInfo(std::map<std::string, DataPtr>& out) const noexcept {
    out = _outputData;
}

void CNNNetworkImpl::getInputsInfo(InputsDataMap& inputs) const noexcept {
    inputs = _inputData;
}

void CNNNetworkImpl::addLayer(const CNNLayerPtr& layer) noexcept {
    if (!layer)
        return;
    _layers[layer->name] = layer;
}

void CNNNetworkImpl::removeLayer(const string& layerName) {
    auto it = _layers.find(layerName);
    if (it != _layers.end()) { _layers.erase(it); }
}

void CNNNetworkImpl::removeData(const string& dataName) {
    auto it = _data.find(dataName);
    if (it != _data.end()) { _data.erase(it); }
}

void CNNNetworkImpl::validate(int version) {
    std::set<std::string> layerNames;
    std::set<std::string> dataNames;

    InputsDataMap inputs;
    this->getInputsInfo(inputs);
    if (inputs.empty()) {
        THROW_IE_EXCEPTION << "No input layers";
    }

    bool res = CNNNetForestDFS(CNNNetGetAllInputLayers(*this), [&](CNNLayerPtr layer) {
        std::string layerName = layer->name;

        for (auto i : layer->insData) {
            auto data = i.lock();
            if (data) {
                auto inputTo = data->getInputTo();
                auto iter = inputTo.find(layerName);
                auto dataName = data->getName();
                if (iter == inputTo.end()) {
                    THROW_IE_EXCEPTION << "Data " << data->getName() << " which inserted into the layer "
                                       << layerName
                                       << " does not point at this layer";
                }
                if (!data->getCreatorLayer().lock()) {
                    THROW_IE_EXCEPTION << "Data " << dataName << " has no creator layer";
                }
            } else {
                THROW_IE_EXCEPTION << "Data which inserted into the layer " << layerName << " is nullptr";
            }
        }
        for (auto data : layer->outData) {
            auto inputTo = data->getInputTo();
            std::string dataName = data->getName();
            for (auto layerIter : inputTo) {
                CNNLayerPtr layerInData = layerIter.second;
                if (!layerInData) {
                    THROW_IE_EXCEPTION << "Layer which takes data " << dataName << " is nullptr";
                }
                auto insertedDatas = layerInData->insData;

                auto it = std::find_if(insertedDatas.begin(), insertedDatas.end(),
                                       [&](InferenceEngine::DataWeakPtr& d) {
                                           return d.lock() == data;
                                       });
                if (it == insertedDatas.end()) {
                    THROW_IE_EXCEPTION << "Layer " << layerInData->name << " which takes data " << dataName
                                       << " does not point at this data";
                }
            }
            auto dataNameSetPair = dataNames.insert(dataName);
            if (!dataNameSetPair.second) {
                THROW_IE_EXCEPTION << "Data name " << dataName << " is not unique";
            }
        }
        auto layerSetPair = layerNames.insert(layerName);
        if (!layerSetPair.second) {
            THROW_IE_EXCEPTION << "Layer name " << layerName << " is not unique";
        }
    }, false);


    std::string inputType = "Input";
    for (auto i : inputs) {
        CNNLayerPtr layer = i.second->getInputData()->getCreatorLayer().lock();
        if (layer && !equal(layer->type, inputType)) {
            THROW_IE_EXCEPTION << "Input layer " << layer->name
                               << " should have Input type but actually its type is " << layer->type;
        }
    }

    if (!res) {
        THROW_IE_EXCEPTION << "Sorting not possible, due to existed loop.";
    }
}

StatusCode CNNNetworkImpl::getLayerByName(const char* layerName, CNNLayerPtr& out, ResponseDesc* resp) const noexcept {
    auto it = _layers.find(layerName);
    if (it == _layers.end())
        return DescriptionBuffer(NOT_FOUND, resp) << "Layer " << layerName << " not found in network";
    out = it->second;
    return OK;
}

StatusCode CNNNetworkImpl::addOutput(const std::string& layerName, size_t outputIndex, ResponseDesc* resp) noexcept {
    CNNLayerPtr outLayer;
    auto rc = getLayerByName(layerName.c_str(), outLayer, resp);
    if (rc != OK) return rc;

    if (outputIndex >= outLayer->outData.size())
        return DescriptionBuffer(OUT_OF_BOUNDS, resp) << "port index " << outputIndex
                                                      << " exceeds layer's outputs which is "
                                                      << outLayer->outData.size();
    shared_ptr<Data> outData = outLayer->outData[outputIndex];
    _outputData[outData->getName()] = outData;
    return OK;
}

void CNNNetworkImpl::resolveOutput() {
    // check orphan nodes...
    for (auto kvp : _data) {
        if (!kvp.second->isInitialized())
            THROW_IE_EXCEPTION << "data name [" << kvp.first << "] dimensions is not known";

        // data nodes not going to any layer are basically graph output...
        if (kvp.second->getInputTo().empty()) {
            _outputData[kvp.first] = kvp.second;
        }
    }
}

void CNNNetworkImpl::addOutput(const string& dataName) {
    auto it = _data.find(dataName);
    if (it == _data.end()) {
        THROW_IE_EXCEPTION << "data [" << dataName << "] doesn't exist";
    }
    auto data = it->second;
    assert(data->getName() == dataName);
    _outputData[dataName] = data;
}

size_t CNNNetworkImpl::getBatchSize() const noexcept {
    if (!_inputData.size())
        return 0;
    // currently CNNNetworkImpl::setBatchSize set the same values
    // for the latest dim as a batch, we can take the first input
    // and return batch size for it
    SizeVector dims = _inputData.cbegin()->second->getTensorDesc().getDims();
    // 3D input layout doesn't have batch notation for input so batch is 1
    if (dims.size() == 3 || dims.size() == 1) {
        return 1;
    }
    return dims.at(0);
}

StatusCode
CNNNetworkImpl::reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                        ResponseDesc* responseDesc) noexcept {
    try {
        if (!_reshaper) _reshaper = std::make_shared<ShapeInfer::Reshaper>(*this);
        _reshaper->run(inputShapes);
    } catch (const InferenceEngineException& e) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, responseDesc) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, responseDesc);
    }
    return OK;
}

StatusCode
CNNNetworkImpl::AddExtension(const InferenceEngine::IShapeInferExtensionPtr& extension,
                             InferenceEngine::ResponseDesc* resp) noexcept {
    try {
        if (!_reshaper) _reshaper = std::make_shared<ShapeInfer::Reshaper>(*this);
        _reshaper->AddExtension(extension);
    } catch (const InferenceEngineException& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, resp);
    }
    return OK;
}

StatusCode CNNNetworkImpl::serialize(const std::string &xmlPath, const std::string &binPath, ResponseDesc* resp) const noexcept {
    try {
        NetworkSerializer::serialize(xmlPath, binPath, (InferenceEngine::ICNNNetwork&)*this);
    } catch (const InferenceEngineException& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, resp);
    }
    return OK;
}

StatusCode CNNNetworkImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        auto originalBatchSize = getBatchSize();
        if (originalBatchSize == size)
            return OK;
        SizeVector dims = _inputData.cbegin()->second->getTensorDesc().getDims();

        // 3D input layout doesn't have batch notation
        if (dims.size() == 3 || dims.size() == 1) {
            return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc) << "Cannot set batch for 1D/3D input";
        }

        std::string constType = "Const";
        for (auto layer : _data) {
            SizeVector dims = layer.second->getDims();
            // Calculates original size for batch = 1
            CNNLayerPtr layerT = layer.second->getCreatorLayer().lock();
            if (!layerT || !equal(layerT->type, constType)) {
                float diff = static_cast<float>(dims.at(0)) / static_cast<float>(originalBatchSize);
                dims.at(0) = static_cast<size_t>(std::ceil(size * diff));
                layer.second->setDims(dims);
            }
        }
        return OK;
    } catch (const InferenceEngineException& e) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, responseDesc) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, responseDesc);
    }
}

StatusCode CNNNetworkImpl::setBatchSizeReshape(size_t size, ResponseDesc* responseDesc) noexcept {
    InputShapes inputShapes;
    try {
        for (const auto& pair : _inputData) {
            auto info = pair.second;
            if (info) {
                auto data = info->getInputData();
                if (data) {
                    auto dims = data->getTensorDesc().getDims();
                    dims[0] = size;
                    inputShapes[data->getName()] = dims;
                }
            }
        }
        return reshape(inputShapes, responseDesc);
    } catch (const InferenceEngineException& e) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, responseDesc) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, responseDesc);
    }
}
