// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include <unordered_set>

#include "debug.h"
#include "exec_graph_info.hpp"
#include <ngraph/graph_util.hpp>
#include <ngraph/pass/manager.hpp>
#include <ie_common.h>

#include "cnn_network_ngraph_impl.hpp"
#include <transformations/init_node_info.hpp>
#include <transformations/common_optimizations/common_optimizations.hpp>
#include <transformations/opset_conversions/convert_opset2_to_opset1.hpp>
#include <transformations/opset_conversions/convert_opset3_to_opset2.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_opset1_to_legacy.hpp>
#include <legacy/transformations/convert_opset1_to_legacy/convert_prior_to_ie_prior.hpp>

#include "legacy/convert_function_to_cnn_network.hpp"
#include "legacy/graph_tools.hpp"
#include "legacy/details/ie_cnn_network_tools.h"
#include <legacy/cnn_network_impl.hpp>

#ifdef ENABLE_V7_SERIALIZE
# include "network_serializer_v7.hpp"
#endif

using namespace std;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

std::map<CNNLayer*, bool> getConstLayersMap(const CNNNetwork& network) {
    std::map<CNNLayer*, bool> result;

    const std::vector<CNNLayerPtr> layers = CNNNetSortTopologically(network);
    for (const CNNLayerPtr& layer : layers) {
        if (equal(layer->type, "Input") || equal(layer->type, "Memory")) {
            result.emplace(layer.get(), false);
            continue;
        }

        if ((equal(layer->type, "Const")) || (equal(layer->type, "Shape")) || (equal(layer->type, "PriorBox"))) {
            result.emplace(layer.get(), true);
            continue;
        }

        bool allParentsAreConst = true;
        for (const DataWeakPtr& insWeakData : layer->insData) {
            const DataPtr insData = insWeakData.lock();
            if (insData == nullptr) {
                IE_THROW() << "input data is absent";
            }

            const CNNLayerWeakPtr parentWeak = getCreatorLayer(insData);
            const CNNLayerPtr parent = parentWeak.lock();
            if (parent == nullptr) {
                IE_THROW() << "parentLayer is absent";
            }

            const auto parentIt = result.find(parent.get());
            if (parentIt == result.end()) {
                IE_THROW() << "parent layer '" << parent->name << "' was not found";
            }

            if (!parentIt->second) {
                result.emplace(layer.get(), false);
                allParentsAreConst = false;
                break;
            }
        }

        if (allParentsAreConst) {
            result.emplace(layer.get(), true);
        }
    }

    return result;
}

CNNNetworkImpl::CNNNetworkImpl() {}

CNNNetworkImpl::CNNNetworkImpl(const CNNNetwork & cnnnetwork) {
    IE_SUPPRESS_DEPRECATED_START
    auto & icnnnetwork = static_cast<const ICNNNetwork &>(cnnnetwork);
    IE_SUPPRESS_DEPRECATED_END
    auto ngraphImplPtr = dynamic_cast<const details::CNNNetworkNGraphImpl*>(&icnnnetwork);
    IE_ASSERT(ngraphImplPtr != nullptr);
    IE_ASSERT(ngraphImplPtr->getFunction() != nullptr);
    auto graph = ngraph::clone_function(*ngraphImplPtr->getFunction());

    ::ngraph::pass::Manager manager;
    manager.register_pass<::ngraph::pass::InitNodeInfo>();
    // WA: ConvertPriorBox must be executed before the 1st ConstantFolding pass
    manager.register_pass<::ngraph::pass::ConvertPriorBox>();
    manager.register_pass<::ngraph::pass::CommonOptimizations>();
    manager.register_pass<::ngraph::pass::ConvertOpSet3ToOpSet2>();
    manager.register_pass<::ngraph::pass::ConvertOpSet2ToOpSet1>();
    manager.register_pass<::ngraph::pass::ConvertOpSet1ToLegacy>();
    manager.run_passes(graph);

    InferenceEngine::details::convertFunctionToICNNNetwork(graph, cnnnetwork, this, false);
}

CNNNetworkImpl::~CNNNetworkImpl() {
    // In case of cycles, memory leaks occur: Layer holds shared_ptr<Data>, and vice versa.
    // Added additional check on cycles.
    bool res = false;
    try {
        res = CNNNetForestDFS(CNNNetGetAllInputLayers(this), [&](CNNLayerPtr layer) {}, false);
    } catch (const std::exception & ex) {
        std::cout << ex.what() << std::endl;
        // Exception means that network was invalid. Reset all data.
    }

    // workaround due to memory leaks
    if (!res) {
        for (const auto& data : _data) {
            if (!data.second) continue;
            for (auto& input : getInputTo(data.second)) {
                if (!input.second) continue;
                input.second.reset();
            }
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
    if (!layer) return;
    _layers[layer->name] = layer;
}

void CNNNetworkImpl::removeLayer(const std::string& layerName) {
    auto it = _layers.find(layerName);
    if (it != _layers.end()) {
        _layers.erase(it);
    }
}

void CNNNetworkImpl::renameLayer(const std::string& currentName, const std::string& newName) {
    const auto currentIt = _layers.find(currentName);
    if (currentIt == _layers.end()) {
        IE_THROW() << "Layer '" << currentName << "' was not found in layers";
    }

    if (_layers.find(newName) != _layers.end()) {
        IE_THROW() << "Layer with name '" << currentName << "' already exists in layers";
    }

    if (_inputData.find(newName) != _inputData.end()) {
        IE_THROW() << "Layer with name '" << currentName << "' already exists in input data";
    }

    if (_outputData.find(newName) != _outputData.end()) {
        IE_THROW() << "Layer with name '" << currentName << "' already exists in output data";
    }

    const auto currentDataIt = _data.find(currentName);
    if (currentDataIt == _data.end()) {
        IE_THROW() << "Layer '" << currentName << "' was not found in data";
    }

    if (_data.find(newName) != _data.end()) {
        IE_THROW() << "Layer with name '" << currentName << "' already exists in data";
    }

    bool wasUpdatedInput = false;
    for (auto inputDataIt = _inputData.begin(); inputDataIt != _inputData.end(); ++inputDataIt) {
        const CNNLayerPtr inputLayer = getCreatorLayer(inputDataIt->second->getInputData()).lock();
        if (inputLayer->name == currentName) {
            _inputData.emplace(newName, inputDataIt->second);
            _inputData.erase(inputDataIt);
            wasUpdatedInput = true;
            break;
        }
    }

    if (!wasUpdatedInput) {
        for (auto outputDataIt = _outputData.begin(); outputDataIt != _outputData.end(); ++outputDataIt) {
            const CNNLayerPtr outputLayer = getCreatorLayer(outputDataIt->second).lock();
            if (outputLayer->name == currentName) {
                _outputData.emplace(newName, outputDataIt->second);
                _outputData.erase(outputDataIt);
                break;
            }
        }
    }

    _layers.emplace(newName, currentIt->second);
    currentIt->second->name = newName;
    _layers.erase(currentIt);

    _data.emplace(newName, currentDataIt->second);
    currentDataIt->second->setName(newName);
    _data.erase(currentDataIt);
}

void CNNNetworkImpl::removeData(const std::string& dataName) {
    auto it = _data.find(dataName);
    if (it != _data.end()) {
        _data.erase(it);
    }
}

void CNNNetworkImpl::validate(int version) {
    std::set<std::string> layerNames;
    std::set<std::string> dataNames;

    InputsDataMap inputs;
    this->getInputsInfo(inputs);
    if (inputs.empty()) {
        IE_THROW() << "No input layers";
    }

    bool res = CNNNetForestDFS(
        CNNNetGetAllInputLayers(this),
        [&](CNNLayerPtr layer) {
            std::string layerName = layer->name;

            for (auto i : layer->insData) {
                auto data = i.lock();
                if (data) {
                    auto inputTo = getInputTo(data);
                    auto iter = inputTo.find(layerName);
                    auto dataName = data->getName();
                    if (iter == inputTo.end()) {
                        IE_THROW() << "Data " << data->getName() << " which inserted into the layer "
                                           << layerName << " does not point at this layer";
                    }
                    if (!getCreatorLayer(data).lock()) {
                        IE_THROW() << "Data " << dataName << " has no creator layer";
                    }
                } else {
                    IE_THROW() << "Data which inserted into the layer " << layerName << " is nullptr";
                }
            }
            for (auto data : layer->outData) {
                auto inputTo = getInputTo(data);
                std::string dataName = data->getName();
                for (auto layerIter : inputTo) {
                    CNNLayerPtr layerInData = layerIter.second;
                    if (!layerInData) {
                        IE_THROW() << "Layer which takes data " << dataName << " is nullptr";
                    }
                    auto insertedDatas = layerInData->insData;

                    auto it =
                        std::find_if(insertedDatas.begin(), insertedDatas.end(), [&](InferenceEngine::DataWeakPtr& d) {
                            return d.lock() == data;
                        });
                    if (it == insertedDatas.end()) {
                        IE_THROW() << "Layer " << layerInData->name << " which takes data " << dataName
                                           << " does not point at this data";
                    }
                }
                auto dataNameSetPair = dataNames.insert(dataName);
                if (!dataNameSetPair.second) {
                    IE_THROW() << "Data name " << dataName << " is not unique";
                }
            }
            auto layerSetPair = layerNames.insert(layerName);
            if (!layerSetPair.second) {
                IE_THROW() << "Layer name " << layerName << " is not unique";
            }
        },
        false);

    std::string inputType = "Input";
    for (auto i : inputs) {
        CNNLayerPtr layer = getCreatorLayer(i.second->getInputData()).lock();
        if (layer && !equal(layer->type, inputType)) {
            IE_THROW() << "Input layer " << layer->name << " should have Input type but actually its type is "
                               << layer->type;
        }
    }

    if (!res) {
        IE_THROW() << "Sorting not possible, due to existed loop.";
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
        return DescriptionBuffer(OUT_OF_BOUNDS, resp)
               << "port index " << outputIndex << " exceeds layer's outputs which is " << outLayer->outData.size();
    shared_ptr<Data> outData = outLayer->outData[outputIndex];
    _outputData[outData->getName()] = outData;
    return OK;
}

void CNNNetworkImpl::resolveOutput() {
    // check orphan nodes...
    for (auto kvp : _data) {
        if (!kvp.second->isInitialized())
            IE_THROW() << "data name [" << kvp.first << "] dimensions is not known";

        // data nodes not going to any layer are basically graph output...
        if (getInputTo(kvp.second).empty()) {
            _outputData[kvp.first] = kvp.second;
        }
    }
}

void CNNNetworkImpl::addOutput(const string& dataName) {
    auto it = _data.find(dataName);
    if (it == _data.end()) {
        IE_THROW() << "data [" << dataName << "] doesn't exist";
    }
    auto data = it->second;
    assert(data->getName() == dataName);
    _outputData[dataName] = data;
}

void CNNNetworkImpl::removeOutput(const string& dataName) {
    removeData(dataName);

    auto it = _outputData.find(dataName);
    if (it != _outputData.end()) {
        _outputData.erase(it);
    }
}

size_t CNNNetworkImpl::getBatchSize() const noexcept {
    if (!_inputData.size()) return 0;
    // currently CNNNetworkImpl::setBatchSize set the same values
    // for the latest dim as a batch, we can take the first input
    // and return batch size for it
    SizeVector dims = _inputData.cbegin()->second->getTensorDesc().getDims();
    // 3D input layout doesn't have batch notation for input so batch is 1
    if (dims.size() == 3 || dims.size() == 1) {
        return 1;
    }
    if (dims.size() == 0) {
        return 0;
    }
    return dims.at(0);
}

StatusCode CNNNetworkImpl::reshape(const std::map<std::string, std::vector<size_t>>& inputShapes,
                                   ResponseDesc* responseDesc) noexcept {
    for (const auto& pair : _inputData) {
        auto info = pair.second;
        if (info) {
            auto data = info->getInputData();
            auto it = inputShapes.find(pair.first);
            if (data && it != inputShapes.end()) {
                auto newDims = it->second;
                auto currentDims = data->getTensorDesc().getDims();
                if (newDims != currentDims) {
                    return DescriptionBuffer(NOT_IMPLEMENTED, responseDesc) <<
                        "You have called setBatchSize + reshape for CNNNetwork object. Please, either: \n"
                        "- [SUGGESTED] Regenerate IR with current version of Model Optimizer\n"
                        "- [WORKAROUND] Call only reshape method where proper batch is already set\n";
                }
            }
        }
    }

    return OK;
}

StatusCode CNNNetworkImpl::serialize(const std::string& xmlPath, const std::string& binPath, ResponseDesc* resp) const
    noexcept {
    try {
#ifdef ENABLE_V7_SERIALIZE
        IE_SUPPRESS_DEPRECATED_START
        Serialization::Serialize(xmlPath, binPath, CNNNetwork(
            std::const_pointer_cast<ICNNNetwork>(shared_from_this())));
        IE_SUPPRESS_DEPRECATED_END
        return OK;
#endif
    } catch (const Exception& e) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, resp) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, resp);
    }

    return DescriptionBuffer(NOT_IMPLEMENTED, resp) << "The CNNNetworkImpl::serialize is not implemented";
}


StatusCode CNNNetworkImpl::serialize(std::ostream& xmlBuf, std::ostream& binBuf, ResponseDesc* resp) const
    noexcept {
    return DescriptionBuffer(NOT_IMPLEMENTED, resp) << "The CNNNetworkImpl::serialize is not implemented";
}

StatusCode CNNNetworkImpl::serialize(std::ostream& xmlBuf, Blob::Ptr& binBlob, ResponseDesc* resp) const
    noexcept {
    return DescriptionBuffer(NOT_IMPLEMENTED, resp) << "The CNNNetworkImpl::serialize is not implemented";
}

StatusCode CNNNetworkImpl::setBatchSize(size_t size, ResponseDesc* responseDesc) noexcept {
    try {
        auto originalBatchSize = getBatchSize();
        if (originalBatchSize == size) return OK;
        if (_inputData.empty())
            return DescriptionBuffer(GENERAL_ERROR, responseDesc) << "Cannot set batch! Topology doesn't contain inputs!";

        SizeVector dims = _inputData.cbegin()->second->getTensorDesc().getDims();

        // 3D/1D/0D input layouts don't have batch notation
        if (dims.size() == 3 || dims.size() == 1 || dims.empty()) {
            return DescriptionBuffer(PARAMETER_MISMATCH, responseDesc) << "Cannot set batch for 0D/1D/3D input";
        }

        IE_SUPPRESS_DEPRECATED_START
        const std::map<CNNLayer*, bool> layersMap = getConstLayersMap(CNNNetwork(shared_from_this()));
        IE_SUPPRESS_DEPRECATED_END
        for (auto& layer : _data) {
            SizeVector dims = layer.second->getDims();
            CNNLayerPtr layerT = getCreatorLayer(layer.second).lock();

            bool constOrAbsent;
            if (layerT) {
                const auto it = layersMap.find(layerT.get());
                if (it == layersMap.end()) {
                    IE_THROW() << "layer '" << layerT->name << "' was not found in layers map";
                }
                constOrAbsent = it->second;
            } else {
                constOrAbsent = true;
            }

            if (!layerT || !constOrAbsent) {
                float diff = static_cast<float>(dims.at(0)) / static_cast<float>(originalBatchSize);
                dims.at(0) = static_cast<size_t>(std::ceil(size * diff));
                layer.second->setDims(dims);
            }
        }
        return OK;
    } catch (const Exception& e) {
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
    } catch (const Exception& e) {
        return DescriptionBuffer(GENERAL_ERROR, responseDesc) << e.what();
    } catch (const std::exception& e) {
        return DescriptionBuffer(UNEXPECTED, responseDesc) << e.what();
    } catch (...) {
        return DescriptionBuffer(UNEXPECTED, responseDesc);
    }
}
