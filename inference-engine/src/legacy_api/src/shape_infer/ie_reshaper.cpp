// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_infer/ie_reshaper.hpp"

#include <debug.h>
#include <ie_layers.h>

#include <blob_factory.hpp>
#include <functional>
#include <graph_tools.hpp>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "details/caseless.hpp"
#include "details/ie_cnn_network_tools.h"
#include "ie_cnn_layer_builder.h"
#include "ie_reshaper.hpp"
#include "shape_infer/built-in/ie_built_in_holder.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;
using namespace ShapeInfer;

inline static std::vector<CNNLayerPtr> SortTopologicallyStartsFrom(const std::vector<DataPtr>& inputs) {
    std::vector<CNNLayerPtr> all_layers;
    CNNNetForestDFS(
        inputs,
        [&](CNNLayerPtr current) {
            all_layers.push_back(current);
        },
        false);
    std::reverse(all_layers.begin(), all_layers.end());
    return all_layers;
}

Reshaper::Reshaper(std::vector<DataPtr> insDatas, const LauncherCreator::Ptr& launcherCreator) {
    auto builtIn = std::make_shared<BuiltInShapeInferHolder>();
    _allTypes = getTypeNamesFromExtension(builtIn);
    _extensions.push_back(builtIn);

    _allSortedLayers = SortTopologicallyStartsFrom(insDatas);
    for (auto& in_data : insDatas) {
        for (auto layer : in_data->getInputTo()) {
            _inputLayers.insert(layer.second);
        }
    }

    if (_inputLayers.empty() || _allSortedLayers.empty())
        THROW_IE_EXCEPTION << "Unsupported model for shape inference: failed to collect inputs and layers";

    for (auto const& currentLayer : _allSortedLayers) {
        auto createdLauncher = launcherCreator->createNotInputLauncher(currentLayer.get(), _extensions);
        _launchers.insert(createdLauncher);
    }
}

Reshaper::Reshaper(ICNNNetwork& network, const LauncherCreator::Ptr& launcherCreator) {
    auto builtIn = std::make_shared<BuiltInShapeInferHolder>();
    _allTypes = getTypeNamesFromExtension(builtIn);
    _extensions.push_back(builtIn);

    auto inputLayers = CNNNetGetAllInputLayers(network);
    for (const auto& layer : inputLayers) {
        _inputLayers.insert(layer);
    }

    _allSortedLayers = CNNNetSortTopologically(network);
    if (_inputLayers.empty() || _allSortedLayers.empty())
        THROW_IE_EXCEPTION << "Unsupported model for shape inference: failed to collect inputs and layers";
    for (auto const& currentLayer : _allSortedLayers) {
        auto foundInput =
            std::find_if(_inputLayers.begin(), _inputLayers.end(), [&currentLayer](const CNNLayerPtr& inputLayer) {
                return currentLayer->name == inputLayer->name;
            });
        ReshapeLauncher::Ptr createdLauncher;
        if (foundInput == _inputLayers.end()) {
            createdLauncher = launcherCreator->createNotInputLauncher(currentLayer.get(), _extensions);
        } else {
            createdLauncher = launcherCreator->createInputLauncher(currentLayer.get(), _extensions);
        }
        _launchers.insert(createdLauncher);
    }
}

void Reshaper::AddExtension(const IShapeInferExtensionPtr& extension) {
    if (!extension)
        THROW_IE_EXCEPTION << "Failed to add empty shape infer extension";

    auto newLayerTypes = getTypeNamesFromExtension(extension);
    std::string badLayerTypes;
    for (const auto& type : newLayerTypes) {
        auto ret = _allTypes.insert(type);
        if (!ret.second) {
            if (!badLayerTypes.empty()) badLayerTypes += ", ";
            badLayerTypes += type;
        }
    }
    if (!badLayerTypes.empty())
        THROW_IE_EXCEPTION << "Failed to add extension with already registered types:" << badLayerTypes;

    for (auto const& layerType : newLayerTypes) {
        auto foundLauncher = _launchers.begin();
        // find all layers with given type
        std::vector<ReshapeLauncher::Ptr> launchersToInsert;
        while (foundLauncher != _launchers.end()) {
            foundLauncher =
                std::find_if(foundLauncher, _launchers.end(), [&layerType](const ReshapeLauncher::Ptr& launcher) {
                    return layerType == launcher->getLayerType();
                });
            if (foundLauncher != _launchers.end()) {
                IE_SUPPRESS_DEPRECATED_START
                IShapeInferImpl::Ptr impl;
                StatusCode sts = extension->getShapeInferImpl(impl, layerType.c_str(), nullptr);
                IE_SUPPRESS_DEPRECATED_END
                if (sts == OK && impl != nullptr) {
                    auto newLauncher = std::make_shared<ReshapeLauncher>((*foundLauncher)->getLayer(), impl);
                    newLauncher->setShapeInferImpl(impl);
                    launchersToInsert.push_back(newLauncher);
                    foundLauncher = _launchers.erase(foundLauncher);
                } else {
                    THROW_IE_EXCEPTION << "Failed to get registered Shape Infer Implementation for type: " << layerType;
                }
            }
        }
        for (const auto& launcher : launchersToInsert) {
            _launchers.insert(launcher);
        }
    }
    _extensions.push_back(extension);
}

ReshapeLauncher::Ptr Reshaper::getLauncherByLayerName(const std::string& layerName) const {
    auto foundLauncher =
        std::find_if(_launchers.begin(), _launchers.end(), [&layerName](const ReshapeLauncher::Ptr& launcher) {
            return launcher->getLayerName() == layerName;
        });
    if (foundLauncher == _launchers.end())
        THROW_IE_EXCEPTION << "Failed to reshape layer ('" << layerName << "'): can't find the corresponding launcher";
    return *foundLauncher;
}

StatusCode Reshaper::run(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {
    // WA: In another case we should change the registration logic of shape implementations
    static std::mutex reshapeMutex;
    {
        std::lock_guard<std::mutex> lock(reshapeMutex);
        // Reset all shapes from previous run
        for (const auto& launcher : _launchers) {
            launcher->reset();
        }

        // Set new input shapes
        for (auto const& input : _inputLayers) {
            std::string layerName = input->name;
            for (auto const& outData : input->outData) {
                std::string dataName = outData->getName();
                auto foundShapeIt = inputShapes.find(dataName);
                auto foundLauncher = getLauncherByLayerName(layerName);
                if (foundShapeIt != inputShapes.end()) {
                    foundLauncher->setShapeByName(foundShapeIt->second, dataName);
                } else {
                    foundLauncher->setIRShapeByName(dataName);
                }
            }
        }

        // do reshape
        for (auto& layer : _allSortedLayers) {
            auto foundLauncher = getLauncherByLayerName(layer->name);
            foundLauncher->reshape(_launchers);
            foundLauncher->constInfer(_launchers);
        }

        // apply changes
        for (auto& layer : _allSortedLayers) {
            auto foundLauncher = getLauncherByLayerName(layer->name);
            foundLauncher->applyChanges(layer.get());
        }
        return OK;
    }
}

StatusCode Reshaper::runNoApply(const std::map<std::string, SizeVector>& inputShapes, ResponseDesc* resp) {
    // Reset all shapes from previous run
    for (const auto& launcher : _launchers) {
        launcher->reset();
    }

    // Set new input shapes
    for (auto const& input : _inputLayers) {
        std::string layerName = input->name;
        for (auto const& inData_w : input->insData) {
            auto inData = inData_w.lock();
            auto dataName = inData->getName();
            auto foundShapeIt = inputShapes.find(dataName);
            auto foundLauncher = getLauncherByLayerName(layerName);
            if (foundShapeIt != inputShapes.end()) {
                foundLauncher->setShapeByName(foundShapeIt->second, dataName);
            } else {
                foundLauncher->setIRShapeByName(dataName);
            }
        }
    }

    // do reshape
    for (auto& layer : _allSortedLayers) {
        auto foundLauncher = getLauncherByLayerName(layer->name);
        foundLauncher->reshape(_launchers);
    }
    return OK;
}

StatusCode Reshaper::apply(ResponseDesc* resp) {
    // apply changes
    for (auto& layer : _allSortedLayers) {
        auto foundLauncher = getLauncherByLayerName(layer->name);
        foundLauncher->applyChanges(layer.get());
    }
    return OK;
}

SizeVector Reshaper::getResultShapeFor(DataPtr& data, ResponseDesc* resp) {
    auto creator_layer = data->getCreatorLayer().lock();
    std::string creator_layer_name;
    if (creator_layer) {
        creator_layer_name = creator_layer->name;
    }
    auto foundLauncher = getLauncherByLayerName(creator_layer_name);
    return foundLauncher->getShapeByName(data->getName());
}

caseless_set<std::string> Reshaper::getTypeNamesFromExtension(const IShapeInferExtensionPtr& extension) {
    char** types = nullptr;
    unsigned int size = 0;
    ResponseDesc resp;
    IE_SUPPRESS_DEPRECATED_START
    StatusCode sts = extension->getShapeInferTypes(types, size, &resp);
    IE_SUPPRESS_DEPRECATED_END
    if (sts != OK) THROW_IE_EXCEPTION << "Failed to get types from extension: " << resp.msg;
    caseless_set<std::string> typesSet;
    for (int i = 0; i < size; i++) {
        std::string type(types[i], strlen(types[i]));
        delete[] types[i];
        typesSet.insert(type);
    }
    delete[] types;
    return typesSet;
}

ReshapeLauncher::Ptr LauncherCreator::createNotInputLauncher(const CNNLayer* layer,
                                                             const std::vector<IShapeInferExtensionPtr>& extensions) {
    auto layerType = layer->type;
    if ((::details::equal(layerType, "memory") && layer->GetParamAsInt("index")) ||
        ::details::equal(layerType, "const") || ::details::equal(layerType, "input")) {
        THROW_IE_EXCEPTION << "Failed to reshape: Layer with type `" << layerType
                           << "` can't be intermediate layer in network";
    }

    for (const auto& extension : extensions) {
        IE_SUPPRESS_DEPRECATED_START
        IShapeInferImpl::Ptr impl = nullptr;
        StatusCode sts = extension->getShapeInferImpl(impl, layerType.c_str(), nullptr);
        IE_SUPPRESS_DEPRECATED_END
        if (sts == OK && impl != nullptr) {
            if (::details::equal(layerType, "memory") && !layer->GetParamAsInt("index")) {
                return std::make_shared<OutMemoryReshapeLauncher>(layer, nullptr);
            }
            return std::make_shared<ReshapeLauncher>(layer, impl);
        }
    }
    return std::make_shared<FakeReshapeLauncher>(layer, nullptr);
}

ReshapeLauncher::Ptr LauncherCreator::createInputLauncher(const CNNLayer* layer,
                                                          const std::vector<IShapeInferExtensionPtr>& extensions) {
    auto layerType = layer->type;
    if (::details::equal(layerType, "memory") && layer->GetParamAsInt("index")) {
        return std::make_shared<InputReshapeLauncher>(layer, nullptr);
    } else if (::details::equal(layerType, "const")) {
        return std::make_shared<ConstReshapeLauncher>(layer, nullptr);
    } else if (::details::equal(layerType, "input")) {
        return std::make_shared<InputReshapeLauncher>(layer, nullptr);
    }
    THROW_IE_EXCEPTION << "Failed to reshape: Layer with type `" << layerType
                       << "` can't be input. Supported input types: Input, Const and Memory(with index=1)";
}
