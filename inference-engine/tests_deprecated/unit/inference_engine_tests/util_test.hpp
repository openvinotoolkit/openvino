// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_algorithm.hpp>
#include <ie_util_internal.hpp>

#include <unordered_set>
#include <unordered_map>
#include <string>
#include <memory>

namespace IE = InferenceEngine;
using IE::details::contains;

class NetBuilder {
    using LayersMap = std::unordered_map<std::string, IE::CNNLayerPtr>;
    using DataMap = std::unordered_map<std::string, IE::DataPtr>;
    using InputsSet = std::unordered_set<IE::InputInfo::Ptr>;
    LayersMap _layers;
    DataMap _data;
    InputsSet _inputs;
public:
    NetBuilder() = default;

    NetBuilder(const NetBuilder&) = delete;

    template<typename... Args>
    NetBuilder& data(Args&& ... args) {
        auto newData = std::make_shared<IE::Data>(std::forward<Args>(args)...);
        assert(!contains(_data, newData->getName()));
        _data[newData->getName()] = newData;
        return *this;
    }

    template<typename T, typename... Args>
    NetBuilder& layer(Args&& ... args) {
        auto newLayer = std::make_shared<T>(std::forward<Args>(args)...);
        assert(!contains(_layers, newLayer->name));
        _layers[newLayer->name] = std::static_pointer_cast<IE::CNNLayer>(newLayer);
        return *this;
    }

    const LayersMap& getLayersMap() const {
        return _layers;
    }

    const DataMap& getDataMap() const {
        return _data;
    }

    NetBuilder& linkDataTo(const std::string& dataName,
                           const std::string& nextlayerName) {
        assert(contains(_layers, nextlayerName));
        assert(contains(_data, dataName));

        auto nextlayer = _layers[nextlayerName];
        auto data = _data[dataName];

        nextlayer->insData.push_back(data);
        data->getInputTo().insert({nextlayerName, nextlayer});
        return *this;
    }

    NetBuilder& linkToData(const std::string& prevlayerName,
                           const std::string& dataName) {
        assert(contains(_layers, prevlayerName));
        assert(contains(_data, dataName));

        auto prevlayer = _layers[prevlayerName];
        auto data = _data[dataName];
        assert(nullptr == data->getCreatorLayer().lock());

        prevlayer->outData.push_back(data);
        data->getCreatorLayer() = prevlayer;
        return *this;
    }

    NetBuilder& linkLayers(const std::string& prevlayerName,
                           const std::string& nextlayerName,
                           const std::string& dataName) {
        linkToData(prevlayerName, dataName);
        linkDataTo(dataName, nextlayerName);
        return *this;
    }

    NetBuilder& linkData(const std::string& prevDataName,
                         const std::string& nextDataName,
                         const std::string& layerName) {
        linkDataTo(prevDataName, layerName);
        linkToData(layerName, nextDataName);
        return *this;
    }

    template<typename... Args>
    NetBuilder& addInput(const std::string& dataName, Args&& ... args) {
        assert(!dataName.empty());
        assert(contains(_data, dataName));
        auto input = std::make_shared<IE::InputInfo>(
                std::forward<Args>(args)...);
        input->setInputData(_data[dataName]);
        _inputs.insert(std::move(input));
        return *this;
    }

    IE::details::CNNNetworkImplPtr finalize() {
        auto net = std::make_shared<IE::details::CNNNetworkImpl>();

        for (auto&& it: _data) {
            auto& data = it.second;
            net->getData(it.first) = data;
            if (nullptr == data->getCreatorLayer().lock()) {
                auto input = std::make_shared<IE::InputInfo>();
                input->setInputData(data);
                net->setInputInfo(input);
            }
        }
        for (auto&& it: _layers) {
            net->addLayer(it.second);
        }
        for (auto& i : _inputs) {
            net->setInputInfo(std::move(i));
        }

        net->resolveOutput();

        return net;
    }
};
