// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fallback_policy.hpp"
#include "hetero_device_loader.hpp"
#include "details/ie_cnn_network_iterator.hpp"
#include "ie_layers.h"
#include "ie_util_internal.hpp"
#include <fstream>
#include <vector>
#include <memory>
#include <utility>
#include <map>
#include <string>

using namespace InferenceEngine;

IE_SUPPRESS_DEPRECATED_START

QueryNetworkResult::QueryNetworkResult() : rc(OK) {
}

const QueryNetworkResult & QueryNetworkResult::operator= (const QueryNetworkResult & q) {
    supportedLayers = q.supportedLayers;
    supportedLayersMap = q.supportedLayersMap;
    rc = q.rc;
    resp = q.resp;

    return *this;
}

QueryNetworkResult & QueryNetworkResult::operator= (QueryNetworkResult && q) {
    supportedLayers = q.supportedLayers;
    supportedLayersMap = q.supportedLayersMap;
    rc = q.rc;
    resp = q.resp;

    return *this;
}

QueryNetworkResult::QueryNetworkResult(const QueryNetworkResult & instance) :
    supportedLayers(instance.supportedLayers),
    supportedLayersMap(instance.supportedLayersMap),
    rc(instance.rc),
    resp(instance.resp) {
}

QueryNetworkResult::~QueryNetworkResult() {
}

IE_SUPPRESS_DEPRECATED_END

void dla_layer_colorer(const CNNLayerPtr layer,
                       ordered_properties &printed_properties,
                       ordered_properties &node_properties) {
    printed_properties.insert(printed_properties.begin(),
                              std::pair<std::string, std::string>("device", layer->affinity));
    if (layer->affinity == "CPU") {
        node_properties.emplace_back("fillcolor", "#5A5DF0");
    } else if (layer->affinity == "FPGA") {
        node_properties.emplace_back("fillcolor", "#20F608");
    } else if (layer->affinity == "GPU") {
        node_properties.emplace_back("fillcolor", "#F1F290");
    } else {
        node_properties.emplace_back("fillcolor", "#11F110");
    }
}

IE_SUPPRESS_DEPRECATED_START
FallbackPolicy::FallbackPolicy(std::map<std::string, InferenceEngine::IHeteroDeviceLoader::Ptr> &deviceLoaders,
                               bool dumpDotFile, const InferenceEngine::ICore * core) :
    _deviceLoaders(deviceLoaders),
    _dumpDotFile(dumpDotFile),
    _core(core) {
}
IE_SUPPRESS_DEPRECATED_END

void FallbackPolicy::init(const std::string &config, const std::map<std::string, std::string> &allConfigs,
                          const std::vector<InferenceEngine::IExtensionPtr> &extensions) {
    if (config.empty()) {
        THROW_IE_EXCEPTION << "Cannot set affinity according to fallback policy because the order of devices was not initialized";
    }
    // parsing the string and splitting to tokens
    std::string::size_type i = 0;
    std::string::size_type idelimeter;
    while ((idelimeter = config.find(',', i)) != std::string::npos) {
        _fallbackDevices.push_back(config.substr(i, idelimeter - i));
        i = idelimeter + 1;
    }
    _fallbackDevices.push_back(config.substr(i, config.length() - i));

    for (auto d : _fallbackDevices) {
        if (_deviceLoaders.find(d) == _deviceLoaders.end()) {
            IE_SUPPRESS_DEPRECATED_START
            IHeteroDeviceLoader::Ptr loader;
            loader = std::make_shared<HeteroDeviceLoader>(d, _core);
            HeteroDeviceLoader *pdl = static_cast<HeteroDeviceLoader *>(loader.get());
            IE_SUPPRESS_DEPRECATED_END
            pdl->initConfigs(allConfigs, extensions);
            _deviceLoaders[d] = loader;
        }
    }
}

QueryNetworkResult FallbackPolicy::getAffinities(const std::map<std::string, std::string>& config, const ICNNNetwork& network) const {
    QueryNetworkResult returnValue;
    returnValue.rc = StatusCode::OK;

    std::map<std::string, QueryNetworkResult> queryResults;
    // go over devices, create appropriate plugins and
    for (const auto &i : _fallbackDevices) {
        QueryNetworkResult r;
        IE_SUPPRESS_DEPRECATED_START
        _deviceLoaders[i]->QueryNetwork(i, network, config, r);
        if (StatusCode::OK != r.rc) {
            returnValue.rc = r.rc;
            std::string msg = r.resp.msg;
            snprintf(returnValue.resp.msg, msg.size(), "%s", msg.c_str());
            THROW_IE_EXCEPTION << "Failed to call QueryNetwork for " << i << " device, error: " << r.resp.msg;
            return r;
        }
        queryResults[i] = r;
        IE_SUPPRESS_DEPRECATED_END
    }

    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        for (auto &&j : _fallbackDevices) {
            auto &qr = queryResults[j];
            if (qr.supportedLayersMap.find(layer->name) != qr.supportedLayersMap.end()) {
                returnValue.supportedLayersMap[layer->name] = j;
                IE_SUPPRESS_DEPRECATED_START
                returnValue.supportedLayers.insert(layer->name);
                IE_SUPPRESS_DEPRECATED_END

                break;
            }
        }
        i++;
    }

    return returnValue;
}

void FallbackPolicy::setAffinity(const QueryNetworkResult & qr, ICNNNetwork& network) const {
    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));

    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        auto it = qr.supportedLayersMap.find(layer->name);
        if (it != qr.supportedLayersMap.end()) {
            layer->affinity = it->second;
        }
        i++;
    }

    if (_dumpDotFile) {
        std::stringstream stream(std::stringstream::out);
        stream << "hetero_affinity_" << network.getName() << ".dot";

        std::ofstream file(stream.str().c_str());
        saveGraphToDot(network, file, dla_layer_colorer);
    }
}
