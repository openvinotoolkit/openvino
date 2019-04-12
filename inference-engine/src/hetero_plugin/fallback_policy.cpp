//
// Copyright (C) 2018-2019 Intel Corporation.
//
// This software and the related documents are Intel copyrighted materials,
// and your use of them is governed by the express license under which they
// were provided to you (End User License Agreement for the Intel(R) Software
// Development Products (Version May 2017)). Unless the License provides
// otherwise, you may not use, modify, copy, publish, distribute, disclose or
// transmit this software or the related documents without Intel's prior
// written permission.
//
// This software and the related documents are provided as is, with no
// express or implied warranties, other than those that are expressly
// stated in the License.
//

#include "fallback_policy.h"
#include "hetero_device_loader.h"
#include "details/ie_cnn_network_iterator.hpp"
#include "ie_layers.h"
#include "ie_util_internal.hpp"
#include <fstream>
#include <vector>
#include <memory>

using namespace InferenceEngine;

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


FallbackPolicy::FallbackPolicy(std::map<std::string, InferenceEngine::IHeteroDeviceLoader::Ptr> &deviceLoaders,
                               bool dumpDotFile) :
    _deviceLoaders(deviceLoaders),
    _dumpDotFile(dumpDotFile) {
}

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
            IHeteroDeviceLoader::Ptr loader;
            loader = std::make_shared<HeteroDeviceLoader>(d);
            HeteroDeviceLoader *pdl = dynamic_cast<HeteroDeviceLoader *>(loader.get());
            pdl->initConfigs(allConfigs, extensions);
            _deviceLoaders[d] = loader;
        }
    }
}

void FallbackPolicy::setAffinity(const std::map<std::string, std::string>& config, ICNNNetwork& network) {
    std::map<std::string, QueryNetworkResult> queryResults;
    // go oger devices, create appropriate plugins and
    for (const auto &i : _fallbackDevices) {
        QueryNetworkResult r;
        _deviceLoaders[i]->QueryNetwork(i, network, config, r);
        queryResults[i] = r;
    }

    details::CNNNetworkIterator i(const_cast<ICNNNetwork *>(&network));
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        for (auto &&j : _fallbackDevices) {
            auto &qr = queryResults[j];
            if (qr.supportedLayers.find(layer->name) != qr.supportedLayers.end()) {
                layer->affinity = j;
                break;
            }
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
