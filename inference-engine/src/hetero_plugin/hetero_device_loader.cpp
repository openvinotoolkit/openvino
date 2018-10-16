// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero_device_loader.h"
#include "ie_plugin_dispatcher.hpp"
#include <string>
#include <stdio.h>
#include <map>
#include <vector>
#include <ie_plugin_config.hpp>

using namespace InferenceEngine;

StatusCode HeteroDeviceLoader::LoadNetwork(
    const std::string &device,
    IExecutableNetwork::Ptr &ret,
    ICNNNetwork &network,
    const std::map<std::string, std::string> &config,
    ResponseDesc *resp) noexcept {
    if (device != _deviceId) {
        if (resp) {
            std::string msg("Current HeteroDeviceLoader doesn't support device passed to LoadNetwork");
            snprintf(resp->msg, msg.size(), "%s", msg.c_str());
        }
        return NETWORK_NOT_LOADED;
    }
    if (!_plugin) {
        return NETWORK_NOT_LOADED;
    }
    // preparing local version of configs which are supported by plugins
    std::map<std::string, std::string> tconfig;

    for (auto c : config) {
        std::map<std::string, std::string> oneconfig;
        oneconfig[c.first] = c.second;
        InferenceEngine::ResponseDesc response;
        if (_plugin->SetConfig(oneconfig, &response) == OK) {
            tconfig[c.first] = c.second;
        }
    }

    return _plugin->LoadNetwork(ret, network, tconfig, resp);
}

void HeteroDeviceLoader::QueryNetwork(const std::string &device,
                                      const ICNNNetwork &network,
                                      QueryNetworkResult &res) noexcept {
    QueryNetwork(device, network, {}, res);
}

void HeteroDeviceLoader::QueryNetwork(const std::string &device,
                                      const ICNNNetwork &network,
                                      const std::map<std::string, std::string>& config,
                                      QueryNetworkResult &res) noexcept {
    if (device != _deviceId) {
        res.rc = GENERAL_ERROR;
        std::string msg("Current HeteroDeviceLoader doesn't support device passed to QueryNetwork");
        snprintf(res.resp.msg, msg.size(), "%s", msg.c_str());
        return;
    }
    if (!_plugin) {
        res.rc = GENERAL_ERROR;
        std::string msg("No plugin, cannot execute QueryNetwork");
        snprintf(res.resp.msg, msg.size(), "%s", msg.c_str());
        return;
    }
    _plugin->QueryNetwork(network, config, res);
}

HeteroDeviceLoader::HeteroDeviceLoader(const std::string& deviceId) {
    _deviceId = deviceId;
    // try to create plugin
    PluginDispatcher dispatcher({ "" });
    _plugin = dispatcher.getPluginByDevice(_deviceId);
}

void HeteroDeviceLoader::initConfigs(const std::map<std::string, std::string> &config,
                 const std::vector<InferenceEngine::IExtensionPtr> &extensions) {
    if (_plugin) {
        if (_deviceId == "CPU") {
            for (auto &&ext : extensions) {
                _plugin->AddExtension(ext, nullptr);
            }
        }
        auto copyConfig = config;
        // preparing local version of configs which are supported by plugins
        for (auto c : copyConfig) {
            std::map<std::string, std::string> oneconfig;
            oneconfig[c.first] = c.second;
            try {
                _plugin->SetConfig(oneconfig, nullptr);
            } catch (InferenceEngine::details::InferenceEngineException &e) {
            }
        }
    }
}

