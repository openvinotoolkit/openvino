// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpp_interfaces/base/ie_inference_plugin_api.hpp"
#include "hetero_device_loader.hpp"
#include "ie_plugin_dispatcher.hpp"
#include "ie_core.hpp"
#include <string>
#include <cstdio>
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
    if (DeviceIDParser(device).getDeviceName() != _deviceName) {
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

    if (!_deviceId.empty()) {
        tconfig.insert({ CONFIG_KEY(DEVICE_ID), _deviceId });
    }

    return _plugin->LoadNetwork(ret, network, tconfig, resp);
}

void HeteroDeviceLoader::QueryNetwork(const std::string &device,
                                      const ICNNNetwork &network,
                                      const std::map<std::string, std::string>& config,
                                      QueryNetworkResult &res) noexcept {
    if (DeviceIDParser(device).getDeviceName() != _deviceName) {
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

    auto tconfig = config;
    if (!_deviceId.empty()) {
        tconfig.insert({ CONFIG_KEY(DEVICE_ID), _deviceId });
    }

    _plugin->QueryNetwork(network, tconfig, res);
}

HeteroDeviceLoader::HeteroDeviceLoader(const std::string& deviceId, const ICore * core) {
    DeviceIDParser parser(deviceId);

    _deviceName = parser.getDeviceName();
    _deviceId = parser.getDeviceID();
    _core = core;

    if (_core != nullptr) {
        _plugin = _core->GetPluginByName(_deviceName);
    } else {
        IE_SUPPRESS_DEPRECATED_START
        // try to create plugin
        PluginDispatcher dispatcher({ "" });
        _plugin = dispatcher.getPluginByDevice(_deviceName);
        IE_SUPPRESS_DEPRECATED_END
    }
}

HeteroDeviceLoader::~HeteroDeviceLoader() {
}

void HeteroDeviceLoader::initConfigs(const std::map<std::string, std::string> &config,
                 const std::vector<InferenceEngine::IExtensionPtr> &extensions) {
    if (_plugin) {
        try {
            for (auto &&ext : extensions) {
                _plugin->AddExtension(ext, nullptr);
            }
        }
        catch (InferenceEngine::details::InferenceEngineException &) {
            // ignore if plugin does not support extensions
        }

        auto copyConfig = config;
        // preparing local version of configs which are supported by plugins
        for (auto c : copyConfig) {
            std::map<std::string, std::string> oneconfig;
            oneconfig[c.first] = c.second;
            try {
                _plugin->SetConfig(oneconfig, nullptr);
            } catch (InferenceEngine::details::InferenceEngineException &) {
            }
        }
    }
}

void HeteroDeviceLoader::SetLogCallback(IErrorListener &listener) {
    _plugin->SetLogCallback(listener);
}
