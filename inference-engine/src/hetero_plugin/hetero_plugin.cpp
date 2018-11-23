// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0
//

#include "hetero_plugin.h"
#include <memory>
#include <vector>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "hetero_plugin_base.hpp"
#include "inference_engine.hpp"
#include "hetero_executable_network.h"
#include "fallback_policy.h"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;
using namespace HeteroPlugin;
using namespace std;

static Version heteroPluginDescription = {
        {1, 4},  // plugin API version
        CI_BUILD_NUMBER,
        "dliaPlugin"  // plugin description message -
};

void Engine::GetVersion(const Version *&versionInfo)noexcept {
    versionInfo = &heteroPluginDescription;
}


Engine::Engine() {
    _config[InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = "YES";
    _config[KEY_HETERO_DUMP_GRAPH_DOT] = NO;
}

InferenceEngine::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(InferenceEngine::ICNNNetwork &network,
                                                                           const std::map<std::string, std::string> &config) {
    std::map<std::string, std::string> tconfig;
    tconfig = config;

    // we must not override the parameter, but need to copy everything from plugin config
    for (auto c : _config) {
        if (tconfig.find(c.first) == tconfig.end()) {
            tconfig[c.first] = c.second;
        }
    }

    return std::make_shared<HeteroExecutableNetwork>(network, tconfig, _extensions, _deviceLoaders, error_listener);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    if (_config.find("TARGET_FALLBACK") == _config.end()) {
        _config["TARGET_FALLBACK"] = "";
    }

    for (auto &&i : config) {
        _config[i.first] = i.second;
    }
}

void Engine::SetDeviceLoader(const std::string &device,
                             IHeteroDeviceLoader::Ptr pLoader) {
    _deviceLoaders[device] = pLoader;
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    _extensions.push_back(extension);
}

void Engine::SetAffinity(InferenceEngine::ICNNNetwork &network,
                         const std::map<std::string, std::string> &config) {
    FallbackPolicy fbPolicy(_deviceLoaders, _config[KEY_HETERO_DUMP_GRAPH_DOT]== YES);
    fbPolicy.init(_config["TARGET_FALLBACK"], config, _extensions);
    fbPolicy.setAffinity(config, network);
}


INFERENCE_PLUGIN_API(StatusCode) CreatePluginEngine(
        IInferencePlugin *&plugin,
        ResponseDesc *resp) noexcept {
    try {
        plugin = new HeteroPluginBase<Engine>(
                {{1, 4}, "heteroPlugin", "heteroPlugin"},
                std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

void Engine::SetLogCallback(IErrorListener &listener) {
    error_listener = &listener;
    for (auto& device_loader : _deviceLoaders)
        device_loader.second->SetLogCallback(*error_listener);
}
