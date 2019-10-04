// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "hetero_plugin.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "hetero_plugin_base.hpp"
#include "hetero_executable_network.hpp"
#include "hetero_fallback_policy.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::HeteroConfigParams;
using namespace HeteroPlugin;
using namespace std;

static Version heteroPluginDescription = {
        {2, 1},  // plugin API version
        CI_BUILD_NUMBER,
        "heteroPlugin"  // plugin description message
};

void Engine::GetVersion(const Version *&versionInfo)noexcept {
    versionInfo = &heteroPluginDescription;
}

Engine::Engine() {
    _pluginName = "HETERO";
    _config[InferenceEngine::PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = "YES";
    _config[KEY_HETERO_DUMP_GRAPH_DOT] = NO;
}

InferenceEngine::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ICore * core, InferenceEngine::ICNNNetwork &network,
                                                                           const std::map<std::string, std::string> &config) {
    // TODO(amalyshe) do we need here verification of input precisions?
    std::map<std::string, std::string> tconfig;
    tconfig = config;

    // we must not override the parameter, but need to copy everything from plugin config
    for (auto && c : _config) {
        if (tconfig.find(c.first) == tconfig.end()) {
            tconfig[c.first] = c.second;
        }
    }

    return std::make_shared<HeteroExecutableNetwork>(network, core, tconfig, _extensions, _deviceLoaders, error_listener);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    if (_config.find("TARGET_FALLBACK") == _config.end()) {
        _config["TARGET_FALLBACK"] = "";
    }

    for (auto &&i : config) {
        _config[i.first] = i.second;
    }
}

IE_SUPPRESS_DEPRECATED_START
void Engine::SetDeviceLoader(const std::string &device,
                             IHeteroDeviceLoader::Ptr pLoader) {
    _deviceLoaders[device] = pLoader;
}
IE_SUPPRESS_DEPRECATED_END

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    _extensions.push_back(extension);
}

void Engine::SetAffinity(InferenceEngine::ICNNNetwork &network,
                         const std::map<std::string, std::string> &config) {
    FallbackPolicy fbPolicy(_deviceLoaders, _config[KEY_HETERO_DUMP_GRAPH_DOT] == YES, GetCore());
    fbPolicy.init(_config["TARGET_FALLBACK"], config, _extensions);
    fbPolicy.setAffinity(fbPolicy.getAffinities(config, network), network);
}

void Engine::SetLogCallback(IErrorListener &listener) {
    error_listener = &listener;

    IE_SUPPRESS_DEPRECATED_START
    for (auto& device_loader : _deviceLoaders)
        device_loader.second->SetLogCallback(*error_listener);
    IE_SUPPRESS_DEPRECATED_END
}

void Engine::QueryNetwork(const ICNNNetwork &network, const std::map<std::string, std::string>& config, QueryNetworkResult &res) const {
    auto _deviceLoaders_ = _deviceLoaders;

    auto it = _config.find(KEY_HETERO_DUMP_GRAPH_DOT);
    IE_ASSERT(it !=  _config.end());
    FallbackPolicy fbPolicy(_deviceLoaders_, it->second == YES, GetCore());
    it = config.find("TARGET_FALLBACK");
    if (it == config.end()) {
        it = _config.find("TARGET_FALLBACK");

        if (it == _config.end()) {
            THROW_IE_EXCEPTION << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
        }
    }
    fbPolicy.init(it->second, config, _extensions);
    res = fbPolicy.getAffinities(config, network);
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter> & options) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{
            HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
            "TARGET_FALLBACK",
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)});
    } else {
        THROW_IE_EXCEPTION << "Unsupported Plugin metric: " << name;
    }
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter> & options) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        auto it = _config.find(KEY_HETERO_DUMP_GRAPH_DOT);
        IE_ASSERT(it != _config.end());
        bool dump = it->second == YES;
        return { dump };
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key: " << name;
    }
}

namespace HeteroPlugin {

InferenceEngine::StatusCode CreateHeteroPluginEngine(
        InferenceEngine::IInferencePlugin *&plugin,
        InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        plugin = new HeteroPluginBase<Engine>(
                {{2, 1}, "heteroPlugin", "heteroPlugin"},
                std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}

}  // namespace HeteroPlugin

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode) CreatePluginEngine(
        InferenceEngine::IInferencePlugin *&plugin,
        InferenceEngine::ResponseDesc *resp) noexcept {
    return HeteroPlugin::CreateHeteroPluginEngine(plugin, resp);
}
