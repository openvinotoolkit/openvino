// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "ie_plugin_dispatcher.hpp"
#include "hetero_plugin.hpp"
#include "ie_util_internal.hpp"
#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include <fstream>
#include <unordered_set>
#include "ie_plugin_config.hpp"
#include "hetero/hetero_plugin_config.hpp"
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include "hetero_executable_network.hpp"
#include "cpp_interfaces/base/ie_inference_plugin_api.hpp"

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
    _config[HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)] = NO;
}

InferenceEngine::ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(const ICore*                     /*core*/,
                                                                           const InferenceEngine::ICNNNetwork&    network,
                                                                           const Configs&                   config) {
    // TODO(amalyshe) do we need here verification of input precisions?
    Configs tconfig;
    tconfig = config;

    // we must not override the parameter, but need to copy everything from plugin config
    for (auto && c : _config) {
        if (tconfig.find(c.first) == tconfig.end()) {
            tconfig[c.first] = c.second;
        }
    }

    return std::make_shared<HeteroExecutableNetwork>(*cloneNet(network), tconfig, this);
}

ExecutableNetwork Engine::ImportNetworkImpl(std::istream& heteroModel, const Configs& config) {
    Configs tconfig;
    tconfig = config;

    // we must not override the parameter, but need to copy everything from plugin config
    for (auto && c : _config) {
        if (tconfig.find(c.first) == tconfig.end()) {
            tconfig[c.first] = c.second;
        }
    }

    IExecutableNetwork::Ptr executableNetwork;
    // Use config provided by an user ignoring default config
    executableNetwork.reset(new ExecutableNetworkBase<ExecutableNetworkInternal>(
                                std::make_shared<HeteroExecutableNetwork>(heteroModel, tconfig, this)),
                            [](InferenceEngine::details::IRelease *p) {p->Release();});

    return ExecutableNetwork{executableNetwork};
}

namespace  {

IE_SUPPRESS_DEPRECATED_START

IInferencePluginAPI * getInferencePluginAPIInterface(IInferencePlugin * iplugin) {
    return dynamic_cast<IInferencePluginAPI *>(iplugin);
}

IInferencePluginAPI * getInferencePluginAPIInterface(InferenceEnginePluginPtr iplugin) {
    return getInferencePluginAPIInterface(static_cast<IInferencePlugin *>(iplugin.operator->()));
}

IInferencePluginAPI * getInferencePluginAPIInterface(InferencePlugin plugin) {
    return getInferencePluginAPIInterface(static_cast<InferenceEnginePluginPtr>(plugin));
}

}  // namespace

Engine::Configs Engine::GetSupportedConfig(const Engine::Configs& globalConfig,
                                           const Engine::Configs& localConfig,
                                           const InferenceEngine::InferencePlugin& plugin) {
    auto pluginApi = getInferencePluginAPIInterface(plugin);
    std::vector<std::string> supportedConfigKeys = pluginApi->GetMetric(METRIC_KEY(SUPPORTED_CONFIG_KEYS), {});
    Engine::Configs supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = localConfig.find(key);
        if (localConfig.end() != itKey) {
            supportedConfig[key] = itKey->second;
        } else {
            itKey = globalConfig.find(key);
            if (globalConfig.end() != itKey) {
                supportedConfig[key] = itKey->second;
            }
        }
    }
    return supportedConfig;
}

Engine::PluginEntry Engine::GetDevicePlugin(const std::string& deviceWithID) const {
    InferenceEngine::InferencePlugin plugin;
    DeviceIDParser deviceParser(deviceWithID);
    std::string deviceName = deviceParser.getDeviceName();

    if (nullptr == _core) {
        IE_SUPPRESS_DEPRECATED_START
        // try to create plugin
        PluginDispatcher dispatcher({file_name_t()});
        plugin = dispatcher.getPluginByDevice(deviceName);
        IE_SUPPRESS_DEPRECATED_END
    } else {
        plugin = InferencePlugin{_core->GetPluginByName(deviceName)};
    }

    try {
        for (auto&& ext : _extensions) {
            plugin.AddExtension(ext);
        }
    } catch (InferenceEngine::details::InferenceEngineException &) {}

    Configs pluginConfig = GetSupportedConfig(_config, {}, plugin);

    // set device ID if any
    std::string deviceIDLocal = deviceParser.getDeviceID();
    if (!deviceIDLocal.empty()) {
        pluginConfig = GetSupportedConfig(pluginConfig, { { KEY_DEVICE_ID, deviceIDLocal } }, plugin);
    }

    return { plugin, pluginConfig };
}

IE_SUPPRESS_DEPRECATED_END

Engine::Plugins Engine::GetDevicePlugins(const std::string& targetFallback) const {
    auto devices = InferenceEngine::DeviceIDParser::getHeteroDevices(targetFallback);
    Engine::Plugins plugins = _plugins;
    for (auto&& device : devices) {
        auto itPlugin = plugins.find(device);
        if (plugins.end() == itPlugin) {
            IE_SUPPRESS_DEPRECATED_START
            plugins[device] = GetDevicePlugin(device);
            IE_SUPPRESS_DEPRECATED_END
        }
    }
    return plugins;
}

Engine::Plugins Engine::GetDevicePlugins(const std::string& targetFallback) {
    _plugins = const_cast<const Engine*>(this)->GetDevicePlugins(targetFallback);
    return _plugins;
}

void Engine::SetConfig(const Configs &configs) {
    for (auto&& config : configs) {
        _config[config.first] = config.second;
    }

    for (auto&& plugin : _plugins) {
        plugin.second._config = GetSupportedConfig(plugin.second._config, configs, plugin.second._ref);
    }
}

void Engine::AddExtension(InferenceEngine::IExtensionPtr extension) {
    _extensions.emplace_back(extension);
    try {
        for (auto&& plugin : _plugins) {
            IE_SUPPRESS_DEPRECATED_START
            plugin.second._ref.AddExtension(extension);
            IE_SUPPRESS_DEPRECATED_END
        }
    } catch (InferenceEngine::details::InferenceEngineException &) {}
}

HeteroLayerColorer::HeteroLayerColorer(const std::vector<std::string>& devices) {
    static const std::vector<std::string> colors = {"#5A5DF0", "#20F608", "#F1F290", "#11F110"};
    for (auto&& device : devices) {
        deviceColorMap[device] = colors[std::distance(&device, devices.data()) % colors.size()];
    }
}

void HeteroLayerColorer::operator()(const CNNLayerPtr layer,
                ordered_properties &printed_properties,
                ordered_properties &node_properties) {
    auto device = layer->affinity;
    printed_properties.insert(printed_properties.begin(), std::make_pair("device", device));
    node_properties.emplace_back("fillcolor", deviceColorMap[device]);
}

void Engine::SetAffinity(InferenceEngine::ICNNNetwork &network, const Configs &config) {
    Configs tconfig = _config;
    for (auto && value : config) {
        tconfig[value.first] = value.second;
    }

    auto it = tconfig.find("TARGET_FALLBACK");
    if (it == tconfig.end()) {
        THROW_IE_EXCEPTION << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
    }

    GetDevicePlugins(it->second);
    QueryNetworkResult qr;
    QueryNetwork(network, tconfig, qr);

    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        auto it = qr.supportedLayersMap.find(layer->name);
        if (it != qr.supportedLayersMap.end()) {
            layer->affinity = it->second;
        }
        i++;
    }

    if (YES == tconfig[HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)]) {
        std::unordered_set<std::string> devicesSet;
        details::CNNNetworkIterator i(&network);
        while (i != details::CNNNetworkIterator()) {
            CNNLayer::Ptr layer = *i;
            if (!layer->affinity.empty()) {
                devicesSet.insert(layer->affinity);
            }
            i++;
        }
        std::vector<std::string> devices{std::begin(devicesSet), std::end(devicesSet)};
        std::stringstream stream(std::stringstream::out);
        stream << "hetero_affinity_" << network.getName() << ".dot";

        std::ofstream file(stream.str());

        saveGraphToDot(network, file, HeteroLayerColorer{devices});
    }
}

void Engine::QueryNetwork(const ICNNNetwork &network, const Configs& config, QueryNetworkResult &qr) const {
    auto it = config.find("TARGET_FALLBACK");
    if (it == config.end()) {
        it = _config.find("TARGET_FALLBACK");

        if (it == _config.end()) {
            THROW_IE_EXCEPTION << "The 'TARGET_FALLBACK' option was not defined for heterogeneous plugin";
        }
    }

    Plugins plugins = GetDevicePlugins(it->second);

    qr.rc = StatusCode::OK;

    std::map<std::string, QueryNetworkResult> queryResults;
    // go over devices, create appropriate plugins and
    for (auto&& value : plugins) {
        auto& device = value.first;
        auto& plugin = value.second;
        QueryNetworkResult r;
        IE_SUPPRESS_DEPRECATED_START
        plugin._ref.QueryNetwork(network, GetSupportedConfig(plugin._config, config, plugin._ref), r);
        IE_SUPPRESS_DEPRECATED_END
        queryResults[device] = r;
    }

    //  WARNING: Here is devices with user set priority
    auto falbackDevices = InferenceEngine::DeviceIDParser::getHeteroDevices(it->second);

    details::CNNNetworkIterator i(&network);
    while (i != details::CNNNetworkIterator()) {
        CNNLayer::Ptr layer = *i;
        for (auto&& device : falbackDevices) {
            auto& deviceQueryResult = queryResults[device];
            if (deviceQueryResult.supportedLayersMap.find(layer->name) != deviceQueryResult.supportedLayersMap.end()) {
                qr.supportedLayersMap[layer->name] = device;
                break;
            }
        }
        i++;
    }
}

Parameter Engine::GetMetric(const std::string& name, const std::map<std::string, Parameter> & /*options*/) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, std::vector<std::string>{
            METRIC_KEY(SUPPORTED_METRICS),
            METRIC_KEY(FULL_DEVICE_NAME),
            METRIC_KEY(SUPPORTED_CONFIG_KEYS)});
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, std::vector<std::string>{
            HETERO_CONFIG_KEY(DUMP_GRAPH_DOT),
            "TARGET_FALLBACK",
            CONFIG_KEY(EXCLUSIVE_ASYNC_REQUESTS)});
    } else if (METRIC_KEY(FULL_DEVICE_NAME) == name) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, std::string{"HETERO"});
    } else {
        THROW_IE_EXCEPTION << "Unsupported Plugin metric: " << name;
    }
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter> & /*options*/) const {
    if (name == HETERO_CONFIG_KEY(DUMP_GRAPH_DOT)) {
        auto it = _config.find(HETERO_CONFIG_KEY(DUMP_GRAPH_DOT));
        IE_ASSERT(it != _config.end());
        bool dump = it->second == YES;
        return { dump };
    } else {
        THROW_IE_EXCEPTION << "Unsupported config key: " << name;
    }
}

IE_SUPPRESS_DEPRECATED_START

INFERENCE_PLUGIN_API(InferenceEngine::StatusCode) CreatePluginEngine(
        InferenceEngine::IInferencePlugin *&plugin,
        InferenceEngine::ResponseDesc *resp) noexcept {
    try {
        plugin = make_ie_compatible_plugin({{2, 1}, CI_BUILD_NUMBER, "heteroPlugin"},
                                           std::make_shared<Engine>());
        return OK;
    }
    catch (std::exception &ex) {
        return DescriptionBuffer(GENERAL_ERROR, resp) << ex.what();
    }
}
