// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>
#include <tuple>
#include <utility>

#include <ie_metric_helpers.hpp>
#include <inference_engine.hpp>
#include <cnn_network_ngraph_impl.hpp>
#include <cpp_interfaces/base/ie_plugin_base.hpp>
#include <cpp_interfaces/impl/ie_executable_network_internal.hpp>

#include <vpu/vpu_plugin_config.hpp>
#include <vpu/parsed_config.hpp>
#include <vpu/utils/profiling.hpp>
#include <vpu/utils/error.hpp>

#include "myriad_plugin.h"

using namespace InferenceEngine;
using namespace InferenceEngine::PluginConfigParams;
using namespace InferenceEngine::VPUConfigParams;
using namespace vpu::MyriadPlugin;

ExecutableNetworkInternal::Ptr Engine::LoadExeNetworkImpl(
        const ICore* /*core*/,
        ICNNNetwork& network,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(LoadExeNetworkImpl);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    return std::make_shared<ExecutableNetwork>(network, _devicePool, parsedConfigCopy);
}

void Engine::SetConfig(const std::map<std::string, std::string> &config) {
    _parsedConfig.update(config);

    for (const auto& entry : config) {
        _config[entry.first] = entry.second;
    }
}

Parameter Engine::GetConfig(const std::string& name, const std::map<std::string, Parameter>& options) const {
    auto supported_keys = _metrics->SupportedConfigKeys();
    if (std::find(supported_keys.begin(),
        supported_keys.end(), name) == supported_keys.end()) {
        THROW_IE_EXCEPTION << "Unsupported config key : " << name;
    }

    Parameter result;
    auto option = _config.find(name);
    if (option != _config.end())
        result = option->second;

    return result;
}

void Engine::QueryNetwork(
        const ICNNNetwork& network,
        const std::map<std::string, std::string>& config,
        QueryNetworkResult& res) const {
    VPU_PROFILE(QueryNetwork);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config);

    const auto log = std::make_shared<Logger>(
        "GraphCompiler",
        parsedConfigCopy.logLevel(),
        defaultOutput(parsedConfigCopy.compilerLogFilePath()));

    const auto layerNames = getSupportedLayers(
        network,
        static_cast<Platform>(parsedConfigCopy.platform()),
        parsedConfigCopy.compileConfig(),
        log);

    for (const auto& layerName : layerNames) {
        res.supportedLayersMap.insert({ layerName, GetName() });
    }
}

Engine::Engine(std::shared_ptr<IMvnc> mvnc) :
        _mvnc(std::move(mvnc)),
        _metrics(std::make_shared<MyriadMetrics>()) {
    if (!_mvnc) {
        THROW_IE_EXCEPTION << "mvnc is invalid";
    }

    _pluginName = "MYRIAD";
}

InferenceEngine::ExecutableNetwork Engine::ImportNetwork(
        std::istream& model,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    auto parsedConfigCopy = _parsedConfig;
    parsedConfigCopy.update(config, ConfigMode::RunTime);

    const auto executableNetwork =
            std::make_shared<ExecutableNetwork>(
                model, _devicePool, parsedConfigCopy);

    return InferenceEngine::ExecutableNetwork{IExecutableNetwork::Ptr(
        new ExecutableNetworkBase<ExecutableNetworkInternal>(executableNetwork),
        [](ie::details::IRelease *p) {p->Release();})};
}

IExecutableNetwork::Ptr Engine::ImportNetwork(
        const std::string& modelFileName,
        const std::map<std::string, std::string>& config) {
    VPU_PROFILE(ImportNetwork);

    std::ifstream blobFile(modelFileName, std::ios::binary);

    if (!blobFile.is_open()) {
        THROW_IE_EXCEPTION << ie::details::as_status << NETWORK_NOT_READ;
    }

    return ImportNetwork(blobFile, config);
}

InferenceEngine::Parameter Engine::GetMetric(const std::string& name,
                                     const std::map<std::string, InferenceEngine::Parameter> & options) const {
    const auto mvnc = _mvnc;
    const auto metrics = _metrics;
    const auto devicePool = _devicePool;
    const auto getSpecifiedDeviceName = [&mvnc, &metrics, &devicePool, &options]() {
        if (options.count(KEY_DEVICE_ID)) {
            return options.at(KEY_DEVICE_ID).as<std::string>();
        }

        const auto availableDevices = metrics->AvailableDevicesNames(mvnc, devicePool);
        VPU_THROW_UNLESS(!availableDevices.empty(), "No devices available.");
        VPU_THROW_UNLESS(availableDevices.size() == 1, "KEY_DEVICE_ID is undefined.");

        return availableDevices.front();
    };
    const auto getDeviceByName = [&devicePool](const std::string& deviceName) {
        const auto deviceIt = std::find_if(
                devicePool.begin(), devicePool.end(), [&deviceName](DevicePtr device) {
                    return device->_name == deviceName;
                });
        if (deviceIt == devicePool.end()) {
            return DevicePtr();
        }
        return *deviceIt;
    };

    if (name == METRIC_KEY(AVAILABLE_DEVICES)) {
        IE_SET_METRIC_RETURN(AVAILABLE_DEVICES, _metrics->AvailableDevicesNames(_mvnc, _devicePool));
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, _metrics->FullName(getSpecifiedDeviceName()));
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, _metrics->SupportedMetrics());
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, _metrics->SupportedConfigKeys());
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, _metrics->OptimizationCapabilities());
    } else if (name == METRIC_KEY(RANGE_FOR_ASYNC_INFER_REQUESTS)) {
        IE_SET_METRIC_RETURN(RANGE_FOR_ASYNC_INFER_REQUESTS, _metrics->RangeForAsyncInferRequests(_config));
    } else if (name == METRIC_KEY(DEVICE_THERMAL)) {
        const auto& device = getDeviceByName(getSpecifiedDeviceName());
        if (device != nullptr) {
            IE_SET_METRIC_RETURN(DEVICE_THERMAL, _metrics->DevicesThermal(device));
        } else {
            return Parameter();
        }
    }
    THROW_IE_EXCEPTION << NOT_IMPLEMENTED_str;
}

std::shared_ptr<ICNNNetwork> Engine::ConvertAndCloneNetwork(ICNNNetwork& network) {
    if (auto networkNGraph = dynamic_cast<InferenceEngine::details::CNNNetworkNGraphImpl*>(&network)) {
        auto networkClone = networkNGraph->cloneNGraphImpl();
        return std::shared_ptr<ICNNNetwork>(networkClone);
    }
    return CloneNetwork(network);
}
