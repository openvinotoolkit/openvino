// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <string>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <unordered_set>

#include <ngraph/opsets/opset1.hpp>
#include <transformations/utils/utils.hpp>

#include <ie_metric_helpers.hpp>
#include <ie_performance_hints.hpp>
#include <threading/ie_executor_manager.hpp>
#include "multi_device_plugin.hpp"
#include <ie_algorithm.hpp>
#include <ie_icore.hpp>

// ------------------------------MultiDeviceInferencePlugin----------------------------
namespace MultiDevicePlugin {
    using namespace InferenceEngine;
namespace {

    std::string GetNetworkPrecision(const InferenceEngine::CNNNetwork &network) {
        auto nGraphFunc = network.getFunction();
        bool isINTModel = ngraph::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc);
        if (isINTModel) {
            return METRIC_VALUE(INT8);
        }
        for (auto & node : nGraphFunc->get_ordered_ops()) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
                std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
                std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
                std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
                auto layerType = node->input(1).get_element_type().get_type_name();
                if (layerType == "f32")
                    return METRIC_VALUE(FP32);
                if (layerType == "f16")
                    return METRIC_VALUE(FP16);
            }
        }
        return METRIC_VALUE(FP32);
    }

    std::map<std::string, std::string> mergeConfigs(std::map<std::string, std::string> config,
                                                    const std::map<std::string, std::string> & local) {
        for (auto && kvp : local) {
            config[kvp.first] = kvp.second;
        }
        return config;
    }
    std::vector<std::string> supported_configKeys = []() -> decltype(PerfHintsConfig::SupportedKeys()) {
                    auto res = PerfHintsConfig::SupportedKeys();
                    res.push_back(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
                    res.push_back(CONFIG_KEY_INTERNAL(MULTI_WORK_MODE_AS_AUTO));
                    return res;
                }();
}  // namespace

std::map<std::string, std::string> MultiDeviceInferencePlugin::GetSupportedConfig(
    const std::map<std::string, std::string> & config, const std::string & deviceName) const {
    std::vector<std::string> supportedConfigKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
    std::map<std::string, std::string> supportedConfig;
    for (auto&& key : supportedConfigKeys) {
        auto itKey = config.find(key);
        if (config.end() != itKey) {
            supportedConfig[key] = itKey->second;
        }
    }
    return supportedConfig;
}

std::vector<DeviceInformation> MultiDeviceInferencePlugin::ParseMetaDevices(const std::string& priorities,
                                                                          const std::map<std::string, std::string> & config) const {
    std::vector<DeviceInformation> metaDevices;

    // parsing the string and splitting to tokens
    std::vector<std::string> devicesWithRequests;
    // parsing the string and splitting the comma-separated tokens
    std::string::size_type i = 0;
    std::string::size_type idelimeter;
    while ((idelimeter = priorities.find(',', i)) != std::string::npos) {
        devicesWithRequests.push_back(priorities.substr(i, idelimeter - i));
        i = idelimeter + 1;
    }
    // last token in the string (which has no comma after that)
    devicesWithRequests.push_back(priorities.substr(i, priorities.length() - i));

    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        std::map<std::string, std::string> tconfig = mergeConfigs(_config, config);

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }

        return GetSupportedConfig(tconfig, deviceName);
    };

    auto getDefaultDeviceID = [this](std::string deviceName) -> std::string {
        std::vector<std::string> supportedMetrics = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS));
        if (std::find(supportedMetrics.begin(), supportedMetrics.end(), METRIC_KEY(SUPPORTED_CONFIG_KEYS)) != supportedMetrics.end()) {
            std::vector<std::string> supportKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));

            if (std::find(supportKeys.begin(), supportKeys.end(), CONFIG_KEY(DEVICE_ID)) != supportKeys.end()) {
                return GetCore()->GetConfig(deviceName, CONFIG_KEY(DEVICE_ID)).as<std::string>();
            }
        }

        return "";
    };

    for (auto && d : devicesWithRequests) {
        auto openingBracket = d.find_first_of('(');
        auto closingBracket = d.find_first_of(')', openingBracket);
        auto deviceName = d.substr(0, openingBracket);

        int numRequests = -1;
        if (closingBracket != std::string::npos && openingBracket < closingBracket) {
            numRequests = std::stol(d.substr(openingBracket + 1, closingBracket - 1));

            if (numRequests <= 0) {
                IE_THROW() << "Priority value for '" << deviceName << "' must be > 0, while " << numRequests
                    << "is passed";
            }
        }

        std::string defaultDeviceID = "";
        DeviceIDParser parsed{deviceName};
        if (parsed.getDeviceID().empty())
            defaultDeviceID = getDefaultDeviceID(deviceName);

        // create meta device
        metaDevices.push_back({ deviceName, getDeviceConfig(deviceName), numRequests, defaultDeviceID });
    }

    return metaDevices;
}

InferenceEngine::Parameter MultiDeviceInferencePlugin::GetConfig(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name)) {
        auto it = _config.find(name);
        if (it == _config.end()) {
            IE_THROW() << "Value for KEY_MULTI_DEVICE_PRIORITIES is not set";
        } else {
            return { it->second };
        }
    } else {
        IE_THROW() << "Unsupported config key: " << name;
    }
}

void MultiDeviceInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    const auto perf_hints_configs = PerfHintsConfig::SupportedKeys();
    for (auto && kvp : config) {
        const auto& name = kvp.first;
        if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name)) {
            if (std::find(perf_hints_configs.begin(), perf_hints_configs.end(), kvp.first) != perf_hints_configs.end())
                PerfHintsConfig::CheckConfigAndValue(kvp);
            _config[name] = kvp.second;
        } else {
            IE_THROW() << "Unsupported config key: " << name;
        }
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "MultiDevicePlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(MultiDeviceInferencePlugin, version)

MultiDeviceInferencePlugin::MultiDeviceInferencePlugin() {
    _pluginName = "MULTI";
}

InferenceEngine::Parameter MultiDeviceInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(FULL_DEVICE_NAME)) {
        std::string device_name = { GetName() };
        IE_SET_METRIC_RETURN(FULL_DEVICE_NAME, device_name);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, supported_configKeys);
    } else {
        IE_THROW() << "Unsupported metric key " << name;
    }
}

// Is called only when caching is enabled
IExecutableNetworkInternal::Ptr MultiDeviceInferencePlugin::LoadNetwork(const std::string& modelPath,
                                                                        const std::map<std::string, std::string>& config) {
    return LoadNetworkImpl(modelPath, {}, config);
}

IExecutableNetworkInternal::Ptr MultiDeviceInferencePlugin::LoadExeNetworkImpl(const CNNNetwork &network,
                                                                               const std::map<std::string, std::string>& config) {
    if (network.getFunction() == nullptr) {
        IE_THROW() << GetName() << " device supports just ngraph network representation";
    }

    auto networkPrecision = GetNetworkPrecision(network);
    return LoadNetworkImpl({}, network, config, networkPrecision);
}

IExecutableNetworkInternal::Ptr MultiDeviceInferencePlugin::LoadNetworkImpl(const std::string& modelPath,
                                                                              CNNNetwork network,
                                                                              const std::map<std::string, std::string>& config,
                                                                              const std::string &networkPrecision) {
    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with " << GetName() << " device via InferenceEngine::Core object";
    }

    if (modelPath.empty() && network.getFunction() == nullptr) {
        IE_THROW() << GetName() << " device supports just ngraph network representation";
    }

    auto fullConfig = mergeConfigs(_config, config);
    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, InferenceEngine::Parameter> multiNetworkConfig;
    std::vector<DeviceInformation> metaDevices;
    auto workMode = fullConfig.find(CONFIG_KEY_INTERNAL(MULTI_WORK_MODE_AS_AUTO));
    bool workModeAuto = workMode != fullConfig.end() && workMode->second == InferenceEngine::PluginConfigParams::YES;
    auto priorities = fullConfig.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);

    // if workMode is AUTO
    if (workModeAuto) {
        // check the configure and check if need to set PerfCounters configure to device
        // and set filter configure
        bool needPerfCounters = false;
        std::map<std::string, std::string> filterConfig;
        CheckConfig(fullConfig, needPerfCounters, filterConfig);
        // filter the device that supports filter configure
        auto strDevices = GetDeviceList(fullConfig);
        auto metaDevices = ParseMetaDevices(strDevices, fullConfig);
        auto supportDevices = FilterDevice(metaDevices, filterConfig);
        if (supportDevices.size() == 0) {
             IE_THROW() << "there is no device support the configure";
        }
        // replace the configure with configure that auto want to pass to device
        // and reset the strDevices to support devices
        auto validConfigKey = PerfHintsConfig::SupportedKeys();
        validConfigKey.push_back(PluginConfigParams::KEY_PERF_COUNT);
        validConfigKey.push_back(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS);
        strDevices = "";
        for (auto iter = supportDevices.begin(); iter != supportDevices.end(); iter++) {
             std::map<std::string, std::string> deviceConfig;
             auto& configs = iter->config;
             for (auto& config : configs) {
                 if (std::find(validConfigKey.begin(), validConfigKey.end(), config.first) != validConfigKey.end()) {
                     deviceConfig.insert({config.first, config.second});
                 }
             }
             iter->config = deviceConfig;
             strDevices += iter->deviceName;
             strDevices += ((iter + 1) == supportDevices.end()) ? "" : ",";
        }

        return std::make_shared<MultiDeviceExecutableNetwork>(modelPath, network, supportDevices, strDevices, this, needPerfCounters);
    }

    if (priorities == fullConfig.end()) {
        IE_THROW() << "KEY_MULTI_DEVICE_PRIORITIES key is not set for " << GetName() << " device";
    } else {  // for use case -d MULTI:xPU or -d AUTO:xPU
        metaDevices = ParseMetaDevices(priorities->second, fullConfig);
        multiNetworkConfig.insert(*priorities);
    }

    DeviceMap<SoExecutableNetworkInternal> executableNetworkPerDevice;
    std::mutex load_mutex;
    std::vector<Task> loads;
    std::once_flag readNetworkFlag;
    for (auto& p : metaDevices) {
        loads.push_back([&]() {
            const auto &deviceName = p.deviceName;
            const auto &deviceConfig = p.config;
            SoExecutableNetworkInternal exec_net;
            if (modelPath.empty()) {
                exec_net = GetCore()->LoadNetwork(network, deviceName, deviceConfig);
            } else if (GetCore()->DeviceSupportsImportExport(deviceName)) {
                exec_net = GetCore()->LoadNetwork(modelPath, deviceName, deviceConfig);
            } else {
                std::call_once(readNetworkFlag, [&]() {
                    network = GetCore()->ReadNetwork(modelPath, std::string());
                });
                exec_net = GetCore()->LoadNetwork(network, deviceName, deviceConfig);
            }
            std::unique_lock<std::mutex> lock{load_mutex};
            executableNetworkPerDevice.insert({deviceName, exec_net});
            multiNetworkConfig.insert(deviceConfig.begin(), deviceConfig.end());
        });
    }
    auto executor = InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor(
            IStreamsExecutor::Config{"MultiDeviceAsyncLoad",
                                     static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                                     1 /*single thread per stream*/,
                                     IStreamsExecutor::ThreadBindingType::NONE});
    executor->runAndWait(loads);
    if (executableNetworkPerDevice.empty())
        IE_THROW(NotFound) << "Failed to load network to any device "
                           <<  "that the " << GetName() << " device is initialized to work with";

    // checking the perf counters config from the loaded network to respect both device's plugin and load-specific setting
    size_t num_plugins_supporting_perf_counters = 0;
    for (auto n : executableNetworkPerDevice) {
            try {
                num_plugins_supporting_perf_counters +=
                        n.second->GetConfig(PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
                        PluginConfigParams::YES;
            } catch (...) {
            }
    }
    // MULTI can enable the perf counters only if all  devices support/enable that
    bool enablePerfCounters = num_plugins_supporting_perf_counters == executableNetworkPerDevice.size();
    auto impl = std::make_shared<MultiDeviceExecutableNetwork>(executableNetworkPerDevice,
                                                               metaDevices,
                                                               multiNetworkConfig,
                                                               enablePerfCounters);
    if (!modelPath.empty()) {
        SetExeNetworkInfo(impl,
                          executableNetworkPerDevice.begin()->second->GetInputsInfo(),
                          executableNetworkPerDevice.begin()->second->GetOutputsInfo());
        SetExeNetworkInfo(impl, executableNetworkPerDevice.begin()->second->GetExecGraphInfo());
    }
    return impl;
}

QueryNetworkResult MultiDeviceInferencePlugin::QueryNetwork(const CNNNetwork&                         network,
                                                            const std::map<std::string, std::string>& config) const {
    QueryNetworkResult queryResult;

    if (GetCore() == nullptr) {
        IE_THROW() << "Please, work with " << GetName() <<  " device via InferencEngine::Core object";
    }

    if (network.getFunction() == nullptr) {
        IE_THROW() << GetName() << " device supports just ngraph network representation";
    }

    queryResult.rc = StatusCode::OK;
    queryResult.supportedLayersMap.clear();

    auto fullConfig = mergeConfigs(_config, config);
    auto priorities = fullConfig.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (priorities == fullConfig.end()) {
        IE_THROW() << "KEY_MULTI_DEVICE_PRIORITIES key is not set for " << GetName() <<  " device";
    }
    auto metaDevices = ParseMetaDevices(priorities->second, fullConfig);
    std::unordered_set<std::string> supportedLayers;
    for (auto&& value : metaDevices) {
        auto deviceQr = GetCore()->QueryNetwork(network, value.deviceName, value.config);
        std::unordered_set<std::string> deviceSupportedLayers;
        for (auto&& layerQr : deviceQr.supportedLayersMap) {
            deviceSupportedLayers.emplace(layerQr.first);
        }
        supportedLayers = supportedLayers.empty()
                        ? deviceSupportedLayers : (deviceSupportedLayers.empty()
                        ? supportedLayers : InferenceEngine::details::Intersection(supportedLayers, deviceSupportedLayers));
    }
    for (auto&& supportedLayer : supportedLayers) {
        queryResult.supportedLayersMap[supportedLayer] = GetName();
    }
    return queryResult;
}

DeviceInformation MultiDeviceInferencePlugin::SelectDevice(const std::vector<DeviceInformation>& metaDevices, const std::string& networkPrecision) {
    if (metaDevices.empty()) {
        IE_THROW(NotFound) << "No available device to select in " << GetName() <<  " plugin";
    }
    if (metaDevices.size() == 1) {
        return metaDevices.at(0);
    }

    std::vector<DeviceInformation> CPU;
    std::vector<DeviceInformation> dGPU;
    std::vector<DeviceInformation> iGPU;
    std::vector<DeviceInformation> MYRIAD;
    std::vector<DeviceInformation> VPUX;

    for (auto& item : metaDevices) {
        if (item.deviceName.find("CPU") == 0) {
            CPU.push_back(item);
            continue;
        }
        if (item.deviceName.find("MYRIAD") == 0) {
            MYRIAD.push_back(item);
            continue;
        }
        if (item.deviceName.find("VPUX") == 0) {
            VPUX.push_back(item);
            continue;
        }
        if (item.deviceName.find("GPU") == 0) {
            auto gpuFullDeviceName = GetCore()->GetMetric(item.deviceName, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
            if (gpuFullDeviceName.find("iGPU") != std::string::npos) {
                iGPU.push_back(item);
            } else if (gpuFullDeviceName.find("dGPU") != std::string::npos) {
                dGPU.push_back(item);
            }
            continue;
        }
    }

    if (CPU.empty() && dGPU.empty() && iGPU.empty() && MYRIAD.empty() && VPUX.empty()) {
        IE_THROW(NotFound) << "No available device found";
    }

    // Priority of selecting device: dGPU > VPUX > iGPU > MYRIAD > CPU
    if (!dGPU.empty()) {
        for (auto&& item : dGPU) {
            std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            auto supportNetwork = std::find(capability.begin(), capability.end(), networkPrecision);
            if (supportNetwork != capability.end()) {
                return item;
            }
        }
    } else if (!VPUX.empty()) {
        for (auto&& item : VPUX) {
            std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            auto supportNetwork = std::find(capability.begin(), capability.end(), networkPrecision);
            if (supportNetwork != capability.end()) {
                return item;
            }
        }
    } else if (!iGPU.empty()) {
        for (auto&& item : iGPU) {
            std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            auto supportNetwork = std::find(capability.begin(), capability.end(), networkPrecision);
            if (supportNetwork != capability.end()) {
                return item;
            }
        }
    } else if (!MYRIAD.empty()) {
        for (auto&& item : MYRIAD) {
            std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
            auto supportNetwork = std::find(capability.begin(), capability.end(), networkPrecision);
            if (supportNetwork != capability.end()) {
                return item;
            }
        }
    }

    // If network is FP32 but there is no device support FP32, offload FP32 network to device support FP16.
    if (networkPrecision == "FP32") {
        if (!dGPU.empty()) {
            for (auto&& item : dGPU) {
                std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
                auto supportNetwork = std::find(capability.begin(), capability.end(), "FP16");
                if (supportNetwork != capability.end()) {
                    return item;
                }
            }
        } else if (!VPUX.empty()) {
            for (auto&& item : VPUX) {
                std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
                auto supportNetwork = std::find(capability.begin(), capability.end(), "FP16");
                if (supportNetwork != capability.end()) {
                    return item;
                }
            }
        } else if (!iGPU.empty()) {
            for (auto&& item : iGPU) {
                std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
                auto supportNetwork = std::find(capability.begin(), capability.end(), "FP16");
                if (supportNetwork != capability.end()) {
                    return item;
                }
            }
        } else if (!MYRIAD.empty()) {
            for (auto&& item : MYRIAD) {
                std::vector<std::string> capability = GetCore()->GetMetric(item.deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES));
                auto supportNetwork = std::find(capability.begin(), capability.end(), "FP16");
                if (supportNetwork != capability.end()) {
                    return item;
                }
            }
        }
    }

    if (CPU.empty()) {
        IE_THROW() << "Cannot select any device";
    }
    return CPU[0];
}

std::string MultiDeviceInferencePlugin::GetDeviceList(const std::map<std::string, std::string>& config) const {
    std::string allDevices;

    auto deviceListConfig = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (deviceListConfig == config.end()) {
        auto deviceList = GetCore()->GetAvailableDevices();
        for (auto&& device : deviceList) {
            allDevices += device;
            allDevices += ((device == deviceList[deviceList.size()-1]) ? "" : ",");
        }
    } else {
        allDevices = deviceListConfig->second;
    }

    if (allDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }

    return allDevices;
}

void MultiDeviceInferencePlugin::CheckConfig(const std::map<std::string, std::string>& config,
        bool& needPerfCounters, std::map<std::string, std::string>& filterConfig) {
    // TODO need to optimize this code, too much duplicated code
    const auto perf_hints_configs = PerfHintsConfig::SupportedKeys();
    for (auto&& kvp : config) {
        if (kvp.first.find("AUTO_") == 0) {
            continue;
        } else if (kvp.first == PluginConfigParams::KEY_PERF_COUNT) {
            if (kvp.second == PluginConfigParams::YES) {
                needPerfCounters = true;
                filterConfig.insert({kvp.first, kvp.second});
            } else if (kvp.second == PluginConfigParams::NO) {
                needPerfCounters = false;
            } else {
                IE_THROW() << "Unsupported config value: " << kvp.second
                           << " for key: " << kvp.first;
            }
        } else if (kvp.first == PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) {
            if (kvp.second == PluginConfigParams::YES ||
                kvp.second == PluginConfigParams::NO) {
                continue;
            } else {
                IE_THROW() << "Unsupported config value: " << kvp.second
                           << " for key: " << kvp.first;
            }
        } else if (std::find(perf_hints_configs.begin(), perf_hints_configs.end(), kvp.first) != perf_hints_configs.end()) {
            PerfHintsConfig::CheckConfigAndValue(kvp);
        } else if (supported_configKeys.end() == std::find(supported_configKeys.begin(), supported_configKeys.end(), kvp.first)) {
            IE_THROW() << "Unsupported config key: " << kvp.first;
        }
    }
}

std::vector<DeviceInformation> MultiDeviceInferencePlugin::FilterDevice(const std::vector<DeviceInformation>& metaDevices,
        const std::map<std::string, std::string>& config) {
    if (metaDevices.empty()) {
        IE_THROW(NotFound) << "No available device to filter " << GetName() <<  " plugin";
    }

    if (config.size() == 0) {
        return metaDevices;
    }

    std::vector<DeviceInformation> filterDevice;
    for (auto&& item : metaDevices) {
        bool support = true;
        std::vector<std::string> supportedMetrics = GetCore()->GetMetric(item.deviceName, METRIC_KEY(SUPPORTED_METRICS));
        if (std::find(supportedMetrics.begin(), supportedMetrics.end(), METRIC_KEY(SUPPORTED_CONFIG_KEYS)) != supportedMetrics.end()) {
            std::vector<std::string> supportKeys = GetCore()->GetMetric(item.deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS));
            for (auto&& kvp : config) {
                auto targetKey = std::find(supportKeys.begin(), supportKeys.end(), kvp.first);
                // if device have the key, we think the device support it
                if (targetKey != supportKeys.end()) {
                    continue;
                } else {
                    support = false;
                    break;
                }
            }
        } else {
            support = false;
        }

        if (support) {
            filterDevice.push_back(item);
        }
    }
    return filterDevice;
}
}  // namespace MultiDevicePlugin
