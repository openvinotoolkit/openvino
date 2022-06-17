// Copyright (C) 2018-2022 Intel Corporation
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
#include "openvino/runtime/intel_auto/properties.hpp"
#include "plugin.hpp"
#include <ie_algorithm.hpp>
#include <ie_icore.hpp>
#include <ie_ngraph_utils.hpp>
#include "bind_multi_schedule.hpp"
#include "multi_executable_network.hpp"
#include "auto_schedule.hpp"
#include "auto_executable_network.hpp"

#include "itt.hpp"
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
                    res.push_back(ov::device::priorities.name());
                    res.push_back(CONFIG_KEY_INTERNAL(MULTI_WORK_MODE_AS_AUTO));
                    res.push_back(ov::enable_profiling.name());
                    res.push_back(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS);
                    res.push_back(ov::hint::model_priority.name());
                    res.push_back(ov::hint::allow_auto_batching.name());
                    res.push_back(ov::log::level.name());
                    res.push_back(ov::intel_auto::device_bind_buffer.name());
                    return res;
                }();
}  // namespace


std::mutex MultiDeviceInferencePlugin::_mtx;
std::map<unsigned int, std::list<std::string>> MultiDeviceInferencePlugin::_priorityMap;

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

        return GetCore()->GetSupportedConfig(deviceName, tconfig);
    };

    auto getDefaultDeviceID = [this](std::string deviceName) -> std::string {
        auto supportedMetrics = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
        if (std::find(supportedMetrics.begin(), supportedMetrics.end(), METRIC_KEY(SUPPORTED_CONFIG_KEYS)) != supportedMetrics.end()) {
            auto supportKeys = GetCore()->GetMetric(deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();

            if (std::find(supportKeys.begin(), supportKeys.end(), CONFIG_KEY(DEVICE_ID)) != supportKeys.end()) {
                return GetCore()->GetConfig(deviceName, CONFIG_KEY(DEVICE_ID)).as<std::string>();
            }
        }

        return "";
    };
    auto checkPriorityConfig = [&] (const std::string& priString) {
        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        while ((endpos = priString.find(",", pos)) != std::string::npos) {
            auto subStr = priString.substr(pos, endpos - pos);
            if (subStr.find("-") != 0)
                return true;
            pos = endpos + 1;
        }
        if (priString.substr(pos, priString.length() - pos).find("-") != 0 )
            return true;
        return false;
    };
    unsigned int devicePriority = 0;
    auto prioritiesIter = config.find(ov::device::priorities.name());
    // if AUTO:-***,-***...., also do not need to enable device priority
    bool enableDevicePriority = (prioritiesIter != config.end()) &&
                                checkPriorityConfig(prioritiesIter->second);

    auto deviceList = GetCore()->GetAvailableDevices();
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

        DeviceIDParser parsed{deviceName};
        std::string deviceid = parsed.getDeviceID();
        std::vector<std::string> sameTypeDevices;
        // if AUTO:GPU case, replace GPU with GPU.0 and GPU.1
        // Disable AUTO:MYRIAD here because of below test case
        // MYRIAD/CoreThreadingTests.smoke_QueryNetwork/targetDevice=MULTI_config=MULTI_DEVICE_PRIORITIES:MYRIAD_
        // faild on windows
        // the error is
        // myriadFuncTests-0 INFO: [E:] [BSL] found 0 ioexpander device
        if (deviceid.empty() && deviceName.find("MYRIAD") == std::string::npos) {
            for (auto&& device : deviceList) {
                if (device.find(deviceName) != std::string::npos) {
                    sameTypeDevices.push_back(std::move(device));
                }
            }
        }
        // it's a virtual device like HETERO, TEMPLATE
        // or real device with ID like GPU.1
        if (sameTypeDevices.size() == 0) {
            sameTypeDevices.push_back(std::move(deviceName));
        }

        for (auto&& deviceNameWithID : sameTypeDevices) {
            DeviceIDParser newParsed{deviceNameWithID};
            std::string defaultDeviceID = "";
            if (newParsed.getDeviceID().empty()) {
                defaultDeviceID = getDefaultDeviceID(deviceNameWithID);
            } else {
                defaultDeviceID = newParsed.getDeviceID();
            }

            std::string fullDeviceName = "";
            std::string uniqueName = "";
            if (newParsed.getDeviceName() == "GPU") {
                auto supportedMetrics = GetCore()->GetMetric(deviceNameWithID, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
                if (std::find(supportedMetrics.begin(), supportedMetrics.end(), METRIC_KEY(FULL_DEVICE_NAME)) != supportedMetrics.end()) {
                    fullDeviceName = GetCore()->GetMetric(deviceNameWithID, METRIC_KEY(FULL_DEVICE_NAME)).as<std::string>();
                }
            }

            if (fullDeviceName.empty()) {
                uniqueName = newParsed.getDeviceName() + "_" + defaultDeviceID;
            } else {
                uniqueName = fullDeviceName + "_" + defaultDeviceID;
            }

            LOG_DEBUG("[AUTOPLUGIN]:deviceNameWithID:%s, defaultDeviceID:%s, uniqueName:%s",
                    deviceNameWithID.c_str(), defaultDeviceID.c_str(), uniqueName.c_str());
            // create meta device
            metaDevices.push_back({deviceNameWithID, getDeviceConfig(deviceNameWithID), numRequests, defaultDeviceID, uniqueName, devicePriority});
        }
        if (enableDevicePriority) {
            devicePriority++;
        }
    }

    return metaDevices;
}

InferenceEngine::Parameter MultiDeviceInferencePlugin::GetConfig(const std::string& name,
        const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (supported_configKeys.end() != std::find(supported_configKeys.begin(), supported_configKeys.end(), name)) {
        auto it = _config.find(name);
        if (it == _config.end()) {
            IE_THROW() << "config key not set" << name;
        } else {
            return { it->second };
        }
    } else {
        IE_THROW() << "2-Unsupported config key: " << name;
    }
}

void MultiDeviceInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    auto autoSContext = std::make_shared<AutoScheduleContext>();
    std::map<std::string, std::string> filterConfig;
    CheckConfig(config, autoSContext, filterConfig);
    for (auto && kvp : config) {
        const auto& name = kvp.first;
        _config[name] = kvp.second;
    }
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "MultiDevicePlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(MultiDeviceInferencePlugin, version)

MultiDeviceInferencePlugin::MultiDeviceInferencePlugin() {
    _pluginName = "MULTI";
}

InferenceEngine::Parameter MultiDeviceInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    auto RO_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RO);
    };
    auto RW_property = [](const std::string& propertyName) {
        return ov::PropertyName(propertyName, ov::PropertyMutability::RW);
    };
    if (name == ov::supported_properties) {
        std::vector<ov::PropertyName> roProperties {RO_property(ov::supported_properties.name()),
                                                    RO_property(ov::device::full_name.name())
        };
        // the whole config is RW before network is loaded.
        std::vector<ov::PropertyName> rwProperties {RW_property(ov::hint::model_priority.name()),
                                                    RW_property(ov::log::level.name()),
                                                    RW_property(ov::device::priorities.name()),
                                                    RW_property(ov::enable_profiling.name()),
                                                    RW_property(ov::hint::allow_auto_batching.name()),
                                                    RW_property(ov::hint::performance_mode.name()),
                                                    RW_property(ov::hint::num_requests.name())
        };
        std::vector<ov::PropertyName> supportedProperties;
        supportedProperties.reserve(roProperties.size() + rwProperties.size());
        supportedProperties.insert(supportedProperties.end(), roProperties.begin(), roProperties.end());
        supportedProperties.insert(supportedProperties.end(), rwProperties.begin(), rwProperties.end());
        return supportedProperties;
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(FULL_DEVICE_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == ov::device::full_name) {
        std::string device_name = { GetName() };
        return decltype(ov::device::full_name)::value_type {device_name};
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, supported_configKeys);
    } else {
        IE_THROW() << "Unsupported metric key: " << name;
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

        OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "MultiDeviceInferencePlugin::LoadNetworkImpl::AutoMode");
        auto autoSContext = std::make_shared<AutoScheduleContext>();
        std::map<std::string, std::string> filterConfig;
        auto strDevices = GetDeviceList(fullConfig);
        auto deviceList = GetCore()->GetAvailableDevices();
        // keep the secondary priorities when the config key is one of the available hardware devices
        CheckConfig(fullConfig, autoSContext, filterConfig, deviceList);
        // filter the device that supports filter configure
        auto metaDevices = ParseMetaDevices(strDevices, fullConfig);
        auto supportDevicesByConfig = FilterDevice(metaDevices, filterConfig);
        if (supportDevicesByConfig.size() == 0) {
             IE_THROW() << "There is no device support the configure";
        }
        auto supportDevices = supportDevicesByConfig;
        CNNNetwork clonedNetwork;
        std::string clonedModelPath = modelPath;
        if (modelPath.empty()) {
            // if network is valid
            LOG_INFO("[AUTOPLUGIN]:load with CNN network");
            supportDevices = FilterDeviceByNetwork(supportDevicesByConfig, network);
            // clone the network, in case of reshape conflict
            clonedNetwork = InferenceEngine::details::cloneNetwork(network);
        } else {
            // model path, enable model load with single device situation
            if (supportDevices.size() > 1) {
                clonedNetwork = GetCore()->ReadNetwork(modelPath, std::string());
                // do we really need to disable model path?
                clonedModelPath = "";
                LOG_INFO("[AUTOPLUGIN]:load with CNN network");
            } else {
                LOG_INFO("[AUTOPLUGIN]:load with model path");
            }
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
                     LOG_INFO("[AUTOPLUGIN]:device:%s, config:%s=%s", iter->deviceName.c_str(),
                             config.first.c_str(), config.second.c_str());
                 }
             }
             auto tmpiter = std::find_if(fullConfig.begin(), fullConfig.end(), [](const std::pair<std::string, std::string>& config) {
                            return (config.first == CONFIG_KEY(ALLOW_AUTO_BATCHING));
                            });
             if (tmpiter != fullConfig.end())
                 deviceConfig.insert({tmpiter->first, tmpiter->second});
             iter->config = deviceConfig;
             strDevices += iter->deviceName;
             strDevices += ((iter + 1) == supportDevices.end()) ? "" : ",";
             LOG_INFO("[AUTOPLUGIN]:device:%s, priority:%ld", iter->deviceName.c_str(), iter->devicePriority);
        }
        autoSContext->_modelPath = clonedModelPath;
        // clone the network, in case of reshape conflict
        autoSContext->_network = clonedNetwork;
        autoSContext->_devicePriorities = supportDevices;
        autoSContext->_devicePrioritiesInitial = supportDevices;
        autoSContext->_strDevices = strDevices;
        autoSContext->_plugin = this;
        autoSContext->_core = GetCore();
        auto tmpiter = fullConfig.find(ov::intel_auto::device_bind_buffer.name());
        if (tmpiter != fullConfig.end() && tmpiter->second == PluginConfigParams::YES)
            autoSContext->_bindBuffer = true;
        return std::make_shared<AutoExecutableNetwork>(autoSContext, std::make_shared<AutoSchedule>());
    }
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "MultiDeviceInferencePlugin::LoadNetworkImpl:MultiMode");
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
            auto tmpiter = fullConfig.find(CONFIG_KEY(ALLOW_AUTO_BATCHING));
            if (tmpiter != fullConfig.end())
                p.config.insert({tmpiter->first, tmpiter->second});
            const auto& deviceName = p.deviceName;
            const auto& deviceConfig = p.config;
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
    auto executor = executorManager()->getIdleCPUStreamsExecutor(
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
    auto multiSContext = std::make_shared<MultiScheduleContext>();
    multiSContext->_devicePriorities = metaDevices;
    multiSContext->_devicePrioritiesInitial = metaDevices;
    multiSContext->_networksPerDevice = executableNetworkPerDevice;
    multiSContext->_config = multiNetworkConfig;
    multiSContext->_needPerfCounters = enablePerfCounters;
    multiSContext->_core = GetCore();
    IExecutableNetworkInternal::Ptr impl;
    auto tmpiter = fullConfig.find(ov::intel_auto::device_bind_buffer.name());
    if (tmpiter != fullConfig.end() && tmpiter->second == PluginConfigParams::YES)
        impl = std::make_shared<MultiExecutableNetwork>(multiSContext, std::make_shared<BinderMultiSchedule>());
    else
        impl = std::make_shared<MultiExecutableNetwork>(multiSContext, std::make_shared<MultiSchedule>());
    if (!modelPath.empty()) {
        SetExeNetworkInfo(impl,
                          executableNetworkPerDevice.begin()->second->GetInputsInfo(),
                          executableNetworkPerDevice.begin()->second->GetOutputsInfo());
        impl->setInputs(executableNetworkPerDevice.begin()->second->getInputs());
        impl->setOutputs(executableNetworkPerDevice.begin()->second->getOutputs());
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

std::list<DeviceInformation> MultiDeviceInferencePlugin::GetValidDevice(
    const std::vector<DeviceInformation>& metaDevices,
    const std::string& networkPrecision) {
    if (metaDevices.empty()) {
        IE_THROW(NotFound) << "No available device to select in " << GetName() << " plugin";
    }

    std::list<DeviceInformation> CPU;
    std::list<DeviceInformation> dGPU;
    std::list<DeviceInformation> iGPU;
    std::list<DeviceInformation> MYRIAD;
    std::list<DeviceInformation> VPUX;

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
            auto& gpuUniqueName = item.uniqueName;
            if (gpuUniqueName.find("iGPU") != std::string::npos) {
                iGPU.push_back(item);
            } else if (gpuUniqueName.find("dGPU") != std::string::npos) {
                dGPU.push_back(item);
            }
            continue;
        }
    }

    // Priority of selecting device: dGPU > VPUX > iGPU > MYRIAD > CPU
    std::list<DeviceInformation> devices;
    if (networkPrecision == "INT8") {
        devices.splice(devices.end(), VPUX);
        devices.splice(devices.end(), dGPU);
    } else {
        devices.splice(devices.end(), dGPU);
        devices.splice(devices.end(), VPUX);
    }
    devices.splice(devices.end(), iGPU);
    devices.splice(devices.end(), MYRIAD);
    devices.splice(devices.end(), CPU);

    std::list<DeviceInformation> validDevices;

    if (metaDevices.size() > 1) {
        auto selectSupportDev = [this, &devices, &validDevices](const std::string& networkPrecision) {
            for (auto iter = devices.begin(); iter != devices.end();) {
                auto capability = GetCore()
                                      ->GetMetric(iter->deviceName, METRIC_KEY(OPTIMIZATION_CAPABILITIES))
                                      .as<std::vector<std::string>>();
                auto supportNetwork = std::find(capability.begin(), capability.end(), (networkPrecision));
                if (supportNetwork != capability.end()) {
                    validDevices.push_back(std::move(*iter));
                    devices.erase(iter++);
                    continue;
                }
                iter++;
            }
        };
        selectSupportDev(networkPrecision);
        // If network is FP32, continue to collect the device support FP16 but not support FP32.
        if (networkPrecision == "FP32") {
            const std::string f16 = "FP16";
            selectSupportDev(f16);
        }
    } else {
        validDevices.push_back(metaDevices[0]);
    }

    if (validDevices.empty()) {
        IE_THROW() << "Cannot select any device";
    }
    // sort validDevices
    validDevices.sort([](const DeviceInformation& a, const DeviceInformation& b) {
        return a.devicePriority < b.devicePriority;
    });

    return validDevices;
}

DeviceInformation MultiDeviceInferencePlugin::SelectDevice(const std::vector<DeviceInformation>& metaDevices,
        const std::string& networkPrecision, unsigned int priority) {
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "MultiDeviceInferencePlugin::SelectDevice");

    std::list<DeviceInformation> validDevices = GetValidDevice(metaDevices, networkPrecision);

    // all available Devices are in validDevices now
    // need to remove higher priority devices
    // save the last device first
    DeviceInformation lastDevice = validDevices.back();
    {
        // begin to filter devices
        std::lock_guard<std::mutex> lck(_mtx);
        for (auto && kvp : _priorityMap) {
            if (kvp.first >= priority) {
                continue;
            }
            auto& filterDevices = kvp.second;
            auto sd = std::remove_if(validDevices.begin(), validDevices.end(), [&filterDevices](const DeviceInformation& device) {
                    auto iter = std::find_if(filterDevices.begin(), filterDevices.end(), [&device](std::string uniqueName) {
                            return (uniqueName == device.uniqueName);
                            });
                    return iter != filterDevices.end() ? true : false;
                    });
            validDevices.erase(sd, validDevices.end());
        }
    }

    DeviceInformation* ptrSelectDevice =  NULL;
    if (validDevices.empty()) {
        // after remove higher priority device,but the available devices is null,
        // so select the last device of all available Devices.
        ptrSelectDevice = &lastDevice;
    } else {
        // select the first device in the rest of available devices.
        ptrSelectDevice = &validDevices.front();
    }
    //recode the device priority
    RegisterPriority(priority, ptrSelectDevice->uniqueName);
    return *ptrSelectDevice;
}

void MultiDeviceInferencePlugin::UnregisterPriority(const unsigned int& priority,
        const std::string& deviceName) {
    std::lock_guard<std::mutex> lck(_mtx);
    auto& priorityDevices = _priorityMap[priority];
    for (auto iter = priorityDevices.begin(); iter != priorityDevices.end();) {
        if (*iter == deviceName) {
            priorityDevices.erase(iter);
            break;
        }
        iter++;
    }
}

void MultiDeviceInferencePlugin::RegisterPriority(const unsigned int& priority,
        const std::string& deviceName) {
    std::lock_guard<std::mutex> lck(_mtx);
    auto& priorityDevices = _priorityMap[priority];
    priorityDevices.push_back(deviceName);
}

std::string MultiDeviceInferencePlugin::GetDeviceList(const std::map<std::string, std::string>& config) const {
    std::string allDevices;
    auto deviceList = GetCore()->GetAvailableDevices();
    auto deviceListConfig = config.find(MultiDeviceConfigParams::KEY_MULTI_DEVICE_PRIORITIES);
    if (deviceListConfig == config.end()) {
        for (auto&& device : deviceList) {
            allDevices += device;
            allDevices += ((device == deviceList[deviceList.size()-1]) ? "" : ",");
        }
    } else {
        // parsing the string and splitting the comma-separated tokens
        std::string::size_type i = 0;
        std::string::size_type idelimeter;
        std::vector<std::string> deviceVec;
        auto priorities = deviceListConfig->second;
        while ((idelimeter = priorities.find(',', i)) != std::string::npos) {
            deviceVec.push_back(priorities.substr(i, idelimeter - i));
            i = idelimeter + 1;
        }
        // last token in the string (which has no comma after that)
        deviceVec.push_back(priorities.substr(i, priorities.length() - i));
        std::vector<std::string> devicesToBeDeleted;
        auto updateDeviceVec = [&](const std::string& delPattern = "") {
            auto iter = deviceVec.begin();
            while (iter != deviceVec.end()) {
                if (delPattern.empty()) {
                    if ((*iter).find("-") == 0) {
                        devicesToBeDeleted.push_back((*iter).erase(0, 1));
                        iter = deviceVec.erase(iter);
                    } else {
                        iter++;
                    }
                } else {
                    if ((*iter).find(delPattern) != std::string::npos)
                        iter = deviceVec.erase(iter);
                    else
                        iter++;
                }
            }
        };
        updateDeviceVec();
        if (devicesToBeDeleted.size() == 0) {
            allDevices = deviceListConfig->second;
        } else {
            auto deviceNeedToMerge = [&](const std::string& devicename) {
                for (auto&& iter : devicesToBeDeleted) {
                    if (iter.find(devicename) != std::string::npos)
                        return true;
                }
                return false;
            };
            auto mergeDeviceList = [&]() {
                std::vector<std::string> mergedList;
                auto prevSize = mergedList.size();
                for (auto&& iter : deviceVec) {
                    for (auto&& viter : deviceList) {
                        if (viter.find(iter) != std::string::npos && deviceNeedToMerge(iter))
                            mergedList.push_back(std::move(viter));
                    }
                    // if virtual devices or mock devices
                    if (mergedList.size() == prevSize)
                        mergedList.push_back(std::move(iter));
                    prevSize = mergedList.size();
                }
                return mergedList;
            };

            deviceVec = deviceVec.size() == 0 ? deviceList : mergeDeviceList();
            for (auto& iter : devicesToBeDeleted) {
                LOG_INFO("[AUTOPLUGIN]:remove %s from device candidate list", iter.c_str());
                updateDeviceVec(iter);
            }
            for (auto&& device : deviceVec) {
                allDevices += device;
                allDevices += ((device == deviceVec[deviceVec.size()-1]) ? "" : ",");
            }
        }
    }

    if (allDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }

    return allDevices;
}

void MultiDeviceInferencePlugin::CheckConfig(const std::map<std::string, std::string>& config,
                                             AutoScheduleContext::Ptr& context,
                                             std::map<std::string, std::string>& filterConfig,
                                             const std::vector<std::string>& devicesList) {
    // TODO need to optimize this code, too much duplicated code
    const auto perf_hints_configs = PerfHintsConfig::SupportedKeys();
    for (auto&& kvp : config) {
        if (kvp.first == ov::enable_profiling) {
            if (kvp.second == PluginConfigParams::YES) {
                context->_needPerfCounters = true;
                filterConfig.insert({kvp.first, kvp.second});
            } else if (kvp.second == PluginConfigParams::NO) {
                context->_needPerfCounters = false;
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
        } else if (kvp.first == ov::log::level.name()) {
               auto success = MultiDevicePlugin::setLogLevel(kvp.second);
               if (!success) {
                   IE_THROW() << "Unsupported config value: " << kvp.second
                              << " for key: " << kvp.first;
               }
        } else if (kvp.first == ov::hint::model_priority) {
            try {
                int priority = -1;
                if (kvp.second == "LOW" ||
                    kvp.second == CONFIG_VALUE(MODEL_PRIORITY_LOW)) {
                    priority = static_cast<int>(ov::hint::Priority::HIGH) - static_cast<int>(ov::hint::Priority::LOW);
                }
                if (kvp.second == "MEDIUM" ||
                    kvp.second == CONFIG_VALUE(MODEL_PRIORITY_MED)) {
                    priority = static_cast<int>(ov::hint::Priority::HIGH) - static_cast<int>(ov::hint::Priority::MEDIUM);
                }
                if (kvp.second == "HIGH" ||
                    kvp.second == CONFIG_VALUE(MODEL_PRIORITY_HIGH)) {
                    priority = static_cast<int>(ov::hint::Priority::HIGH) - static_cast<int>(ov::hint::Priority::HIGH);
                }
                if (priority < 0) {
                    IE_THROW() << "Unsupported config value: " << kvp.second
                        << " for key: " << kvp.first;
                }
                context->_modelPriority = priority;
            } catch(...) {
                IE_THROW() << "Unsupported config value: " << kvp.second
                           << " for key: " << kvp.first;
            }
        } else if (kvp.first == ov::hint::allow_auto_batching) {
            if (kvp.second == PluginConfigParams::NO) {
                context->_batchingDisabled = true;
                continue;
            }
        } else if (kvp.first == ov::intel_auto::device_bind_buffer.name()) {
            if (kvp.second == PluginConfigParams::YES ||
                kvp.second == PluginConfigParams::NO) {
                continue;
            } else {
                IE_THROW() << "Unsupported config value: " << kvp.second
                           << " for key: " << kvp.first;
            }
        } else if (std::find(perf_hints_configs.begin(), perf_hints_configs.end(), kvp.first) != perf_hints_configs.end()) {
            PerfHintsConfig::CheckConfigAndValue(kvp);
            if (kvp.first == PluginConfigParams::KEY_PERFORMANCE_HINT) {
                context->_performanceHint = kvp.second;
            }
        } else if (std::find(devicesList.begin(), devicesList.end(), kvp.first) != devicesList.end() ||
                   kvp.first == "HETERO" || kvp.first == "MULTI" || kvp.first == "AUTO") {
            // keep secondary prperties for HW or virtual device
            continue;
        } else if (supported_configKeys.end() ==
                   std::find(supported_configKeys.begin(), supported_configKeys.end(), kvp.first)) {
            IE_THROW() << "1-Unsupported config key: " << kvp.first;
        } else if (kvp.first.find("AUTO_") == 0) {
            continue;
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
        auto supportedMetrics = GetCore()->GetMetric(item.deviceName, METRIC_KEY(SUPPORTED_METRICS)).as<std::vector<std::string>>();
        if (std::find(supportedMetrics.begin(), supportedMetrics.end(), METRIC_KEY(SUPPORTED_CONFIG_KEYS)) != supportedMetrics.end()) {
            auto supportKeys = GetCore()->GetMetric(item.deviceName, METRIC_KEY(SUPPORTED_CONFIG_KEYS)).as<std::vector<std::string>>();
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
std::vector<DeviceInformation> MultiDeviceInferencePlugin::FilterDeviceByNetwork(const std::vector<DeviceInformation>& metaDevices,
                                                InferenceEngine::CNNNetwork network) {
    if (metaDevices.empty()) {
        IE_THROW(NotFound) << "No available device to filter " << GetName() <<  " plugin";
    } else if (metaDevices.size() == 1) {
        return metaDevices;
    }

    std::vector<DeviceInformation> filterDevice;
    auto model = network.getFunction();
    auto isStateful = [&]() {
        for (auto& op : model->get_ops()) {
            if (std::dynamic_pointer_cast<ngraph::op::AssignBase>(op) ||
                std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(op)) {
                    LOG_INFO("[AUTOPLUGIN]:stateful mode, try deployed to CPU");
                    return true;
                }
        }
        return false;
    };
    if (model->is_dynamic() || isStateful()) {
        for (auto& iter : metaDevices) {
            if (iter.deviceName.find("CPU") != std::string::npos) {
                filterDevice.push_back(iter);
                break;
            }
        }
        if (filterDevice.size() == 0)
            IE_THROW(NotFound) << "No available device for dynamic shape network !";
        return filterDevice;
    }
    return metaDevices;
}
}  // namespace MultiDevicePlugin
