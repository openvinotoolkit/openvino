// Copyright (C) 2018-2023 Intel Corporation
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
#include "openvino/runtime/auto/properties.hpp"
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
        bool isINTModel = ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(nGraphFunc);
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
    int MapPriorityValues(ov::hint::Priority priority) {
        switch (priority) {
            case ov::hint::Priority::HIGH:
                return 0;
            case ov::hint::Priority::MEDIUM:
                return 1;
            case ov::hint::Priority::LOW:
                return 2;
            default:
                return 1;
        }
    }
    std::map<std::string, std::string> ConvertToStringMap(ov::AnyMap& properties) {
        std::map<std::string, std::string> configs;
        for (auto& property : properties) {
            configs[property.first] = property.second.as<std::string>();
        }
        return configs;
    }
}  // namespace

std::mutex MultiDeviceInferencePlugin::_mtx;
std::map<unsigned int, std::list<std::string>> MultiDeviceInferencePlugin::_priorityMap;

ov::AnyMap MultiDeviceInferencePlugin::PreProcessConfig(const std::map<std::string, std::string>& orig_config) const {
    ov::AnyMap properties = ov::AnyMap(orig_config.begin(), orig_config.end());
    for (auto& property : properties) {
        // for model_priority, the values need to be converted
        if (property.first == ov::hint::model_priority.name()) {
            ov::Any converted_val{nullptr};
            auto legacy_val = property.second.as<std::string>();
            if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH) {
                converted_val = ov::hint::Priority::HIGH;
            } else if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED) {
                converted_val = ov::hint::Priority::MEDIUM;
            } else if (legacy_val == InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW) {
                converted_val = ov::hint::Priority::LOW;
            } else {
                converted_val = legacy_val;
            }
            property.second = converted_val;
        }
    }
    return properties;
}

std::vector<DeviceInformation> MultiDeviceInferencePlugin::ParseMetaDevices(const std::string& priorities,
                                                                          const std::map<std::string, std::string> & config) const {
    std::vector<DeviceInformation> metaDevices;

    // parsing the string and splitting to tokens
    std::vector<std::string> devicesWithRequests = _pluginConfig.ParsePrioritiesDevices(priorities);

    auto setDefaultHint = [&](const std::string& targetDevice,
                              std::map<std::string, std::string>& deviceConfig,
                              const std::map<std::string, std::string>& mergedConfig) {
        auto isSetPerHint = mergedConfig.find(PluginConfigParams::KEY_PERFORMANCE_HINT) != mergedConfig.end();
        auto isSetDeviceProperties = mergedConfig.find(targetDevice) != mergedConfig.end();
        if (GetName() == "AUTO" && !isSetPerHint && !isSetDeviceProperties) {
            // setting latency as the default performance mode if
            // 1. no hints setting for AUTO plugin
            // 2. no ov::device::properties(secondary properties) setting for target device
            deviceConfig[PluginConfigParams::KEY_PERFORMANCE_HINT] = PluginConfigParams::LATENCY;
            return;
        }

        if (GetName() == "MULTI") {
            auto isSetNumStreams = mergedConfig.find(ov::num_streams.name()) != mergedConfig.end();
            auto isSetAffinity = mergedConfig.find(ov::affinity.name()) != mergedConfig.end();
            auto isSetNumThreads = mergedConfig.find(ov::inference_num_threads.name()) != mergedConfig.end();
            if (!isSetPerHint && !isSetAffinity && !isSetNumThreads && !isSetDeviceProperties && !isSetNumStreams) {
                // setting tput as the default performance mode if
                // 1. no hints setting for MULTI plugin
                // 2. no affinity setting for MULTI plugin
                // 3. no inference_num_threads setting for MULTI plugin
                // 4. no ov::device::properties(secondary properties) setting for target device
                // 5. no ov::num_streams setting for target device
                deviceConfig[PluginConfigParams::KEY_PERFORMANCE_HINT] = PluginConfigParams::THROUGHPUT;
            }
        }
    };

    auto getDeviceConfig = [&] (const DeviceName & deviceWithID) {
        DeviceIDParser deviceParser(deviceWithID);
        std::string deviceName = deviceParser.getDeviceName();
        std::map<std::string, std::string> tconfig = config;

        // set device ID if any
        std::string deviceIDLocal = deviceParser.getDeviceID();
        if (!deviceIDLocal.empty()) {
            tconfig[PluginConfigParams::KEY_DEVICE_ID] = deviceIDLocal;
        }
        auto deviceConfig = GetCore()->GetSupportedConfig(deviceName, tconfig);
        setDefaultHint(deviceName, deviceConfig, tconfig);
        return deviceConfig;
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
        if (priString.empty())
            return false;
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
            std::string tempDeviceID = "";
            if (newParsed.getDeviceID().empty()) {
                defaultDeviceID = getDefaultDeviceID(deviceNameWithID);
                tempDeviceID = defaultDeviceID;
            } else {
                tempDeviceID = newParsed.getDeviceID();
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
                uniqueName = newParsed.getDeviceName() + "_" + tempDeviceID;
            } else {
                uniqueName = fullDeviceName + "_" + tempDeviceID;
            }

            LOG_DEBUG_TAG("deviceNameWithID:%s, defaultDeviceID:%s, uniqueName:%s",
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
    auto val = _pluginConfig.get_property(name);
    if (!IsNewAPI()) {
        if (name == ov::hint::model_priority.name()) { // need to convert priority values to old API
            ov::Any legacy_val{nullptr};
            if (!val.empty()) {
            switch (val.as<ov::hint::Priority>()) {
                case ov::hint::Priority::LOW: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_LOW; break;
                case ov::hint::Priority::MEDIUM: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_MED; break;
                case ov::hint::Priority::HIGH: legacy_val = InferenceEngine::PluginConfigParams::MODEL_PRIORITY_HIGH; break;
            default: OPENVINO_ASSERT(false, "Unsupported model priority value");
            }
        }
        return legacy_val;
        } else {
            return val;
        }
    } else {
        return val;
    }
    return val;
}

void MultiDeviceInferencePlugin::SetConfig(const std::map<std::string, std::string> & config) {
    // with setConfig, only multi/auto supported internal configs can be accepted
    _pluginConfig.set_property(PreProcessConfig(config));
}

static const Version version = {{2, 1}, CI_BUILD_NUMBER, "MultiDevicePlugin"};
IE_DEFINE_PLUGIN_CREATE_FUNCTION(MultiDeviceInferencePlugin, version)

MultiDeviceInferencePlugin::MultiDeviceInferencePlugin() {
    _pluginName = "MULTI";
}

InferenceEngine::Parameter MultiDeviceInferencePlugin::GetMetric(const std::string& name,
                                         const std::map<std::string, InferenceEngine::Parameter> & options) const {
    if (name == ov::supported_properties) {
        auto ret = _pluginConfig.supportedProperties(GetName());
        return ret;
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, _pluginConfig.supportedMetrics(GetName()));
    } else if (name == ov::device::full_name) {
        std::string device_name = { GetName() };
        return decltype(ov::device::full_name)::value_type {device_name};
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        auto deviceList = GetCore()->GetAvailableDevices();
        std::vector<std::string> capabilities;
        for (auto device : deviceList) {
            auto devCapabilities = GetCore()->GetMetric(device, ov::device::capabilities.name()).as<std::vector<std::string>>();
            capabilities.insert(capabilities.end(), devCapabilities.begin(), devCapabilities.end());
        }
        std::sort(capabilities.begin(), capabilities.end());
        capabilities.resize(std::distance(capabilities.begin(), std::unique(capabilities.begin(), capabilities.end())));
        auto delItem = std::find(capabilities.begin(), capabilities.end(), ov::device::capability::EXPORT_IMPORT);
        if (delItem != capabilities.end()) {
            capabilities.erase(delItem);
        }
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, _pluginConfig.supportedConfigKeys(GetName()));
    } else {
        IE_THROW() << "Unsupported metric key: " << name;
    }
}

// Is called only when caching is enabled
ov::SoPtr<InferenceEngine::IExecutableNetworkInternal> MultiDeviceInferencePlugin::LoadNetwork(const std::string& modelPath,
                                                                        const std::map<std::string, std::string>& config) {
    return {LoadNetworkImpl(modelPath, {}, config), nullptr};
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
    // to use plugin's name as the log tag
    _LogTag = GetName();
    bool workModeAuto = GetName() == "AUTO";
    auto loadConfig = _pluginConfig;
    // if no perf hint from user with compiled model, or already been set with plugin
    // apply latency for AUTO, tput for MULTI
    bool isHintSet = _pluginConfig.is_set_by_user(ov::hint::performance_mode) || config.find(ov::hint::performance_mode.name()) != config.end();
    if (!isHintSet) {
        if (workModeAuto) {
            // set performance hint to 'LATENCY' model for AutoExecutable Network.
            loadConfig.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        } else {
            // set performance hint to 'THROUGHPUT' model for MultiExecutable Network.
            loadConfig.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        }
    }
    // updateFromMap will check config valid
    loadConfig.set_user_property(PreProcessConfig(config), workModeAuto? true : false);
    loadConfig.apply_user_properties();
    auto fullProperty = loadConfig.get_full_properties();
    // this can be updated when plugin switch to 2.0 API
    std::map<std::string, std::string> fullConfig = ConvertToStringMap(fullProperty);
    // Remove the performance hint as this is set by plugin logic, not from user
    if (!isHintSet)
        fullConfig.erase(ov::hint::performance_mode.name());
    if (!loadConfig.is_set_by_user(ov::cache_dir))
        fullConfig.erase(ov::cache_dir.name());
    if (!loadConfig.is_set_by_user(ov::hint::execution_mode))
        fullConfig.erase(ov::hint::execution_mode.name());
    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, ov::Any> multiNetworkConfig;
    std::vector<DeviceInformation> metaDevices;
    auto priorities = loadConfig.get_property(ov::device::priorities);
    if (priorities.find("AUTO") != std::string::npos || priorities.find("MULTI") != std::string::npos) {
        IE_THROW() << "The device candidate list should not include the meta plugin for " << GetName() << " device";
    }
    // If the user sets the property, insert the property into the deviceConfig
    auto insertPropToConfig = [&](std::string property,
                                  std::string& deviceName,
                                  std::map<std::string, std::string>& deviceConfig) {
        if (deviceConfig.find(property) == deviceConfig.end()) {
            auto tmpiter = fullConfig.find(property);
            if (tmpiter != fullConfig.end()) {
                deviceConfig.insert({tmpiter->first, tmpiter->second});
                LOG_INFO_TAG("device:%s, config:%s=%s",
                                deviceName.c_str(),
                                tmpiter->first.c_str(),
                                tmpiter->second.c_str());
            }
        }
    };

    // if workMode is AUTO
    // only AUTO uses CheckConfig() to check fullConfig's parameters, MULTI does not
    if (workModeAuto) {
        // check the configure and check if need to set PerfCounters configure to device
        // and set filter configure

        OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "MultiDeviceInferencePlugin::LoadNetworkImpl::AutoMode");
        auto autoSContext = std::make_shared<AutoScheduleContext>();
        std::map<std::string, std::string> filterConfig;
        auto strDevices = GetDeviceList(fullConfig);
        // fill in the context for auto
        if (loadConfig.get_property(ov::enable_profiling)) {
            filterConfig.insert({ov::enable_profiling.name(), PluginConfigParams::YES});
            autoSContext->_needPerfCounters = true;
        }
        autoSContext->_modelPriority = MapPriorityValues(loadConfig.get_property(ov::hint::model_priority));
        autoSContext->_batchingDisabled = !(loadConfig.get_property(ov::hint::allow_auto_batching));
        autoSContext->_performanceHint = loadConfig.get_property(ov::hint::performance_mode.name()).as<std::string>();
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
            LOG_INFO_TAG("load with CNN network");
            supportDevices = FilterDeviceByNetwork(supportDevicesByConfig, network);
            // clone the network, in case of reshape conflict
            clonedNetwork = InferenceEngine::details::cloneNetwork(network);
        } else {
            // model path, enable model load with single device situation
            if (supportDevices.size() > 1) {
                clonedNetwork = GetCore()->ReadNetwork(modelPath, std::string());
                // do we really need to disable model path?
                clonedModelPath = "";
                LOG_INFO_TAG("load with CNN network");
            } else {
                LOG_INFO_TAG("load with model path");
            }
        }
        // reset the strDevices to support devices
        strDevices = "";
        // calling GetValidDevices() to get a prioritized list of devices
        auto devicesWithPriority = GetValidDevice(supportDevices, networkPrecision);
        for (auto iter = devicesWithPriority.begin(); iter != devicesWithPriority.end(); iter++) {
            strDevices += iter->deviceName;
            strDevices += ",";
        }
        strDevices.pop_back();
        for (auto iter = supportDevices.begin(); iter != supportDevices.end(); iter++) {
            auto& configs = iter->config;
            for (auto& config : configs) {
                LOG_INFO_TAG("device:%s, config:%s=%s",
                             iter->deviceName.c_str(),
                             config.first.c_str(),
                             config.second.c_str());
            }
            // carry on batch configs only if user explicitly sets
            if (loadConfig.is_set_by_user(ov::hint::allow_auto_batching))
                insertPropToConfig(ov::hint::allow_auto_batching.name(), iter->deviceName, configs);
            if (loadConfig.is_set_by_user(ov::auto_batch_timeout))
                insertPropToConfig(ov::auto_batch_timeout.name(), iter->deviceName, configs);
            insertPropToConfig(ov::cache_dir.name(), iter->deviceName, configs);
            LOG_INFO_TAG("device:%s, priority:%ld", iter->deviceName.c_str(), iter->devicePriority);
        }
        autoSContext->_modelPath = clonedModelPath;
        // clone the network, in case of reshape conflict
        autoSContext->_network = clonedNetwork;
        autoSContext->_devicePriorities = supportDevices;
        autoSContext->_devicePrioritiesInitial = supportDevices;
        autoSContext->_strDevices = strDevices;
        autoSContext->_plugin = this;
        autoSContext->_core = GetCore();
        autoSContext->_LogTag = _LogTag;
        autoSContext->_bindBuffer = loadConfig.get_property(ov::intel_auto::device_bind_buffer);
        autoSContext->_startupfallback = loadConfig.get_property(ov::intel_auto::enable_startup_fallback);
        return std::make_shared<AutoExecutableNetwork>(autoSContext, std::make_shared<AutoSchedule>());
    }
    OV_ITT_SCOPED_TASK(itt::domains::MULTIPlugin, "MultiDeviceInferencePlugin::LoadNetworkImpl:MultiMode");
    // if is cumulative, PERFORMANCE_HINT set to THROUGHPUT and _LogTag set to AUTO
    auto configIter =
        std::find_if(fullConfig.begin(), fullConfig.end(), [](const std::pair<std::string, std::string>& config) {
            return (config.first == CONFIG_KEY(PERFORMANCE_HINT));
        });
    if (configIter != fullConfig.end() && configIter->second == InferenceEngine::PluginConfigParams::CUMULATIVE_THROUGHPUT) {
        configIter->second = InferenceEngine::PluginConfigParams::THROUGHPUT;
        _LogTag = "AUTO";
        LOG_INFO_TAG("CUMULATIVE Call MULTI PERFORMANCE_HINT set to THROUGHPUT");
    }
    if (priorities.empty()) {
        IE_THROW() << "KEY_MULTI_DEVICE_PRIORITIES key is not set for " << GetName() << " device";
    } else {  // for use case -d MULTI:xPU or -d AUTO:xPU
        auto metaDevicesByConfig = ParseMetaDevices(priorities, fullConfig);
        metaDevices = modelPath.empty() ? FilterDeviceByNetwork(metaDevicesByConfig, network)
                                        : metaDevicesByConfig;
        if (metaDevicesByConfig.size() != metaDevices.size()) {
            LOG_DEBUG_TAG("stateful/dynamic model, loaded to single device");
            multiNetworkConfig[ov::device::priorities.name()]
                    = metaDevices[0].deviceName;
        } else {
            multiNetworkConfig[ov::device::priorities.name()] = priorities;
        }
    }
    auto multiSContext = std::make_shared<MultiScheduleContext>();
    DeviceMap<SoExecutableNetworkInternal> executableNetworkPerDevice;
    std::mutex load_mutex;
    std::vector<Task> loads;

    auto loadInferEngTask = [&](DeviceInformation& p) {
        auto tmpiter = fullConfig.find(CONFIG_KEY(ALLOW_AUTO_BATCHING));
        if (tmpiter != fullConfig.end()) {
            if (tmpiter->second == PluginConfigParams::NO) {
                LOG_INFO_TAG("set %s=%s", tmpiter->first.c_str(), tmpiter->second.c_str());
                multiSContext->_batchingDisabled = true;
            }
            if (loadConfig.is_set_by_user(ov::hint::allow_auto_batching))
                p.config.insert({tmpiter->first, tmpiter->second});
        }
        if (loadConfig.is_set_by_user(ov::auto_batch_timeout))
            insertPropToConfig(ov::auto_batch_timeout.name(), p.deviceName, p.config);
        insertPropToConfig(ov::cache_dir.name(), p.deviceName, p.config);
        const auto& deviceName = p.deviceName;
        const auto& deviceConfig = p.config;
        SoExecutableNetworkInternal exec_net;
        LOG_DEBUG_TAG("load network to device:%s", deviceName.c_str());
        try {
            if (modelPath.empty()) {
                exec_net = GetCore()->LoadNetwork(network, deviceName, deviceConfig);
            } else {
                exec_net = GetCore()->LoadNetwork(modelPath, deviceName, deviceConfig);
            }
        } catch (const IE::Exception& iie) {
            if (_LogTag == "AUTO") {
                LOG_DEBUG_TAG("Failed to load network to device:%s with error: %s", deviceName.c_str(), iie.what());
                return;
            } else {
                IE_THROW() << "Failed to load network to device: " << deviceName.c_str() << " with error:" <<
                    iie.what();
            }
        }

        try {
            std::string sStreamNums = "";
            std::string sThreadNums = "";
            if (deviceName.find("CPU") != std::string::npos) {
                sStreamNums = exec_net->GetMetric(ov::num_streams.name()).as<std::string>();
                sThreadNums = exec_net->GetMetric(ov::inference_num_threads.name()).as<std::string>();
            } else if (deviceName.find("GPU") != std::string::npos) {
                sStreamNums = exec_net->GetConfig(ov::num_streams.name()).as<std::string>();
                sThreadNums = exec_net->GetConfig(ov::compilation_num_threads.name()).as<std::string>();
            }

            // print CPU or GPU streams num and threads num
            if (!sStreamNums.empty() && !sThreadNums.empty()) {
                LOG_INFO_TAG("after load network, %s streamNums:%s, %s threadNums:%s",
                             deviceName.c_str(),
                             sStreamNums.c_str(),
                             deviceName.c_str(),
                             sThreadNums.c_str());
            }
        } catch (const IE::Exception&) {
            LOG_DEBUG_TAG("deviceName:%s cannot get streamNums and threadNums from exec_net", deviceName.c_str());
        }
        std::unique_lock<std::mutex> lock{load_mutex};
        executableNetworkPerDevice.insert({deviceName, exec_net});
        multiNetworkConfig.insert(deviceConfig.begin(), deviceConfig.end());
    };

    // Check if CPU is in device list
    auto iterCPU = std::find_if(metaDevices.begin(), metaDevices.end(), [&](DeviceInformation& d) {
        return d.deviceName.find("CPU") != std::string::npos;
    });
    // Load devices other than CPU first
    for (auto& p : metaDevices) {
        if (iterCPU != metaDevices.end() && p.deviceName == iterCPU->deviceName) {
            continue;
        }
        loads.push_back([&]() {
            loadInferEngTask(p);
        });
    }

    auto executor = executorManager()->getIdleCPUStreamsExecutor(
        IStreamsExecutor::Config{"MultiDeviceAsyncLoad",
                                 static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                                 0 /*default threads per stream, workaround for ticket 62376*/,
                                 IStreamsExecutor::ThreadBindingType::NONE});
    if (loads.size() > 0) {
        // Wait for the device to load the network
        executor->runAndWait(loads);
        loads.clear();
    }

    // Finally load the CPU
    if (iterCPU != metaDevices.end()) {
        if (!executableNetworkPerDevice.empty() && iterCPU->config.find(ov::affinity.name()) == iterCPU->config.end()) {
            LOG_DEBUG_TAG("set affinity to NUMA and disable hyper thread for CPU");
            // If the other devices load successfully and no user set affinity then set NUMA to CPU
            iterCPU->config.insert({ov::affinity.name(), ov::affinity(ov::Affinity::NUMA).second.as<std::string>()});
            iterCPU->config.insert({CONFIG_KEY_INTERNAL(ENABLE_HYPER_THREAD), CONFIG_VALUE(NO)});
        }
        loads.push_back([&]() {
            loadInferEngTask(*iterCPU);
        });
        // Wait for CPU to load the network
        executor->runAndWait(loads);
    }

    if (executableNetworkPerDevice.empty())
        IE_THROW(NotFound) << "Failed to load network to any device "
                           <<  "that the " << GetName() << " device is initialized to work with";

    // checking the perf counters config from the loaded network to respect both device's plugin and load-specific setting
    size_t num_plugins_supporting_perf_counters = 0;
    for (auto& n : executableNetworkPerDevice) {
            try {
                num_plugins_supporting_perf_counters +=
                        n.second->GetConfig(PluginConfigParams::KEY_PERF_COUNT).as<std::string>() ==
                        PluginConfigParams::YES;
            } catch (const IE::Exception&) {
            }
    }
    // MULTI can enable the perf counters only if all  devices support/enable that
    bool enablePerfCounters = num_plugins_supporting_perf_counters == executableNetworkPerDevice.size();
    multiSContext->_devicePriorities = metaDevices;
    multiSContext->_devicePrioritiesInitial = metaDevices;
    multiSContext->_networksPerDevice = executableNetworkPerDevice;
    multiSContext->_config = multiNetworkConfig;
    multiSContext->_needPerfCounters = enablePerfCounters;
    multiSContext->_core = GetCore();
    multiSContext->_LogTag = _LogTag;
    IExecutableNetworkInternal::Ptr impl;
    auto tmp = loadConfig.get_property(ov::intel_auto::device_bind_buffer);
    if (tmp) {
        multiSContext->_bindBuffer = true;
        impl = std::make_shared<MultiExecutableNetwork>(multiSContext, std::make_shared<BinderMultiSchedule>());
    } else {
        impl = std::make_shared<MultiExecutableNetwork>(multiSContext, std::make_shared<MultiSchedule>());
    }
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

    auto queryconfig = _pluginConfig;
    // updateFromMap will check config valid
    queryconfig.set_user_property(PreProcessConfig(config), (GetName() == "AUTO")? true : false);
    queryconfig.apply_user_properties();
    auto fullproperty = queryconfig.get_full_properties();
    // this can be updated when plugin switch to 2.0 API
    std::map<std::string, std::string> fullConfig =  ConvertToStringMap(fullproperty);;
    auto priorities = fullConfig.find(ov::device::priorities.name());
    if (!priorities->second.empty()) {
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
    auto deviceListConfig = config.find(ov::device::priorities.name());
    if (deviceListConfig->second.empty()) {
        for (auto&& device : deviceList) {
            // filter out the supported devices
            if (!_pluginConfig.isSupportedDevice(device))
                continue;
            allDevices += device + ",";
        }
    } else {
        auto priorities = deviceListConfig->second;
        // parsing the string and splitting the comma-separated tokens
        std::vector<std::string> deviceVec = _pluginConfig.ParsePrioritiesDevices(priorities);
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
                LOG_INFO_TAG("remove %s from device candidate list", iter.c_str());
                updateDeviceVec(iter);
            }
            for (auto&& device : deviceVec) {
                if (!_pluginConfig.isSupportedDevice(device))
                    continue;
                allDevices += device + ",";
            }
        }
    }

    // remove the last ',' if exist
    if (allDevices.back() == ',')
        allDevices.pop_back();

    if (allDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }

    return allDevices;
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
    }

    std::vector<DeviceInformation> filterDevice;
    auto model = network.getFunction();
    auto isStateful = [&]() {
        for (auto& op : model->get_ops()) {
            if (std::dynamic_pointer_cast<ngraph::op::AssignBase>(op) ||
                std::dynamic_pointer_cast<ngraph::op::ReadValueBase>(op)) {
                    LOG_INFO_TAG("stateful mode, try deployed to CPU");
                    return true;
                }
        }
        return false;
    };

    // Check if CPU is in candidate list
    auto cpuiter = std::find_if(metaDevices.begin(), metaDevices.end(), [](const DeviceInformation& deviceInfo) {
        return deviceInfo.deviceName.find("CPU") != std::string::npos;
    });

    // If CPU is in candidate list, load dynamic network to CPU first
    // For MULTI do not only load stateful network to CPU
    // For AUTO CTPUT only load stateful network to CPU
    if ((model->is_dynamic() || (isStateful() && _LogTag != "MULTI")) && cpuiter != metaDevices.end()) {
        filterDevice.push_back(*cpuiter);
        return filterDevice;
    }

    // If CPU is not in candidate list, continue to run selection logic regardless of whether the input network is a
    // dynamic network or not
    return metaDevices;
}
std::string MultiDeviceInferencePlugin::GetLogTag() const noexcept {
    return _LogTag;
}
}  // namespace MultiDevicePlugin
