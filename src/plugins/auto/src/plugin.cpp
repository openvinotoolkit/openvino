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
#include "openvino/runtime/device_id_parser.hpp"
#include "plugin.hpp"
#include <ie_algorithm.hpp>
#include <ie_icore.hpp>
#include <ie_ngraph_utils.hpp>
#include "bind_multi_schedule.hpp"
#include "auto_executable_network.hpp"
#include "auto_schedule.hpp"
#include "auto_executable_network.hpp"

#include "itt.hpp"
namespace {
    std::string get_network_precision(const std::shared_ptr<const ov::Model> &model) {
        bool is_int_model = ov::op::util::has_op_with_type<ngraph::op::FakeQuantize>(model);
        if (is_int_model) {
            return METRIC_VALUE(INT8);
        }
        for (auto & node : model->get_ordered_ops()) {
            if (std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node) ||
                std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node) ||
                std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData>(node) ||
                std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
                auto layer_type = node->input(1).get_element_type().get_type_name();
                if (layer_type == "f32")
                    return METRIC_VALUE(FP32);
                if (layer_type == "f16")
                    return METRIC_VALUE(FP16);
            }
        }
        return METRIC_VALUE(FP32);
    }
    int map_priority_value(ov::hint::Priority priority) {
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
    std::map<std::string, std::string> convert_to_string_map(ov::AnyMap& properties) {
        std::map<std::string, std::string> configs;
        for (auto& property : properties) {
            configs[property.first] = property.second.as<std::string>();
        }
        return configs;
    }
}  // namespace

namespace ov {
namespace auto_plugin {

std::mutex Plugin::m_mtx;
std::map<unsigned int, std::list<std::string>> Plugin::m_priority_map;

ov::AnyMap Plugin::pre_process_config(const std::map<std::string, std::string>& orig_config) const {
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

std::vector<DeviceInformation> Plugin::parse_meta_devices(const std::string& priorities,
                                                          const ov::AnyMap& properties) const {
    std::vector<DeviceInformation> meta_devices;

    // parsing the string and splitting to tokens
    std::vector<std::string> devices_with_requests = m_plugin_config.parse_priorities_devices(priorities);

    auto set_default_hint = [&](const std::string& target_device,
                              ov::AnyMap& device_config,
                              const ov::AnyMap& properties) {
        auto isSetPerHint = properties.find(ov::hint::performance_mode.name()) != properties.end();
        auto isSetDeviceProperties = properties.find(target_device) != properties.end();
        if (get_device_name() == "AUTO" && !isSetPerHint && !isSetDeviceProperties) {
            // setting latency as the default performance mode if
            // 1. no hints setting for AUTO plugin
            // 2. no ov::device::properties(secondary properties) setting for target device
            device_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
            return;
        }

        if (get_device_name() == "MULTI") {
            auto isSetNumStreams = properties.find(ov::num_streams.name()) != properties.end();
            auto isSetAffinity = properties.find(ov::affinity.name()) != properties.end();
            auto isSetNumThreads = properties.find(ov::inference_num_threads.name()) != properties.end();
            if (!isSetPerHint && !isSetAffinity && !isSetNumThreads && !isSetDeviceProperties && !isSetNumStreams) {
                // setting tput as the default performance mode if
                // 1. no hints setting for MULTI plugin
                // 2. no affinity setting for MULTI plugin
                // 3. no inference_num_threads setting for MULTI plugin
                // 4. no ov::device::properties(secondary properties) setting for target device
                // 5. no ov::num_streams setting for target device
                device_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::THROUGHPUT;
            }
        }
    };

    auto get_device_config = [&] (const DeviceName & device_with_id) {
        auto device_config = get_core()->get_supported_property(device_with_id, properties);
        set_default_hint(device_with_id, device_config, properties);
        return device_config;
    };

    auto get_default_device_id = [this](std::string device_name) -> std::string {
        try {
            auto device_id = get_core()->get_property(device_name, ov::device::id);
            return device_id;
        } catch (ov::Exception& err) {
            LOG_DEBUG_TAG("get default device id failed for ", device_name.c_str());
            return "";
        }
    };
    auto check_priority_config = [&] (const std::string& pri_string) {
        if (pri_string.empty())
            return false;
        std::string::size_type pos = 0;
        std::string::size_type endpos = 0;
        while ((endpos = pri_string.find(",", pos)) != std::string::npos) {
            auto subStr = pri_string.substr(pos, endpos - pos);
            if (subStr.find("-") != 0)
                return true;
            pos = endpos + 1;
        }
        if (pri_string.substr(pos, pri_string.length() - pos).find("-") != 0 )
            return true;
        return false;
    };
    unsigned int device_priority = 0;
    auto prioritiesIter = properties.find(ov::device::priorities.name());
    // if AUTO:-***,-***...., also do not need to enable device priority
    bool enable_device_priority = (prioritiesIter != properties.end()) &&
                                check_priority_config(prioritiesIter->second.as<std::string>());

    auto device_list = get_core()->get_available_devices();
    for (auto && d : devices_with_requests) {
        auto opening_bracket = d.find_first_of('(');
        auto closing_bracket = d.find_first_of(')', opening_bracket);
        auto device_name = d.substr(0, opening_bracket);

        int num_requests = -1;
        if (closing_bracket != std::string::npos && opening_bracket < closing_bracket) {
            num_requests = std::stol(d.substr(opening_bracket + 1, closing_bracket - 1));

            if (num_requests <= 0) {
                OPENVINO_THROW("Priority value for", device_name, "must be > 0, while ", num_requests, "is passed");
            }
        }

        ov::DeviceIDParser parsed{device_name};
        std::string deviceid = parsed.get_device_id();
        std::vector<std::string> same_type_devices;
        // if AUTO:GPU case, replace GPU with GPU.0 and GPU.1
        if (deviceid.empty()) {
            for (auto&& device : device_list) {
                if (device.find(device_name) != std::string::npos) {
                    same_type_devices.push_back(std::move(device));
                }
            }
        }
        // it's a virtual device like HETERO, TEMPLATE
        // or real device with ID like GPU.1
        if (same_type_devices.size() == 0) {
            same_type_devices.push_back(std::move(device_name));
        }

        for (auto&& device_name_with_id : same_type_devices) {
            ov::DeviceIDParser new_parsed{device_name_with_id};
            std::string default_device_id = "";
            std::string temp_device_id = "";
            if (new_parsed.get_device_id().empty()) {
                default_device_id = get_default_device_id(device_name_with_id);
                temp_device_id = default_device_id;
            } else {
                temp_device_id = new_parsed.get_device_id();
            }

            std::string full_device_name = "";
            std::string unique_name = "";
            if (new_parsed.get_device_name() == "GPU") {
                try {
                    full_device_name = get_core()->get_property(device_name_with_id, ov::device::full_name);
                } catch (ov::Exception& err) {
                    LOG_DEBUG_TAG("get full device name failed for ", device_name_with_id.c_str());
                }
            }

            if (full_device_name.empty()) {
                unique_name = new_parsed.get_device_name() + "_" + temp_device_id;
            } else {
                unique_name = full_device_name + "_" + temp_device_id;
            }

            LOG_DEBUG_TAG("deviceNameWithID:%s, defaultDeviceID:%s, uniqueName:%s",
                    device_name_with_id.c_str(), default_device_id.c_str(), unique_name.c_str());
            // create meta device
            metaDevices.push_back({device_name_with_id, get_device_config(device_name_with_id), num_requests, default_device_id, unique_name, device_priority});
        }
        if (enable_device_priority) {
            device_priority++;
        }
    }

    return metaDevices;
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, m_plugin_config.supported_metrics(get_device_name()));
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, m_plugin_config.supported_config_keys(get_device_name()));
    } else if (ov::supported_properties == name) {
        auto ret = m_plugin_config.supported_properties(get_device_name());
        return ret;
    } else if (name == METRIC_KEY(OPTIMIZATION_CAPABILITIES)) {
        auto device_list = get_core()->get_available_devices();
        std::vector<std::string> capabilities;
        for (auto const & device : device_list) {
            auto devCapabilities = get_core()->get_property(device, ov::device::capabilities);
            capabilities.insert(capabilities.end(), devCapabilities.begin(), devCapabilities.end());
        }
        std::sort(capabilities.begin(), capabilities.end());
        capabilities.resize(std::distance(capabilities.begin(), std::unique(capabilities.begin(), capabilities.end())));
        auto delItem = std::find(capabilities.begin(), capabilities.end(), ov::device::capability::EXPORT_IMPORT);
        if (delItem != capabilities.end()) {
            capabilities.erase(delItem);
        }
        IE_SET_METRIC_RETURN(OPTIMIZATION_CAPABILITIES, capabilities);
    }
    auto val = m_plugin_config.get_property(name);
    if (!is_new_api()) {
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

void Plugin::set_property(const ov::AnyMap& properties) {
    // with setConfig, only multi/auto supported internal configs can be accepted
    m_plugin_config.set_property(properties);
}

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_auto_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::auto_plugin::Plugin, version)
// ! [plugin:create_plugin_engine]

Plugin::Plugin() {
    set_device_name("AUTO");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) {
    auto network_precision = get_network_precision(model);
    return compile_model_impl(model, properties, network_precision);                                                        

}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model_impl(const std::shared_ptr<const ov::Model>& model,
                                                                        const ov::AnyMap& properties,
                                                                        const std::string &networkPrecision = METRIC_VALUE(FP32)) {
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, "Plugin::compile_model");
    if (get_core() == nullptr) {
        OPENVINO_THROW("ov::core object not found", get_device_name());
    }
    m_log_tag = get_device_name();
    bool workModeAuto = get_device_name() == "AUTO";
    auto loadConfig = m_plugin_config;
    // if no perf hint from user with compiled model, or already been set with plugin
    // apply latency for AUTO, tput for MULTI
    auto itorConfig = properties.find(ov::hint::performance_mode.name());
    bool isHintSet = m_plugin_config.is_set_by_user(ov::hint::performance_mode) || itorConfig != properties.end();
    if (!isHintSet && workModeAuto) {
        // NO user sets perfHint, then set perfhint to 'LATENCY' for AutoExecutableNetwork.
        loadConfig.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    }
    // updateFromMap will check config valid
    loadConfig.set_user_property(properties);
    loadConfig.apply_user_properties();
    if (!workModeAuto) {
        if (itorConfig != properties.end() && itorConfig->second != ov::hint::PerformanceMode::THROUGHPUT) {
            LOG_WARNING_TAG("User set perf_hint:%s, but MULTI supports THROUGHPUT only", itorConfig->second.as<std::string>().c_str());
        }
        loadConfig.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    }
    auto fullProperty = loadConfig.get_full_properties();

    // Remove the performance hint as this is set by plugin logic, not from user
    if (!isHintSet)
        fullProperty.erase(ov::hint::performance_mode.name());
    if (!loadConfig.is_set_by_user(ov::hint::execution_mode))
        fullProperty.erase(ov::hint::execution_mode.name());
    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, ov::Any> multiNetworkConfig;
    std::vector<DeviceInformation> metaDevices;
    auto priorities = loadConfig.get_property(ov::device::priorities);
     if (priorities.empty() && !workModeAuto)
        OPENVINO_THROW("KEY_MULTI_DEVICE_PRIORITIES key is not set for ", get_device_name());
    if (priorities.find("AUTO") != std::string::npos || priorities.find("MULTI") != std::string::npos) {
        OPENVINO_THROW("The device candidate list should not include the meta plugin for ", get_device_name());
    }
    // check the configure and check if need to set PerfCounters configure to device
    // and set filter configure
    auto autoSContext = std::make_shared<AutoScheduleContext>();
    std::map<std::string, std::string> filterConfig;
    auto strDevices = get_device_list(fullConfig);
    // fill in the context for auto
    if (loadConfig.get_property(ov::enable_profiling)) {
        filterConfig.insert({ov::enable_profiling.name(), PluginConfigParams::YES});
        autoSContext->_needPerfCounters = true;
    }
    autoSContext->_modelPriority = MapPriorityValues(loadConfig.get_property(ov::hint::model_priority));
    autoSContext->_batchingDisabled = loadConfig.is_batching_disabled();
    // set performanceHint for AutoExecutableNetwork
    autoSContext->_performanceHint = loadConfig.get_property(ov::hint::performance_mode.name()).as<std::string>();
    // filter the device that supports filter configure
    metaDevices = ParseMetaDevices(strDevices, fullProperty);
    auto supportDevicesByConfig = FilterDevice(metaDevices, filterConfig);
    if (supportDevicesByConfig.empty()) {
        IE_THROW() << "There is no device support the configure";
    }
    auto supportDevices = supportDevicesByConfig;
    CNNNetwork clonedNetwork;
    std::string clonedModelPath = modelPath;
    // reset the strDevices to support devices
    strDevices = "";
}


IExecutableNetworkInternal::Ptr MultiDeviceInferencePlugin::LoadNetworkImpl(const std::string& modelPath,
                                                                              CNNNetwork network,
                                                                              const std::map<std::string, std::string>& config,
                                                                              const std::string &networkPrecision) {
    if (GetCore() == nullptr) {
        OPENVINO_THROW("ov::core object not found", get_device_name());
    }

    if (modelPath.empty() && network.getFunction() == nullptr) {
        IE_THROW() << GetName() << " device supports just ngraph network representation";
    }
    // to use plugin's name as the log tag
    _LogTag = GetName();
    bool workModeAuto = GetName() == "AUTO";
    auto loadConfig = m_plugin_config;
    // if no perf hint from user with compiled model, or already been set with plugin
    // apply latency for AUTO, tput for MULTI
    auto itorConfig = config.find(ov::hint::performance_mode.name());
    bool isHintSet = m_plugin_config.is_set_by_user(ov::hint::performance_mode) || itorConfig != config.end();
    if (!isHintSet && workModeAuto) {
        // NO user sets perfHint, then set perfhint to 'LATENCY' for AutoExecutableNetwork.
        loadConfig.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    }
    // updateFromMap will check config valid
    loadConfig.set_user_property(PreProcessConfig(config));
    loadConfig.apply_user_properties();
    if (!workModeAuto) {
        if (itorConfig != config.end() && itorConfig->second != InferenceEngine::PluginConfigParams::THROUGHPUT) {
            LOG_WARNING_TAG("User set perf_hint:%s, but MULTI supports THROUGHPUT only", itorConfig->second.c_str());
        }
        loadConfig.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    }
    auto fullProperty = loadConfig.get_full_properties();
    // this can be updated when plugin switch to 2.0 API
    std::map<std::string, std::string> fullConfig = ConvertToStringMap(fullProperty);
    // Remove the performance hint as this is set by plugin logic, not from user
    if (!isHintSet)
        fullConfig.erase(ov::hint::performance_mode.name());
    if (!loadConfig.is_set_by_user(ov::hint::execution_mode))
        fullConfig.erase(ov::hint::execution_mode.name());
    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, ov::Any> multiNetworkConfig;
    std::vector<DeviceInformation> metaDevices;
    auto priorities = loadConfig.get_property(ov::device::priorities);
    if (priorities.empty() && !workModeAuto)
        IE_THROW() << "KEY_MULTI_DEVICE_PRIORITIES key is not set for " << GetName() << " device";
    if (priorities.find("AUTO") != std::string::npos || priorities.find("MULTI") != std::string::npos) {
        IE_THROW() << "The device candidate list should not include the meta plugin for " << GetName() << " device";
    }
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
    autoSContext->_batchingDisabled = loadConfig.is_batching_disabled();
    // set performanceHint for AutoExecutableNetwork
    autoSContext->_performanceHint = loadConfig.get_property(ov::hint::performance_mode.name()).as<std::string>();
    // filter the device that supports filter configure
    metaDevices = ParseMetaDevices(strDevices, fullConfig);
    auto supportDevicesByConfig = FilterDevice(metaDevices, filterConfig);
    if (supportDevicesByConfig.empty()) {
        IE_THROW() << "There is no device support the configure";
    }
    auto supportDevices = supportDevicesByConfig;
    CNNNetwork clonedNetwork;
    std::string clonedModelPath = modelPath;
    // reset the strDevices to support devices
    strDevices = "";
    // calling GetValidDevices() to get a prioritized list of devices
    bool isCumulative =
        (autoSContext->_performanceHint == IE::PluginConfigParams::CUMULATIVE_THROUGHPUT) ? true : false;
    std::list<DeviceInformation> devicesWithPriority(supportDevices.begin(), supportDevices.end());
    if (modelPath.empty()) {
        // if network is valid
        LOG_INFO_TAG("load with CNN network");
        supportDevices = FilterDeviceByNetwork(supportDevicesByConfig, network);
        clonedNetwork = InferenceEngine::details::cloneNetwork(network);
        // clone the network, in case of reshape conflict
    } else {
        // model path, enable model load with single device situation
        if (supportDevices.size() > 1 && !isCumulative) {
            clonedNetwork = GetCore()->ReadNetwork(modelPath, std::string());
            // do we really need to disable model path?
            clonedModelPath = "";
            LOG_INFO_TAG("load with CNN network");
        } else {
            LOG_INFO_TAG("load with model path");
        }
    }
    if (!isCumulative) {
        devicesWithPriority = GetValidDevice(supportDevices, networkPrecision);
    }
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
    autoSContext->_startupfallback = loadConfig.get_property(ov::intel_auto::enable_startup_fallback);
    autoSContext->_runtimeFallback = loadConfig.get_property(ov::intel_auto::enable_runtime_fallback);
    IExecutableNetworkInternal::Ptr impl;
    // enable bind only in cumulative_throughput mode
    if (loadConfig.get_property(ov::intel_auto::device_bind_buffer) &&
        autoSContext->_performanceHint == "CUMULATIVE_THROUGHPUT") {
        LOG_INFO_TAG("runtime fallback set to disabled in binder mode");
        autoSContext->_runtimeFallback = false;
        impl = std::make_shared<AutoExecutableNetwork>(autoSContext, std::make_shared<BinderMultiSchedule>());
    } else {
        impl = std::make_shared<AutoExecutableNetwork>(autoSContext, std::make_shared<AutoSchedule>());
    }
    if (!modelPath.empty()) {
        SetExeNetworkInfo(impl,
                          autoSContext->_hwExecutableNetwork->GetInputsInfo(),
                          autoSContext->_hwExecutableNetwork->GetOutputsInfo());
        impl->setInputs(autoSContext->_hwExecutableNetwork->getInputs());
        impl->setOutputs(autoSContext->_hwExecutableNetwork->getOutputs());
    }
    return impl;
}

QueryNetworkResult Plugin::QueryNetwork(const CNNNetwork&                         network,
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

    auto queryconfig = m_plugin_config;
    // updateFromMap will check config valid
    queryconfig.set_user_property(PreProcessConfig(config));
    queryconfig.apply_user_properties();
    auto fullproperty = queryconfig.get_full_properties();
    // this can be updated when plugin switch to 2.0 API
    std::map<std::string, std::string> fullConfig =  ConvertToStringMap(fullproperty);
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

std::list<DeviceInformation> Plugin::GetValidDevice(
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
            std::string deviceType;
            try {
                deviceType = GetCore()->GetMetric(item.deviceName, METRIC_KEY(DEVICE_TYPE)).as<std::string>();
            } catch (const IE::Exception&) {
                LOG_DEBUG_TAG("GetMetric:%s for %s failed ", "DEVICE_TYPE", item.deviceName.c_str());
            }
            if (deviceType == "integrated") {
                iGPU.push_back(item);
            } else if (deviceType == "discrete") {
                dGPU.push_back(item);
            } else {
                LOG_DEBUG_TAG("Unknown device type for %s", item.deviceName.c_str());
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

DeviceInformation Plugin::SelectDevice(const std::vector<DeviceInformation>& meta_devices,
        const std::string& network_precision, unsigned int priority) {
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, "AutoPlugin::SelectDevice");

    std::list<DeviceInformation> valid_devices = GetValidDevice(meta_devices, network_precision);

    // all available Devices are in validDevices now
    // need to remove higher priority devices
    // save the last device first
    DeviceInformation last_device = valid_devices.back();
    {
        // begin to filter devices
        std::lock_guard<std::mutex> lck(_mtx);
        for (auto && kvp : m_priority_map) {
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
    register_priority(priority, ptrSelectDevice->uniqueName);
    return *ptrSelectDevice;
}

void Plugin::unregister_priority(const unsigned int& priority,
        const std::string& device_name) {
    std::lock_guard<std::mutex> lck(m_mtx);
    auto& priority_devices = m_priority_map[priority];
    for (auto iter = priority_devices.begin(); iter != priority_devices.end();) {
        if (*iter == device_name) {
            priority_devices.erase(iter);
            break;
        }
        iter++;
    }
}

void Plugin::register_priority(const unsigned int& priority,
        const std::string& device_name) {
    std::lock_guard<std::mutex> lck(m_mtx);
    auto& priority_devices = m_priority_map[priority];
    priority_devices.push_back(device_name);
}

std::string Plugin::get_device_list(const ov::AnyMap& properties) const {
    std::string allDevices;
    std::string device_architecture;
    auto device_list = get_core()->get_available_devices();
    auto device_list_config = properties.find(ov::device::priorities.name());
    auto get_gpu_architecture = [&](const std::string& name) -> std::string {
        try {
            auto architectureInfo = get_core.get_property(name, ov::device::architecture);
            return architectureInfo;
        } catch (const IE::Exception&) {
            LOG_DEBUG_TAG("GetMetric:%s for %s failed ", "DEVICE_ARCHITECTURE", name.c_str());
        }
        return "";
    };
    for (auto&& device : device_list) {
        // filter out the supported devices
        if (device.find("GPU") != std::string::npos) {
            device_architecture = get_gpu_architecture(device);
        }
        if (!m_plugin_config.is_supported_evice(device, device_architecture))
            continue;
        allDevices += device + ",";
    }
    std::vector<std::string> devices_merged;
    if (device_list_config != config.end() && !device_list_config->second.empty()) {
        auto priorities = device_list_config->second;
        // parsing the string and splitting the comma-separated tokens
        std::vector<std::string> devices_to_be_merged = m_plugin_config.parse_priorities_devices(priorities);
        std::vector<std::string> devices_to_be_deleted(devices_to_be_merged.size());
        const auto& iterDel = std::copy_if(devices_to_be_merged.begin(),
                                           devices_to_be_merged.end(),
                                           devices_to_be_deleted.begin(),
                                           [](const std::string& item) {
                                               return item.front() == '-';
                                           });
        devices_to_be_deleted.resize(std::distance(devices_to_be_deleted.begin(), iterDel));
        const auto& iter_merge =
            std::remove_if(devices_to_be_merged.begin(), devices_to_be_merged.end(), [](const std::string& item) {
                return item.front() == '-';
            });
        devices_to_be_merged.resize(std::distance(devices_to_be_merged.begin(), iter_merge));
        for (auto&& device : devices_to_be_deleted)
            LOG_INFO_TAG("remove %s from device candidate list", device.c_str());
        auto is_any_dev = [](std::string& device, const std::vector<std::string>& devices) {
            auto iter = std::find_if(devices.begin(), devices.end(), [device](const std::string& dev_item) {
                return dev_item.find(device) != std::string::npos;
            });
            return iter != devices.end();
        };
        auto is_any_dev_with_empty_merged = [](std::string& device, const std::vector<std::string>& devices) {
            auto iter = std::find_if(devices.begin(), devices.end(), [device](const std::string& dev_item) {
                std::string device_name = device;
                std::string::size_type real_end_pos = 0;
                if ((real_end_pos = device_name.find('.')) != std::string::npos && dev_item.find('.') == std::string::npos) {
                    device_name = device_name.substr(0, real_end_pos);
                }
                return dev_item.find(device_name) != std::string::npos;
            });
            return iter != devices.end();
        };
        auto device_with_default_id = [](std::string& device) {
            // AUTO assume the default device ID will be "0" for the single device.
            return device.find(".") == std::string::npos ? device + ".0" : device;
        };
        if (devices_to_be_merged.empty()) {
            for (auto&& device : device_list) {
                if (device.find("GPU") != std::string::npos) {
                    device_architecture = get_gpu_architecture(device);
                }
                if (is_any_dev_with_empty_merged(device, devices_to_be_deleted) || !m_plugin_config.is_supported_device(device, device_architecture))
                    continue;
                devices_merged.push_back(device);
            }
        } else {
            for (auto&& device : devices_to_be_merged) {
                if (!is_any_dev(device, device_list)) {
                    ov::DeviceIDParser parsed{device};
                    auto iter = std::find(devices_merged.begin(), devices_merged.end(), parsed.get_device_name());
                    if (iter != devices_merged.end() && parsed.get_device_name() != device && parsed.get_device_id() == "0")
                        // The device is the device with default device ID (eg. GPU.0) and
                        // its wide name (eg. GPU) has been in device candidate list.
                        continue;
                    // Add user specified device into candidate list
                    devices_merged.push_back(device);
                } else {
                    // Update device name if supported device with id existed
                    for (auto&& item : device_list) {
                        auto real_device = device_with_default_id(item);
                        if (is_any_dev(real_device, devices_to_be_deleted) || item.find(device) == std::string::npos)
                            continue;
                        auto iter = std::find(devices_merged.begin(), devices_merged.end(), device_with_default_id(item));
                        // Remove the device with default device id from candidate device list (eg. GPU.0)
                        // if its wide name is a single device (eg. GPU).
                        ov::DeviceIDParser parsed{item};
                        if (parsed.get_device_name() == item && iter != devices_merged.end())
                            devices_merged.erase(iter);
                        // continue if targe device has been in the candidate device list.
                        if (std::find(devices_merged.begin(), devices_merged.end(), item) != devices_merged.end())
                            continue;
                        devices_merged.push_back(item);
                    }
                }
            }
        }
        allDevices.clear();
        std::for_each(devices_merged.begin(), devices_merged.end(), [&allDevices](const std::string& device) {
            allDevices += device + ",";
        });
    }
    if (allDevices.empty()) {
        IE_THROW() << "Please, check environment due to no supported devices can be used";
    }
    // remove the last ',' if exist
    if (allDevices.back() == ',')
        allDevices.pop_back();

    return allDevices;
}

std::vector<DeviceInformation> Plugin::filter_device(const std::vector<DeviceInformation>& meta_devices,
        const std::map<std::string, std::string>& config) {
    if (meta_devices.empty()) {
        IE_THROW(NotFound) << "No available device to filter " << GetName() <<  " plugin";
    }

    if (config.size() == 0) {
        return metaDevices;
    }

    std::vector<DeviceInformation> filter_device;
    for (auto&& item : meta_devices) {
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
            filter_device.push_back(item);
        }
    }
    return filter_device;
}

std::vector<DeviceInformation> Plugin::filter_device_by_network(const std::vector<DeviceInformation>& meta_devices,
                                                InferenceEngine::CNNNetwork network) {
    if (meta_devices.empty()) {
        IE_THROW(NotFound) << "No available device to filter " << GetName() <<  " plugin";
    }

    std::vector<DeviceInformation> filter_device;
    auto model = network.getFunction();
    auto is_stateful = [&]() {
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
    auto cpuiter = std::find_if(meta_devices.begin(), meta_devices.end(), [](const DeviceInformation& device_info) {
        return device_info.device_name.find("CPU") != std::string::npos;
    });

    // If CPU is in candidate list, load dynamic network to CPU first
    // For MULTI do not only load stateful network to CPU
    // For AUTO CTPUT only load stateful network to CPU
    if (((model->is_dynamic()) || (is_stateful() && m_log_tag != "MULTI")) && cpuiter != meta_devices.end()) {
        filterDevice.push_back(*cpuiter);
        return filterDevice;
    }

    // If CPU is not in candidate list, continue to run selection logic regardless of whether the input network is a
    // dynamic network or not
    return metaDevices;
}

std::string Plugin::get_log_tag() const noexcept {
    return m_log_tag;
}
} // namespace auto_plugin
} // namespace ov
