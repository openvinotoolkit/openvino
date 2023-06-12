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

#include <threading/ie_executor_manager.hpp>
#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/device_id_parser.hpp"
#include "plugin.hpp"
//#include "bind_multi_schedule.hpp"
#include "auto_schedule.hpp"
#include "auto_executable.hpp"
#include "cumulative_executable.hpp"
#include "cumulative_schedule.hpp"
#include "itt.hpp"

namespace {
    const std::string get_network_precision(const std::shared_ptr<const ov::Model> &model) {
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
    template <typename T>
    T inter_section(const T& lhs, const T& rhs) {
        T result;
        const auto& min_set = (lhs.size() < rhs.size()) ? lhs : rhs;
        const auto& max_set = (lhs.size() >= rhs.size()) ? lhs : rhs;
        for (auto&& val : min_set) {
            if (max_set.find(val) != max_set.end()) {
                result.insert(val);
            }
        }
        return result;
    }
}  // namespace

namespace ov {
namespace auto_plugin {

std::mutex Plugin::m_mtx;
std::map<unsigned int, std::list<std::string>> Plugin::m_priority_map;

std::shared_ptr<ov::IRemoteContext> Plugin::create_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::IRemoteContext> Plugin::get_default_context(const ov::AnyMap& remote_properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::import_model(std::istream& model,
                                                         const ov::RemoteContext& context,
                                                         const ov::AnyMap& properties) const {
    OPENVINO_NOT_IMPLEMENTED;
}

ov::AnyMap Plugin::pre_process_config(const ov::AnyMap& orig_config) const {
    ov::AnyMap properties = orig_config;
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
        auto is_set_perfhint = properties.find(ov::hint::performance_mode.name()) != properties.end();
        auto is_set_deviceproperties = properties.find(target_device) != properties.end();
        if (get_device_name() == "AUTO" && !is_set_perfhint && !is_set_deviceproperties) {
            // setting latency as the default performance mode if
            // 1. no hints setting for AUTO plugin
            // 2. no ov::device::properties(secondary properties) setting for target device
            device_config[ov::hint::performance_mode.name()] = ov::hint::PerformanceMode::LATENCY;
            return;
        }

        if (get_device_name() == "MULTI") {
            auto is_set_numstreams = properties.find(ov::num_streams.name()) != properties.end();
            auto is_set_affinity = properties.find(ov::affinity.name()) != properties.end();
            auto is_set_numthreads = properties.find(ov::inference_num_threads.name()) != properties.end();
            if (!is_set_perfhint && !is_set_affinity && !is_set_numthreads && !is_set_deviceproperties && !is_set_numstreams) {
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
        } catch (const InferenceEngine::Exception& e) {
            // some may throw IE exceptions
            LOG_DEBUG_TAG("get default device id failed for ", device_name.c_str());
            return "";
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
            meta_devices.push_back({device_name_with_id, get_device_config(device_name_with_id),
                                    num_requests, default_device_id, unique_name, device_priority});
        }
        if (enable_device_priority) {
            device_priority++;
        }
    }

    return meta_devices;
}

ov::Any Plugin::get_property(const std::string& name, const ov::AnyMap& arguments) const {
    if (METRIC_KEY(SUPPORTED_METRICS) == name) {
        return m_plugin_config.supported_metrics(get_device_name());
    } else if (METRIC_KEY(SUPPORTED_CONFIG_KEYS) == name) {
        return m_plugin_config.supported_config_keys(get_device_name());
    } else if (ov::supported_properties == name) {
        auto ret = m_plugin_config.supported_properties(get_device_name());
        return ret;
    } else if (name == ov::device::full_name) {
        std::string device_name = { get_device_name() };
        return decltype(ov::device::full_name)::value_type {device_name};
    } else if (name == ov::device::capabilities.name()) {
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
        return capabilities;
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
    m_plugin_config.set_property(pre_process_config(properties));
}

// ! [plugin:create_plugin_engine]
static const ov::Version version = {CI_BUILD_NUMBER, "openvino_auto_plugin"};
OV_DEFINE_PLUGIN_CREATE_FUNCTION(ov::auto_plugin::Plugin, version)
// ! [plugin:create_plugin_engine]

Plugin::Plugin() {
    set_device_name("AUTO");
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                              const ov::AnyMap& properties,
                                                              const ov::RemoteContext& context) const {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::shared_ptr<const ov::Model>& model,
                                                          const ov::AnyMap& properties) const {
    auto network_precision = get_network_precision(model);
    return compile_model_impl({}, model, properties, network_precision);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model(const std::string& model_path,
                                                          const ov::AnyMap& properties) const {
    return compile_model_impl(model_path, nullptr, properties);
}

std::shared_ptr<ov::ICompiledModel> Plugin::compile_model_impl(const std::string& model_path,
                                                               const std::shared_ptr<const ov::Model>& model,
                                                               const ov::AnyMap& properties,
                                                               const std::string& network_precision) const {
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, "Plugin::compile_model");
    OPENVINO_ASSERT(get_core() , "OpenVINO Core is missing!");
    if (model_path.empty() && model == nullptr)
        OPENVINO_THROW("OpenVino Model is empty!");
    bool work_mode_auto = get_device_name() == "AUTO";
    auto load_config = m_plugin_config;
    // if no perf hint from user with compiled model, or already been set with plugin
    // apply latency for AUTO, tput for MULTI
    auto iter_config = properties.find(ov::hint::performance_mode.name());
    bool is_hint_set = m_plugin_config.is_set_by_user(ov::hint::performance_mode) || iter_config != properties.end();
    if (!is_hint_set && work_mode_auto) {
        // NO user sets perfHint, then set perfhint to 'LATENCY' for AutoExecutableNetwork.
        load_config.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    }
    // updateFromMap will check config valid
    load_config.set_user_property(pre_process_config(properties));
    load_config.apply_user_properties();
    if (!work_mode_auto) {
        if (iter_config != properties.end() && iter_config->second != ov::hint::PerformanceMode::THROUGHPUT) {
            LOG_WARNING_TAG("User set perf_hint:%s, but MULTI supports THROUGHPUT only", iter_config->second.as<std::string>().c_str());
        }
        load_config.set_property(ov::hint::performance_mode(ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT));
    }
    auto full_property = load_config.get_full_properties();

    // Remove the performance hint as this is set by plugin logic, not from user
    if (!is_hint_set)
        full_property.erase(ov::hint::performance_mode.name());
    if (!load_config.is_set_by_user(ov::hint::execution_mode))
        full_property.erase(ov::hint::execution_mode.name());
    // collect the settings that are applicable to the devices we are loading the network to
    std::unordered_map<std::string, ov::Any> multi_network_config;
    std::vector<DeviceInformation> meta_devices;
    auto priorities = load_config.get_property(ov::device::priorities);
     if (priorities.empty() && !work_mode_auto)
        OPENVINO_THROW("KEY_MULTI_DEVICE_PRIORITIES key is not set for ", get_device_name());
    if (priorities.find("AUTO") != std::string::npos || priorities.find("MULTI") != std::string::npos) {
        OPENVINO_THROW("The device candidate list should not include the meta plugin for ", get_device_name());
    }
    // check the configure and check if need to set PerfCounters configure to device
    // and set filter configure
    auto auto_s_context = std::make_shared<ScheduleContext>();
    ov::AnyMap filter_property;
    auto str_devices = get_device_list(full_property);
    // fill in the context for auto
    if (load_config.get_property(ov::enable_profiling)) {
        filter_property.insert({ov::enable_profiling(true)});
        auto_s_context->m_need_perf_counters = true;
    }
    auto_s_context->m_model_priority = map_priority_value(load_config.get_property(ov::hint::model_priority));
    auto_s_context->m_batching_disabled = load_config.is_batching_disabled();
    // set performanceHint for AutoExecutableNetwork
    auto_s_context->m_performance_hint = load_config.get_property(ov::hint::performance_mode.name());
    // filter the device that supports filter configure
    meta_devices = parse_meta_devices(str_devices, full_property);
    auto support_devices_by_property = filter_device(meta_devices, filter_property);
    if (support_devices_by_property.empty()) {
        OPENVINO_THROW("There is no device support the current configure");
    }
    auto support_devices = support_devices_by_property;
    // reset the str_devices to support devices
    str_devices = "";
    bool is_cumulative =
        (auto_s_context->m_performance_hint == ov::hint::PerformanceMode::CUMULATIVE_THROUGHPUT) ? true : false;
    std::list<DeviceInformation> devices_with_priority(support_devices.begin(), support_devices.end());
    std::shared_ptr<ov::Model> cloned_model;
    if (model_path.empty()) {
        support_devices = filter_device_by_network(support_devices_by_property, model);
        cloned_model = model->clone();
    } else {
        // AUTO / MULTI don't support caching explicitly, but can redirect this functionality to actual HW plugin
        LOG_INFO_TAG("compile model with model path");
    }
    if (!is_cumulative) {
        devices_with_priority = get_valid_device(support_devices, network_precision);
    }
    for (auto iter = devices_with_priority.begin(); iter != devices_with_priority.end(); iter++) {
        str_devices += iter->device_name;
        str_devices += ",";
    }
    str_devices.pop_back();
    for (auto iter = support_devices.begin(); iter != support_devices.end(); iter++) {
        auto& configs = iter->config;
        for (auto& config : configs) {
            LOG_INFO_TAG("device:%s, config:%s=%s",
                         iter->device_name.c_str(),
                         config.first.c_str(),
                         config.second.as<std::string>().c_str());
        }
        LOG_INFO_TAG("device:%s, priority:%ld", iter->device_name.c_str(), iter->device_priority);
    }
    // clone the network, in case of reshape conflict
    auto_s_context->m_model = cloned_model;
    auto_s_context->m_model_path = model_path;
    auto_s_context->m_device_priorities = support_devices;
    auto_s_context->m_device_priorities_initial = support_devices;
    auto_s_context->m_str_devices = str_devices;
    auto_s_context->m_plugin = shared_from_this();
    auto_s_context->m_ov_core = get_core();
    OPENVINO_ASSERT(auto_s_context->m_ov_core);
    auto_s_context->m_log_tag = get_device_name();
    auto_s_context->m_network_precision = network_precision;
    auto_s_context->m_startup_fallback = load_config.get_property(ov::intel_auto::enable_startup_fallback);
    auto_s_context->m_runtime_fallback = load_config.get_property(ov::intel_auto::enable_runtime_fallback);
    auto_s_context->m_bind_buffer = load_config.get_property(ov::intel_auto::device_bind_buffer);
    std::shared_ptr<ov::ICompiledModel> impl;
    if (is_cumulative) {
        impl = std::make_shared<AutoCumuCompiledModel>(cloned_model, shared_from_this(), auto_s_context, std::make_shared<CumuSchedule>());
    } else {
        impl = std::make_shared<AutoCompiledModel>(cloned_model, shared_from_this(), auto_s_context, std::make_shared<AutoSchedule>());
    }
    return impl;
}

ov::SupportedOpsMap Plugin::query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const {
    OPENVINO_ASSERT(model, "OpenVINO Model is empty!");
    OPENVINO_ASSERT(get_core(), "Core is missing!");
    ov::SupportedOpsMap res;

    auto queryconfig = m_plugin_config;
    // updateFromMap will check config valid
    queryconfig.set_user_property(properties);
    queryconfig.apply_user_properties();
    auto full_property = queryconfig.get_full_properties();
    auto priorities = full_property.find(ov::device::priorities.name());
    if (!priorities->second.empty()) {
        auto meta_devices = parse_meta_devices(priorities->second.as<std::string>(), full_property);
        std::unordered_set<std::string> supported_layers;
        for (auto&& value : meta_devices) {
            auto device_qm = get_core()->query_model(model, value.device_name, value.config);
            std::unordered_set<std::string> device_supported_layers;
            for (auto&& layer_qm : device_qm) {
                device_supported_layers.emplace(layer_qm.first);
            }
            supported_layers = supported_layers.empty()
                            ? device_supported_layers : (device_supported_layers.empty()
                            ? supported_layers : inter_section(supported_layers, device_supported_layers));
        }
        for (auto&& iter : supported_layers) {
            res[iter] = get_device_name();
        }
    }
    return res;
}

std::list<DeviceInformation> Plugin::get_valid_device(
    const std::vector<DeviceInformation>& meta_devices,
    const std::string& network_precision) const {
    if (meta_devices.empty()) {
        OPENVINO_THROW("No available device to select in ", get_device_name());
    }

    std::list<DeviceInformation> CPU;
    std::list<DeviceInformation> dGPU;
    std::list<DeviceInformation> iGPU;
    std::list<DeviceInformation> MYRIAD;
    std::list<DeviceInformation> VPUX;

    for (auto& item : meta_devices) {
        if (item.device_name.find("CPU") == 0) {
            CPU.push_back(item);
            continue;
        }
        if (item.device_name.find("MYRIAD") == 0) {
            MYRIAD.push_back(item);
            continue;
        }
        if (item.device_name.find("VPUX") == 0) {
            VPUX.push_back(item);
            continue;
        }
        if (item.device_name.find("GPU") == 0) {
            ov::Any device_type;
            try {
                device_type = get_core()->get_property(item.device_name, ov::device::type);
            } catch (const ov::Exception&) {
                LOG_DEBUG_TAG("GetMetric:%s for %s failed ", "DEVICE_TYPE", item.device_name.c_str());
            }
            if (device_type == ov::device::Type::INTEGRATED) {
                iGPU.push_back(item);
            } else if (device_type == ov::device::Type::DISCRETE) {
                dGPU.push_back(item);
            } else {
                LOG_DEBUG_TAG("Unknown device type for %s", item.device_name.c_str());
            }
            continue;
        }
    }

    // Priority of selecting device: dGPU > VPUX > iGPU > MYRIAD > CPU
    std::list<DeviceInformation> devices;
    if (network_precision == "INT8") {
        devices.splice(devices.end(), VPUX);
        devices.splice(devices.end(), dGPU);
    } else {
        devices.splice(devices.end(), dGPU);
        devices.splice(devices.end(), VPUX);
    }
    devices.splice(devices.end(), iGPU);
    devices.splice(devices.end(), MYRIAD);
    devices.splice(devices.end(), CPU);

    std::list<DeviceInformation> valid_devices;

    if (meta_devices.size() > 1) {
        auto select_support_dev = [this, &devices, &valid_devices](const std::string& networkPrecision) {
            for (auto iter = devices.begin(); iter != devices.end();) {
                auto capability = get_core()->get_property(iter->device_name, ov::device::capabilities);
                auto support_network = std::find(capability.begin(), capability.end(), (networkPrecision));
                if (support_network != capability.end()) {
                    valid_devices.push_back(std::move(*iter));
                    devices.erase(iter++);
                    continue;
                }
                iter++;
            }
        };
        select_support_dev(network_precision);
        // If network is FP32, continue to collect the device support FP16 but not support FP32.
        if (network_precision == "FP32") {
            const std::string f16 = "FP16";
            select_support_dev(f16);
        }
    } else {
        valid_devices.push_back(meta_devices[0]);
    }

    if (valid_devices.empty()) {
        OPENVINO_THROW("Cannot select any device");
    }
    // sort validDevices
    valid_devices.sort([](const DeviceInformation& a, const DeviceInformation& b) {
        return a.device_priority < b.device_priority;
    });

    return valid_devices;
}

DeviceInformation Plugin::select_device(const std::vector<DeviceInformation>& meta_devices,
        const std::string& network_precision, unsigned int priority) {
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, "Plugin::SelectDevice");

    std::list<DeviceInformation> valid_devices = get_valid_device(meta_devices, network_precision);

    // all available Devices are in valid_devices now
    // need to remove higher priority devices
    // save the last device first
    DeviceInformation last_device = valid_devices.back();
    {
        // begin to filter devices
        std::lock_guard<std::mutex> lck(m_mtx);
        for (auto && kvp : m_priority_map) {
            if (kvp.first >= priority) {
                continue;
            }
            auto& filter_devices = kvp.second;
            auto sd = std::remove_if(valid_devices.begin(), valid_devices.end(), [&filter_devices](const DeviceInformation& device) {
                    auto iter = std::find_if(filter_devices.begin(), filter_devices.end(), [&device](std::string uniqueName) {
                            return (uniqueName == device.unique_name);
                            });
                    return iter != filter_devices.end() ? true : false;
                    });
            valid_devices.erase(sd, valid_devices.end());
        }
    }

    DeviceInformation* ptr_select_device =  NULL;
    if (valid_devices.empty()) {
        // after remove higher priority device,but the available devices is null,
        // so select the last device of all available Devices.
        ptr_select_device = &last_device;
    } else {
        // select the first device in the rest of available devices.
        ptr_select_device = &valid_devices.front();
    }
    //recode the device priority
    register_priority(priority, ptr_select_device->unique_name);
    return *ptr_select_device;
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
            auto architectureInfo = get_core()->get_property(name, ov::device::architecture);
            return architectureInfo;
        } catch (const ov::Exception&) {
            LOG_DEBUG_TAG("GetMetric:%s for %s failed ", "DEVICE_ARCHITECTURE", name.c_str());
        }
        return "";
    };
    for (auto&& device : device_list) {
        // filter out the supported devices
        if (device.find("GPU") != std::string::npos) {
            device_architecture = get_gpu_architecture(device);
        }
        if (!m_plugin_config.is_supported_device(device, device_architecture))
            continue;
        allDevices += device + ",";
    }
    std::vector<std::string> devices_merged;
    if (device_list_config != properties.end() && !device_list_config->second.empty()) {
        auto priorities = device_list_config->second;
        // parsing the string and splitting the comma-separated tokens
        std::vector<std::string> devices_to_be_merged = m_plugin_config.parse_priorities_devices(priorities.as<std::string>());
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
        OPENVINO_THROW("Please, check environment due to no supported devices can be used");
    }
    // remove the last ',' if exist
    if (allDevices.back() == ',')
        allDevices.pop_back();

    return allDevices;
}

std::vector<DeviceInformation> Plugin::filter_device(const std::vector<DeviceInformation>& meta_devices,
        const ov::AnyMap& properties) const {
    if (meta_devices.empty()) {
        OPENVINO_THROW("No available device to filter ", get_device_name(), " plugin");
    }

    if (properties.size() == 0) {
        return meta_devices;
    }

    std::vector<DeviceInformation> filter_device;
    for (auto&& item : meta_devices) {
        bool support = true;
        try {
            auto support_keys = get_core()->get_property(item.device_name, ov::supported_properties);
            for (auto&& kvp : properties) {
                auto target_key = std::find(support_keys.begin(), support_keys.end(), kvp.first);
                // if device have the key, and is a mutable key, we think the device support it
                if (target_key != support_keys.end() && target_key->is_mutable()) {
                    continue;
                } else {
                    support = false;
                    break;
                }
            }
        } catch (ov::Exception&) {
            support = false;
        }
        if (support) {
            filter_device.push_back(item);
        }
    }
    return filter_device;
}

std::vector<DeviceInformation> Plugin::filter_device_by_network(const std::vector<DeviceInformation>& meta_devices,
                                                                const std::shared_ptr<const ov::Model>& model) const {
    if (meta_devices.empty()) {
        OPENVINO_THROW("No available device to filter ", get_device_name(), " plugin");
    }

    std::vector<DeviceInformation> filter_device;
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
    if (((model->is_dynamic()) || (is_stateful() && get_device_name() != "MULTI")) && cpuiter != meta_devices.end()) {
        filter_device.push_back(*cpuiter);
        return filter_device;
    }

    // If CPU is not in candidate list, continue to run selection logic regardless of whether the input network is a
    // dynamic network or not
    return meta_devices;
}

std::string Plugin::get_log_tag() const noexcept {
    return get_device_name();
}
} // namespace auto_plugin
} // namespace ov
