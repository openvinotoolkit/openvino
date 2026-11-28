// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "auto_compiled_model.hpp"
#include "common.hpp"
#include <memory>

#include "async_infer_request.hpp"
#include "itt.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin.hpp"

namespace ov {
namespace auto_plugin {
namespace {
ActiveDevice require_active_device(AutoSchedule& scheduler, const std::string& log_tag) {
    auto active = scheduler.get_active_device();
    OPENVINO_ASSERT(active.compiled_model, log_tag, ": no compiled model is available");
    return active;
}
}  // namespace

AutoCompiledModel::AutoCompiledModel(const std::shared_ptr<ov::Model>& model,
                                     const std::shared_ptr<const ov::IPlugin>& plugin,
                                     const ov::SoPtr<ov::IRemoteContext>& remote_context,
                                     ScheduleContext::Ptr& schedule_context,
                                     Schedule::Ptr& scheduler)
    : CompiledModel(model, plugin, remote_context, schedule_context, scheduler),
      m_model(model) {
      m_scheduler = std::dynamic_pointer_cast<AutoSchedule>(scheduler);
}

void AutoCompiledModel::set_property(const ov::AnyMap& properties) {
        OPENVINO_THROW_NOT_IMPLEMENTED("It's not possible to set property of an already compiled model. "
                                       "Set property to Core::compile_model during compilation");
}

std::shared_ptr<const ov::Model> AutoCompiledModel::get_runtime_model() const {
    const auto active = require_active_device(*m_scheduler, get_log_tag());
    auto model = active.compiled_model->get_runtime_model();
    set_model_shared_object(const_cast<ov::Model&>(*model), active.compiled_model._so);
    return model;
}

ov::Any AutoCompiledModel::get_property(const std::string& name) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::supported_properties,
                                                    ov::execution_devices,
                                                    ov::hint::performance_mode,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::device::priorities,
                                                    ov::device::properties,
                                                    ov::hint::model_priority,
                                                    ov::loaded_from_cache,
                                                    ov::enable_profiling};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{};
        return rw_properties;
    };
    if (name == ov::supported_properties) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(std::move(supported_properties));
    } else if (name == ov::enable_profiling) {
        return m_context->m_need_perf_counters;
    } else if (name == ov::hint::performance_mode) {
        return m_context->m_performance_hint;
    } else if (name == ov::device::priorities) {
        // device priority does not support change on-the-fly
        return decltype(ov::device::priorities)::value_type(m_context->m_str_devices);
    } else if (name == ov::device::properties) {
        // the properties of the instance that serves the requests right now
        const auto active = require_active_device(*m_scheduler, get_log_tag());
        return get_device_supported_properties(active.instance.device, active.compiled_model);
    } else if (name == ov::hint::model_priority) {
        auto value = m_context->m_model_priority;
        return value ? ((value > 1) ? ov::hint::Priority::LOW : ov::hint::Priority::MEDIUM) : ov::hint::Priority::HIGH;
    } else if (name == ov::optimal_number_of_infer_requests) {
        if (m_context->m_dynamic_device_selection) {
            // inference is serialized in dynamic mode, so more than one request cannot run at a time
            return decltype(ov::optimal_number_of_infer_requests)::value_type{1u};
        }
        const unsigned int default_num_for_tput = 4u;
        const unsigned int default_num_for_latency = 1u;
        unsigned int real = 0;
        if (m_scheduler->m_compile_context[ACTUALDEVICE].is_ready()) {
            real = m_scheduler->m_compile_context[ACTUALDEVICE].compiled_model()->get_property(name).as<unsigned int>();
        } else {
            DeviceInformation device_info;
            {
                std::lock_guard<std::mutex> lock(m_context->m_mutex);
                device_info = m_scheduler->m_compile_context[ACTUALDEVICE].request.m_device_info;
            }
            unsigned int optimal_batch_size = 0;
            unsigned int requests = 0;
            bool tput_enabled_in_plugin = false;
            auto actual_dev_supported_properties =
                m_context->m_ov_core->get_property(device_info.device_name, ov::supported_properties);
            bool is_supported_num_request = std::find(actual_dev_supported_properties.begin(),
                                                      actual_dev_supported_properties.end(),
                                                      ov::hint::num_requests) != actual_dev_supported_properties.end();
            auto reqs_iter = device_info.config.find(ov::hint::num_requests.name());
            auto ireq_iter = std::find(actual_dev_supported_properties.begin(), actual_dev_supported_properties.end(), name);
            if (ireq_iter != actual_dev_supported_properties.end()) {
                real = m_context->m_ov_core->get_property(device_info.device_name,
                                                          ov::optimal_number_of_infer_requests,
                                                          device_info.config);
            }
            if (real > 0) {
                if (is_supported_num_request && reqs_iter != device_info.config.end()) {
                    requests = reqs_iter->second.as<unsigned int>();
                    real = requests > 0 ? (std::min)(requests, real) : real;
                }
                return decltype(ov::optimal_number_of_infer_requests)::value_type{real};
            }
            requests = 0;
            try {
                // for benchmark through AUTO:CPU,GPU
                // SetConfig directly set to CPU/GPU in this case
                if (std::find(actual_dev_supported_properties.begin(),
                              actual_dev_supported_properties.end(),
                              ov::hint::performance_mode) != actual_dev_supported_properties.end())
                    tput_enabled_in_plugin =
                        m_context->m_ov_core->get_property(device_info.device_name, ov::hint::performance_mode) ==
                        ov::hint::PerformanceMode::THROUGHPUT;
            } catch (const ov::Exception&) {
                LOG_DEBUG_TAG("get_property:%s for %s",
                              "PERF_HINT config not supported",
                              device_info.device_name.c_str());
            }
            const auto& mode = device_info.config.find(ov::hint::performance_mode.name());
            if (tput_enabled_in_plugin ||
                (mode != device_info.config.end() && mode->second == ov::hint::PerformanceMode::THROUGHPUT)) {
                unsigned int upper_bound_streams_num = 0;
                if (std::find(actual_dev_supported_properties.begin(),
                              actual_dev_supported_properties.end(),
                              ov::range_for_streams) != actual_dev_supported_properties.end()) {
                    try {
                        auto range_of_streams =
                            m_context->m_ov_core->get_property(device_info.device_name, ov::range_for_streams);
                        upper_bound_streams_num = std::get<1>(range_of_streams);
                    } catch (const ov::Exception&) {
                        LOG_DEBUG_TAG("get_property range_for_streams from %s failed", device_info.device_name.c_str());
                    }
                }
                if (!m_context->m_batching_disabled && m_model) {
                    if (std::find(actual_dev_supported_properties.begin(),
                                  actual_dev_supported_properties.end(),
                                  ov::optimal_batch_size) != actual_dev_supported_properties.end()) {
                        try {
                            optimal_batch_size = m_context->m_ov_core->get_property(device_info.device_name,
                                                                                    ov::optimal_batch_size,
                                                                                    {ov::hint::model(m_model)});
                            LOG_DEBUG_TAG("BATCHING:%s:%ld", "optimal batch size", optimal_batch_size);
                        } catch (const ov::Exception&) {
                            LOG_DEBUG_TAG("BATCHING:%s by device %s",
                                          "property optimal_batch_size not supported",
                                          device_info.device_name.c_str());
                        }
                    }
                }
                if (optimal_batch_size > 1) {
                    // batching is supported with the device
                    // go with auto-batching
                    try {
                        // check if app have set preferred value
                        requests = m_context->m_ov_core->get_property(device_info.device_name, ov::hint::num_requests);
                        const auto& reqs = device_info.config.find(ov::hint::num_requests.name());
                        if (reqs != device_info.config.end()) {
                            requests = reqs->second.as<unsigned int>();
                        }
                        LOG_DEBUG_TAG("BATCHING:%s:%ld", "user requested size", requests);
                        if (!requests) {  // no limitations from user
                            requests = optimal_batch_size * upper_bound_streams_num * 2;
                            LOG_DEBUG_TAG("BATCHING:%s:%ld", "deduced size:", requests);
                        }
                    } catch (const ov::Exception& iie) {
                        LOG_WARNING_TAG("deduce optimal infer requset num for auto-batch failed :%s", iie.what());
                    }
                    real = (std::max)(requests, optimal_batch_size);
                } else if (device_info.device_name.find("NPU") != std::string::npos) {
                    real = 8u;
                } else {
                    real = upper_bound_streams_num ? 2 * upper_bound_streams_num : default_num_for_tput;
                }
            } else {
                real = default_num_for_latency;
            }
            if (reqs_iter != device_info.config.end()) {
                requests = reqs_iter->second.as<unsigned int>();
                real = requests > 0 ? (std::min)(requests, real) : real;
            }
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type {real};
    } else if (name == ov::execution_devices) {
        const auto active = m_scheduler->get_active_device();
        if (!active.compiled_model) {
            return decltype(ov::execution_devices)::value_type{};
        }
        const auto& instance = active.instance;
        // the startup helper is reported in parentheses, it serves the requests only until the device is ready
        return decltype(ov::execution_devices)::value_type{
            instance.is_startup_helper() ? "(" + instance.device + ")" : instance.device};
    } else if (name == ov::model_name) {
        return m_context->m_hw_metadata->model_name;
    } else if (name == ov::loaded_from_cache) {
        const auto active = m_scheduler->get_active_device();
        try {
            return active.compiled_model ? active.compiled_model->get_property(name).as<bool>() : false;
        } catch (const ov::Exception&) {
            LOG_DEBUG_TAG("get_property loaded_from_cache from %s failed", active.instance.to_string().c_str());
            return false;
        }
    }
    OPENVINO_THROW(get_log_tag(), ": not supported property ", name);
}

void AutoCompiledModel::export_model(std::ostream& model_stream) const {
    OPENVINO_NOT_IMPLEMENTED;
}
} // namespace auto_plugin
} // namespace ov
