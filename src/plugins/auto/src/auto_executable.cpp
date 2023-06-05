// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "auto_executable.hpp"
#include "common.hpp"
#include <memory>

#include "async_infer_request.hpp"
#include "itt.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin.hpp"
#include <ie_performance_hints.hpp>

namespace ov {
namespace auto_plugin {
AutoCompiledModel::AutoCompiledModel(const std::shared_ptr<ov::Model>& model,
                                                      const std::shared_ptr<const ov::IPlugin>& plugin,
                                                      ScheduleContext::Ptr context,
                                                      Schedule::Ptr scheduler)
    : CompiledModel(model, plugin, context, scheduler),
      m_model(model),
      m_context(context) {
      m_scheduler = std::dynamic_pointer_cast<AutoSchedule>(scheduler);
}

void AutoCompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> AutoCompiledModel::get_runtime_model() const {
    OPENVINO_ASSERT(m_context->m_hw_compiled_model);
    return m_context->m_hw_compiled_model->get_runtime_model();
}

ov::Any AutoCompiledModel::get_property(const std::string& name) const {
    const auto& add_ro_properties = [](const std::string& name, std::vector<ov::PropertyName>& properties) {
        properties.emplace_back(ov::PropertyName{name, ov::PropertyMutability::RO});
    };
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::supported_properties,
                                                    ov::execution_devices,
                                                    ov::hint::performance_mode,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::device::priorities,
                                                    ov::device::properties,
                                                    ov::hint::model_priority,
                                                    ov::loaded_from_cache};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{};
        return rw_properties;
    };
    const auto& to_string_vector = [](const std::vector<ov::PropertyName>& properties) {
        std::vector<std::string> ret;
        for (const auto& property : properties) {
            ret.emplace_back(property);
        }
        return ret;
    };
    if (name == ov::supported_properties) {
        auto ro_properties = default_ro_properties();
        auto rw_properties = default_rw_properties();

        std::vector<ov::PropertyName> supported_properties;
        supported_properties.reserve(ro_properties.size() + rw_properties.size());
        supported_properties.insert(supported_properties.end(), ro_properties.begin(), ro_properties.end());
        supported_properties.insert(supported_properties.end(), rw_properties.begin(), rw_properties.end());
        return decltype(ov::supported_properties)::value_type(supported_properties);
    } else if (name == ov::hint::performance_mode) {
        return m_context->m_performance_hint;
    } else if (name == ov::device::priorities) {
        // device priority does not support change on-the-fly
        return decltype(ov::device::priorities)::value_type(m_context->m_str_devices);
    } else if (name == ov::device::properties) {
        ov::AnyMap all_devices = {};
        {
            std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
            if (m_scheduler->m_loadcontext[FALLBACKDEVICE].m_is_already) {
                all_devices = get_device_supported_metrics(m_scheduler->m_loadcontext[FALLBACKDEVICE]);
            }
        }
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        if (m_scheduler->m_loadcontext[ACTUALDEVICE].m_is_already) {
            all_devices = get_device_supported_metrics(m_scheduler->m_loadcontext[ACTUALDEVICE]);
        } else {
            all_devices = get_device_supported_metrics(m_scheduler->m_loadcontext[CPU]);
        }
        return all_devices;
    } else if (name == ov::hint::model_priority) {
        auto value = m_context->m_model_priority;
        if (m_context->m_ov_core->is_new_api()) {
            return value ? ((value > 1) ? ov::hint::Priority::LOW :
                    ov::hint::Priority::MEDIUM) : ov::hint::Priority::HIGH;
        } else {
            return value ? ((value > 1) ? CONFIG_VALUE(MODEL_PRIORITY_LOW) : CONFIG_VALUE(
                        MODEL_PRIORITY_MED)) : CONFIG_VALUE(MODEL_PRIORITY_HIGH);
        }
    } else if (name == ov::optimal_number_of_infer_requests) {
        const unsigned int defaultNumForTPUT = 4u;
        const unsigned int defaultNumForLatency = 1u;
        unsigned int real = 0;
        if (m_scheduler->m_loadcontext[ACTUALDEVICE].m_is_already) {
            real = m_scheduler->m_loadcontext[ACTUALDEVICE].
                m_exe_network->get_property(name).as<unsigned int>();
        } else {
            OPENVINO_ASSERT(m_scheduler->m_loadcontext[CPU].m_is_already == true);
            std::unique_lock<std::mutex> lock(m_context->m_mutex);
            auto device_info = m_scheduler->m_loadcontext[ACTUALDEVICE].m_device_info;
            lock.unlock();
            unsigned int optimalBatchSize = 0;
            unsigned int requests = 0;
            bool bThroughputEnabledInPlugin = false;
            try {
                // for benchmark through AUTO:CPU,GPU
                // SetConfig directly set to CPU/GPU in this case
                bThroughputEnabledInPlugin =
                    m_context->m_ov_core->get_property(device_info.device_name,
                        ov::hint::performance_mode)== ov::hint::PerformanceMode::THROUGHPUT;
            } catch (const ov::Exception&) {
                LOG_DEBUG_TAG("get_property:%s for %s", "PERF_HINT config not supported",
                    device_info.device_name.c_str());
            }
            const auto& mode = device_info.config.find(ov::hint::performance_mode.name());
            if (bThroughputEnabledInPlugin ||
                (mode != device_info.config.end() && mode->second == ov::hint::PerformanceMode::THROUGHPUT)) {
                unsigned int upperBoundStreamsNum = 0;
                try {
                    auto rangeOfStreams = m_context->m_ov_core->get_property(device_info.device_name,
                            ov::range_for_streams);
                    upperBoundStreamsNum = std::get<1>(rangeOfStreams);
                } catch (const ov::Exception&) {
                    LOG_DEBUG_TAG("get_property RANGE_FOR_STREAMS failed");
                }
                if (!m_context->m_batching_disabled) {
                    try {
                        if (true) {//ov::details::is_model_batchable(m_model, device_info.device_name, true)
                            //!= ov::details::NetworkBatchAbility::NO) {
                            optimalBatchSize = m_context->m_ov_core->get_property(device_info.device_name,
                                    ov::optimal_batch_size, {ov::hint::model(m_model)});
                            LOG_DEBUG_TAG("BATCHING:%s:%ld", "optimal batch size",
                                optimalBatchSize);
                        }
                    } catch (const ov::Exception&) {
                        LOG_DEBUG_TAG("BATCHING:%s", "metric OPTIMAL_BATCH_SIZE not supported");
                    }
                }
                if (optimalBatchSize > 1) {
                    // batching is supported with the device
                    // go with auto-batching
                    try {
                        // check if app have set preferred value
                        auto res =
                            m_context->m_ov_core->get_property(device_info.device_name, ov::hint::num_requests);
                        requests = InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue(std::to_string(res));
                        const auto& reqs = device_info.config.find(ov::hint::num_requests.name());
                        if (reqs != device_info.config.end()) {
                            requests = static_cast<unsigned int>
                                (InferenceEngine::PerfHintsConfig::CheckPerformanceHintRequestValue((reqs->second).as<std::string>()));
                        }
                        LOG_DEBUG_TAG("BATCHING:%s:%ld", "user requested size", requests);
                        if (!requests) { // no limitations from user
                            requests = optimalBatchSize * upperBoundStreamsNum * 2;
                            LOG_DEBUG_TAG("BATCHING:%s:%ld", "deduced size:", requests);
                        }
                    } catch (const ov::Exception& iie) {
                        LOG_WARNING_TAG("deduce optimal infer requset num for auto-batch failed :%s",
                            iie.what());
                    }
                    real = (std::max)(requests, optimalBatchSize);
                } else if (device_info.device_name.find("VPUX") != std::string::npos) {
                    real = 8u;
                } else {
                    real = upperBoundStreamsNum ? 2 * upperBoundStreamsNum : defaultNumForTPUT;
                }
            } else {
                real = defaultNumForLatency;
            }
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type {real};
    } else if (name == ov::execution_devices) {
        ov::Any execution_devices;
        auto GetExecutionDevices = [&execution_devices](std::string ExeDevicesString) {
            std::vector<std::string> exeDevices = {};
            if (ExeDevicesString == "CPU_HELP")
                ExeDevicesString = "(CPU)";
            exeDevices.push_back(ExeDevicesString);
            execution_devices = decltype(ov::execution_devices)::value_type {exeDevices};
        };
        {
            std::lock_guard<std::mutex> lock(m_context->m_mutex);
            for (int i = 0; i < CONTEXTNUM; i++) {
                if (m_scheduler->m_loadcontext[i].m_is_enabled && m_scheduler->m_loadcontext[i].m_is_already) {
                    if (i == 0 && !m_scheduler->m_loadcontext[CPU].m_exe_network._ptr) {
                        continue;
                    } else {
                        GetExecutionDevices(m_scheduler->m_loadcontext[i].m_worker_name);
                        break;
                    }
                }
            }
        }
        return execution_devices;
    } else if (name == ov::model_name) {
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        {
            if (m_scheduler->m_loadcontext[CPU].m_is_enabled && m_scheduler->m_loadcontext[CPU].m_is_already)
                return m_scheduler->m_loadcontext[CPU].m_exe_network->get_property(name);
            return m_scheduler->m_loadcontext[ACTUALDEVICE].m_exe_network->get_property(name);
        }
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        auto metrics = default_ro_properties();
        add_ro_properties(METRIC_KEY(SUPPORTED_METRICS), metrics);
        add_ro_properties(METRIC_KEY(SUPPORTED_CONFIG_KEYS), metrics);
        return to_string_vector(metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        auto configs = default_rw_properties();
        return to_string_vector(configs);
    } else if (name == ov::loaded_from_cache) {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        if (m_scheduler->m_loadcontext[FALLBACKDEVICE].m_is_already) {
                return m_scheduler->m_loadcontext[FALLBACKDEVICE].m_exe_network->get_property(name).as<bool>();
            }
        if (m_scheduler->m_loadcontext[ACTUALDEVICE].m_is_already) {
            return m_scheduler->m_loadcontext[ACTUALDEVICE].
                m_exe_network->get_property(name).as<bool>();
        } else {
            OPENVINO_ASSERT(m_scheduler->m_loadcontext[CPU].m_is_already == true);
             std::lock_guard<std::mutex> lock(m_context->m_mutex);
            return m_scheduler->m_loadcontext[CPU].
                m_exe_network->get_property(name).as<bool>();
        }
    }
    OPENVINO_THROW(get_log_tag(), ": not supported property ", name);
}

void AutoCompiledModel::export_model(std::ostream& model_stream) const {
    OPENVINO_NOT_IMPLEMENTED;
}
} // namespace auto_plugin
} // namespace ov
