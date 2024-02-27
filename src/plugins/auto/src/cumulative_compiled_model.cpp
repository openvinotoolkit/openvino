// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cumulative_compiled_model.hpp"
#include "common.hpp"
#include <memory>

#include "async_infer_request.hpp"
#include "itt.hpp"
#include "openvino/runtime/exec_model_info.hpp"
#include "openvino/runtime/properties.hpp"
#include "plugin.hpp"

namespace ov {
namespace auto_plugin {
AutoCumuCompiledModel::AutoCumuCompiledModel(const std::shared_ptr<ov::Model>& model,
                                             const std::shared_ptr<const ov::IPlugin>& plugin,
                                             const ov::SoPtr<ov::IRemoteContext>& remote_context,
                                             ScheduleContext::Ptr& schedule_context,
                                             Schedule::Ptr& scheduler)
    : CompiledModel(model, plugin, remote_context, schedule_context, scheduler) {
      m_scheduler = std::dynamic_pointer_cast<CumuSchedule>(scheduler);
}

void AutoCumuCompiledModel::set_property(const ov::AnyMap& properties) {
    OPENVINO_NOT_IMPLEMENTED;
}

std::shared_ptr<const ov::Model> AutoCumuCompiledModel::get_runtime_model() const {
    if (m_context->m_hw_compiled_model) {
        auto model = m_context->m_hw_compiled_model->get_runtime_model();
        set_model_shared_object(const_cast<ov::Model&>(*model), m_context->m_hw_compiled_model._so);
        return model;
    }
    OPENVINO_NOT_IMPLEMENTED;
}

ov::Any AutoCumuCompiledModel::get_property(const std::string& name) const {
    const auto& default_ro_properties = []() {
        std::vector<ov::PropertyName> ro_properties{ov::model_name,
                                                    ov::supported_properties,
                                                    ov::execution_devices,
                                                    ov::hint::performance_mode,
                                                    ov::optimal_number_of_infer_requests,
                                                    ov::device::properties,
                                                    ov::hint::model_priority,
                                                    ov::loaded_from_cache,
                                                    ov::intel_auto::schedule_policy};
        return ro_properties;
    };
    const auto& default_rw_properties = []() {
        std::vector<ov::PropertyName> rw_properties{ov::device::priorities};
        return rw_properties;
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
    } else if (name == ov::intel_auto::schedule_policy) {
        return m_context->m_schedule_policy;
    } else if (name == ov::device::priorities) {
        // device priority does not support change on-the-fly
        return decltype(ov::device::priorities)::value_type(m_context->m_str_devices);
    } else if (name == ov::device::properties) {
        ov::AnyMap all_devices = {};
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                auto temp = get_device_supported_properties(m_scheduler->m_p_ctput_loadcontext[i]);
                all_devices.insert(temp.begin(), temp.end());
            }
        }
        return all_devices;
    } else if (name == ov::hint::model_priority) {
        auto value = m_context->m_model_priority;
        return value ? ((value > 1) ? ov::hint::Priority::LOW : ov::hint::Priority::MEDIUM) : ov::hint::Priority::HIGH;
    } else if (name == ov::optimal_number_of_infer_requests) {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        unsigned int res = 0u;
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            try {
                if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                    res += (m_scheduler->m_p_ctput_loadcontext[i])
                                .m_compiled_model->get_property(ov::optimal_number_of_infer_requests.name())
                                .as<unsigned int>();
                }
            } catch (const ov::Exception& err) {
                OPENVINO_THROW("Every device used in cumulative mode should support OPTIMAL_NUMBER_OF_INFER_REQUESTS property from compiled model",
                        "Failed to query the property with error:", err.what());
            }
        }
        return decltype(ov::optimal_number_of_infer_requests)::value_type {res};
    } else if (name == ov::execution_devices) {
        std::vector<std::string> exeDevices = {};
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        for (auto const & n : m_context->m_device_priorities) {
            exeDevices.push_back(n.device_name);
        }
        return decltype(ov::execution_devices)::value_type {exeDevices};
    } else if (name == ov::model_name) {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                return m_scheduler->m_p_ctput_loadcontext[i].m_compiled_model->get_property(name);
            }
        }
        OPENVINO_THROW("No valid compiled model found to get", name);
    } else if (name == ov::loaded_from_cache) {
        bool loaded_from_cache = true;
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        for (size_t i = 0; i < m_scheduler->m_n_ctput_devicenums; i++) {
            if (m_scheduler->m_p_ctput_loadcontext[i].m_is_already) {
                try {
                    loaded_from_cache &=
                        (m_scheduler->m_p_ctput_loadcontext[i].m_compiled_model->get_property(name).as<bool>());
                } catch (const ov::Exception&) {
                    LOG_DEBUG_TAG("get_property loaded_from_cache from %s failed",
                                  m_scheduler->m_p_ctput_loadcontext[i].m_device_info.device_name.c_str());
                    return false;
                }
            }
        }
        return loaded_from_cache;
    }
    OPENVINO_THROW(get_log_tag(), ": not supported property ", name);;
}

void AutoCumuCompiledModel::export_model(std::ostream& model_stream) const {
    OPENVINO_NOT_IMPLEMENTED;
}
} // namespace auto_plugin
} // namespace ov
