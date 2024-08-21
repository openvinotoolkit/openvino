// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "cumulative_schedule.hpp"
#include "async_infer_request.hpp"
#include "plugin.hpp"

// ------------------------------CumuSchedule----------------------------
namespace ov {
namespace auto_plugin {
std::string CumuSchedule::schedule_to_next_device(const std::vector<DeviceInformation>& devices,
                                                  std::size_t current_device_index) {
    std::string selected_device_name = "";
    {
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        m_n_ctput_schedule_next_device =
            m_n_ctput_schedule_next_device >= devices.size() ? 0 : m_n_ctput_schedule_next_device;
        selected_device_name = devices[m_n_ctput_schedule_next_device].device_name;
    }
    const auto& schedule_policy = m_context->m_schedule_policy;
    if (schedule_policy == ov::intel_auto::SchedulePolicy::ROUND_ROBIN) {
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        m_n_ctput_schedule_next_device++;
    } else if (schedule_policy == ov::intel_auto::SchedulePolicy::DEVICE_PRIORITY) {
        selected_device_name = devices[current_device_index].device_name;
    }
    return selected_device_name;
}

bool CumuSchedule::select_other_device(const std::string& cur_dev_name) {
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);

        auto remove_inferfail_device = [&](const std::string& device_name) {
            if (m_context->m_device_priorities.size() > 1) {
                const auto current_device_iter =
                    deviceChecker().check_and_return_if_device_in_list<DeviceInformation>(device_name, m_context->m_device_priorities);
                if (current_device_iter != m_context->m_device_priorities.end()) {
                    m_context->m_device_priorities.erase(current_device_iter);
                    return true;
                }
            }
            return false;
        };

        if (m_p_ctput_loadcontext) {
            return remove_inferfail_device(cur_dev_name);
        }
        return false;
    }
}

void CumuSchedule::init() {
    if (m_context->m_bind_buffer) {
        // disable run time fallback , as not applicable in bind mode
        m_context->m_runtime_fallback = false;
        LOG_INFO_TAG("disable runtime fallback in bind mode");
    }
    std::string profilingTask = "CumuSchedule::CumuSchedule:compile_model";
    const auto& valid_devices = m_context->m_device_priorities;
    {
        // Total number of devices in CTPUT
        m_n_ctput_devicenums = valid_devices.size();
        // Generate contexts for loading each device
        m_p_ctput_loadcontext.reset(new AutoCompileContext[m_n_ctput_devicenums]);
        int idx = 0;
        DeviceInformation cpu_device_information;
        for (auto& device : valid_devices) {
            if (device.device_name.find("CPU") == std::string::npos) {
                m_p_ctput_loadcontext[idx].m_device_info = device;
                idx++;
            } else {
                cpu_device_information = device;
                OPENVINO_SUPPRESS_DEPRECATED_START
                cpu_device_information.config.insert(
                    {ov::affinity.name(), ov::Any(ov::Affinity::CORE).as<std::string>()});
                OPENVINO_SUPPRESS_DEPRECATED_END
            }
        }
        if (!cpu_device_information.device_name.empty())
            m_p_ctput_loadcontext[idx].m_device_info = cpu_device_information;
    }
    if (m_context->m_log_tag == "MULTI") {
        // MULTI's performance hint always is tput
        m_context->m_performance_hint = ov::hint::PerformanceMode::THROUGHPUT;
    }

    auto load_device_task = [&](AutoCompileContext* context_ptr,
                                const std::shared_ptr<ov::Model>& model) {
        try_to_compile_model(*context_ptr, model);
        if (context_ptr->m_is_load_success) {
            if (context_ptr->m_worker_name.empty()) {
                context_ptr->m_worker_name = context_ptr->m_device_info.device_name;
            }
            generate_workers(context_ptr->m_worker_name, context_ptr->m_compiled_model);
            context_ptr->m_is_already = true;
            // reloadsuccess flag only for m_compile_context[FALLBACKDEVICE]
            context_ptr->m_is_reload_success = true;
            auto& device_name = context_ptr->m_device_info.device_name;
            LOG_INFO_TAG("device:%s compiling model finished", device_name.c_str());
            DEBUG_RUN([this, &context_ptr, &device_name] {
                auto supported_config_keys = context_ptr->m_compiled_model->get_property(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
                std::lock_guard<std::mutex> lock(m_context->m_mutex);
                for (const auto& cfg : supported_config_keys) {
                    try {
                        LOG_DEBUG_TAG("device:%s, GetConfig:%s=%s",
                                      device_name.c_str(),
                                      cfg.c_str(),
                                      context_ptr->m_compiled_model->get_property(cfg).as<std::string>().c_str());
                    } catch (const ov::Exception&) {
                    }
                }
            });
        }
        // Handle device load failure in case of ctput
        if (!context_ptr->m_is_load_success) {
            std::string failedDeviceName = context_ptr->m_device_info.device_name;
            std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
            const auto DeviceIter = deviceChecker().check_and_return_if_device_in_list(failedDeviceName, m_context->m_device_priorities);
            // Remove failed device from m_device_priorities
            if (DeviceIter != m_context->m_device_priorities.end()) {
                m_context->m_device_priorities.erase(DeviceIter);
            }
        }
    };
    m_executor =
        m_plugin->get_executor_manager()->get_idle_cpu_streams_executor(ov::threading::IStreamsExecutor::Config{
            "CTPUTDeviceAsyncLoad",
            static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
            0 /*default threads per stream, workaround for ticket 62376*/});
    std::vector<ov::threading::Task> other_devices_loads;
    std::vector<ov::threading::Task> cpu_loads;
    for (size_t i = 0; i < m_n_ctput_devicenums; i++) {
        auto* context_ptr = &m_p_ctput_loadcontext[i];
        auto model = m_context->m_model;
        m_p_ctput_loadcontext[i].m_task = std::bind(load_device_task, context_ptr, model);
        if (i == m_n_ctput_devicenums - 1 &&
            m_p_ctput_loadcontext[i].m_device_info.device_name.find("CPU") != std::string::npos) {
            cpu_loads.push_back(m_p_ctput_loadcontext[i].m_task);
        } else {
            other_devices_loads.push_back(m_p_ctput_loadcontext[i].m_task);
        }
    }
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, openvino::itt::handle(profilingTask));
    for (auto&& device : m_context->m_device_priorities) {
        // initialize containers before run async task, if not initialized, it will hang during infer
        m_idle_worker_requests[device.device_name];
        m_worker_requests[device.device_name];
        m_infer_pipeline_tasks_device_specific[device.device_name] = nullptr;
    }
    // load devices other than CPU first
    if (other_devices_loads.size() > 0) {
        // Wait for the devices other than CPU to compile the model
        m_executor->run_and_wait(other_devices_loads);
    }
    // Finally load the CPU
    if (cpu_loads.size() > 0) {
        // Wait for CPU to compile the model
        m_executor->run_and_wait(cpu_loads);
    }
    if (m_n_ctput_devicenums == 1 && m_p_ctput_loadcontext[0].m_is_already) {
        m_passthrough_compiled_model = m_p_ctput_loadcontext[0].m_compiled_model;
        m_context->m_hw_compiled_model = m_passthrough_compiled_model;
    }
    m_context->m_hw_compiled_model = wait_first_compiled_model_ready();
}

void CumuSchedule::try_to_compile_model(AutoCompileContext& context, const std::shared_ptr<ov::Model>& model) {
    auto& device = context.m_device_info.device_name;
    auto& device_config = context.m_device_info.config;
    bool cur_dev_is_gpu = (device.find("GPU") != std::string::npos);
    {
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        if (cur_dev_is_gpu) {
            // user does not set the compiling threads
            // limit the threads num for compiling
            int max_threads = 0;
            try {
                max_threads = m_context->m_ov_core->get_property(device, ov::compilation_num_threads);
            } catch (const ov::Exception&) {
                LOG_DEBUG_TAG("cannot get MAX_NUM_THREADS from GPU");
            }
            if (max_threads == static_cast<int>(std::thread::hardware_concurrency())) {
                int thread_num = max_threads / 2;
                device_config.insert(ov::compilation_num_threads(thread_num));
                LOG_DEBUG_TAG("gpu streams number for compiling: %d", thread_num);
            } else {
                // user set the compiling threads num
                // use the user's val anyway
                LOG_DEBUG_TAG("user defined compiling threads: %d", max_threads);
            }
        }
    }
    try {
        if (!(m_context->m_model_path.empty())) {
            context.m_compiled_model = m_context->m_ov_core->compile_model(m_context->m_model_path, device, device_config);
        } else {
            context.m_compiled_model = m_context->m_ov_core->compile_model(model, device, device_config);
        }
        context.m_is_load_success = true;
    } catch (const ov::Exception& e) {
        context.m_err_message += device + ":" + e.what();
        context.m_is_load_success = false;
    } catch (const std::exception& e) {
        context.m_err_message += device + ":" + e.what();
        context.m_is_load_success = false;
    }
}
SoCompiledModel CumuSchedule::wait_first_compiled_model_ready() {
    std::ostringstream result;
    result << "compile model failed, ";
    for (size_t i = 0; i < m_n_ctput_devicenums; i++) {
        // check if device loaded successfully
        if (m_p_ctput_loadcontext[i].m_is_already) {
            return m_p_ctput_loadcontext[i].m_compiled_model;
        } else {
            result << m_p_ctput_loadcontext[i].m_err_message.c_str();
            result << "; ";
        }
    }
    OPENVINO_THROW("[", get_log_tag(), "] ", result.str());
}

bool CumuSchedule::schedule_to_worker_infer_request(ov::threading::Task pipeline_task, DeviceName preferred_device) {
    std::vector<DeviceInformation> devices;
    // AUTO work mode
    // Devices that fail infer will be removed from the priority list in the callback, need lock here
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        if (!preferred_device.empty()) {
            devices = m_context->m_device_priorities;
            if (!deviceChecker().check_if_device_in_list<DeviceInformation>(preferred_device, devices)) {
                OPENVINO_THROW("The preferred device should be the selected device");
            }
        } else {
            devices = m_context->m_device_priorities;
        }
    }

    std::size_t current_device_index = 0;
    while (current_device_index < devices.size()) {
        if (!preferred_device.empty() && (devices[current_device_index].device_name != preferred_device)) {
            current_device_index++;
            continue;
        }
        auto selected_device_name =
            preferred_device.empty() ? schedule_to_next_device(devices, current_device_index) : preferred_device;
        if (run_pipeline_task(pipeline_task, m_idle_worker_requests[selected_device_name], preferred_device)) {
            return true;
        } else {
            current_device_index++;
        }
    }

    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty()) {
        m_infer_pipeline_tasks_device_specific[preferred_device]->push(std::move(pipeline_task));
    } else {
        m_infer_pipeline_tasks.push(std::move(pipeline_task));
    }
    return false;
}

CumuSchedule::~CumuSchedule() {
    if (m_context) {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        m_context->m_device_priorities.clear();
    }
    /* NOTE: The only threads that use `MultiSchedule` worker infer requests' threads.
     *       But AsyncInferRequest destructor should wait for all asynchronous tasks by the request
     */
    for (auto&& idleWorker : m_idle_worker_requests) {
        // stop accepting any idle requests back (for re-scheduling)
        idleWorker.second.set_capacity(0);
    }
}
}  // namespace auto_plugin
}  // namespace ov
