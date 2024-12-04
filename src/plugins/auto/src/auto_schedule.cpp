// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include "auto_schedule.hpp"

#include "async_infer_request.hpp"
#include "openvino/runtime/compilation_context.hpp"
#include "openvino/util/file_util.hpp"
#include "plugin.hpp"

// ------------------------------AutoSchedule----------------------------
namespace ov {
namespace auto_plugin {
bool AutoSchedule::select_other_device(const std::string& cur_dev_name) {
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        // a recursive function to select other devices
        std::function<bool(std::string)> get_execution_devices;
        get_execution_devices = [&](const std::string& device_name) {
            std::string real_device_name;
            bool is_cpuhelp = false;
            m_compile_context[FALLBACKDEVICE].m_model_precision = m_context->m_model_precision;
            if (device_name == "CPU_HELP") {
                // if infer failed in CPU_HELP, we will remove CPU from m_device_priorities
                // and re-run infer request when m_compile_context[ACTUALDEVICE] is ready
                real_device_name = "CPU";
                is_cpuhelp = true;
                wait_actual_compiled_model_ready();
            } else {
                real_device_name = device_name;
            }
            const auto current_device_iter = deviceChecker().check_and_return_if_device_in_list<DeviceInformation>
                                             (real_device_name, m_context->m_device_priorities);
            if (current_device_iter != m_context->m_device_priorities.end()) {
                if (m_context->m_device_priorities.size() == 1) {
                    LOG_INFO_TAG("No other devices in m_device_priorities");
                    return false;
                }
                m_context->m_device_priorities.erase(current_device_iter);
                if (is_cpuhelp) {
                    return true;
                }
            } else {
                LOG_DEBUG_TAG("Already selected the fallback device");
                return m_compile_context[FALLBACKDEVICE].m_is_reload_success ? true : false;
            }
            m_compile_context[FALLBACKDEVICE].m_meta_devices = m_context->m_device_priorities;
            m_compile_context[FALLBACKDEVICE].m_is_load_success = false;
            m_compile_context[FALLBACKDEVICE].m_worker_name = "";
            m_compile_context[FALLBACKDEVICE].m_is_reload_success = false;
            m_compile_context[FALLBACKDEVICE].m_device_info =
                m_plugin->select_device(m_context->m_device_priorities,
                                        m_compile_context[FALLBACKDEVICE].m_model_precision,
                                        m_context->m_model_priority);
            try {
                m_compile_context[FALLBACKDEVICE].m_task();
                // FALLBACKDEVICE need to be load again if infer failed, so reset promise here
                m_compile_context[FALLBACKDEVICE].m_promise = {};
                m_compile_context[FALLBACKDEVICE].m_future = m_compile_context[FALLBACKDEVICE].m_promise.get_future();
            } catch (const ov::Exception& iie) {
                LOG_DEBUG_TAG("Load context in FALLBACKDEVICE with error: %s", iie.what());
            }
            if (m_compile_context[FALLBACKDEVICE].m_is_reload_success) {
                m_compile_context[ACTUALDEVICE].m_is_enabled = false;
                m_compile_context[ACTUALDEVICE].m_is_load_success = false;
                m_compile_context[ACTUALDEVICE].m_is_already = false;
                LOG_INFO_TAG("Select fallback device:%s", m_compile_context[FALLBACKDEVICE].m_device_info.device_name.c_str());
                return true;
            } else {
                // load failed or generate works failed, so reselect other devices
                return get_execution_devices(m_compile_context[FALLBACKDEVICE].m_device_info.device_name.c_str());
            }
        };

        return get_execution_devices(cur_dev_name);
    }
}

void AutoSchedule::init() {
    if (m_context->m_bind_buffer) {
        LOG_INFO_TAG("bind buffer supported only under cumulative mode, ignoring");
    }
    // initialize cpuHelpReleasetime
    m_cpuhelp_release_time = std::chrono::steady_clock::now();
    std::string profilingTask = "AutoSchedule::AutoSchedule:AutoMode";
    // loadContext[ACTUALDEVICE] is always enabled,
    // when there is CPU and there are more than two devices, loadContext[CPU] is enabled
    m_compile_context[ACTUALDEVICE].m_is_enabled = true;
    if (m_context->m_runtime_fallback) {
        m_compile_context[FALLBACKDEVICE].m_is_enabled = true;
    }
    m_compile_context[ACTUALDEVICE].m_model_precision = m_context->m_model_precision;
    m_compile_context[ACTUALDEVICE].m_meta_devices = m_context->m_device_priorities;
    m_compile_context[ACTUALDEVICE].m_device_info =
        m_plugin->select_device(m_context->m_device_priorities,
                                m_compile_context[ACTUALDEVICE].m_model_precision,
                                m_context->m_model_priority);

    auto load_device_task = [&](AutoCompileContext* context_ptr, const std::shared_ptr<ov::Model>& model) {
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
        context_ptr->m_promise.set_value();
        // the first compile model process finished
        std::call_once(m_firstload_oc, [this]() {
            m_firstload_promise.set_value();
        });
    };
    auto customize_helper_context_from_cache_setting = [this](bool is_actual_cpu,
                                                              AutoCompileContext m_compile_context[],
                                                              ScheduleContext::Ptr& m_context) {
        const auto cpu_iter = deviceChecker().check_and_return_if_device_in_list("CPU", m_context->m_device_priorities);
        if (cpu_iter == m_context->m_device_priorities.end()) {
            m_compile_context[CPU].m_is_enabled = false;
            return;
        }
        m_compile_context[CPU].m_is_enabled = true;
        if (!is_actual_cpu) {
            const auto& device = m_compile_context[ACTUALDEVICE].m_device_info.device_name;
            auto& device_config = m_compile_context[ACTUALDEVICE].m_device_info.config;
            std::string cache_dir = device_config.count(ov::cache_dir.name())
                                        ? device_config[ov::cache_dir.name()].as<std::string>()
                                        : m_context->m_ov_core->get_property("", ov::cache_dir);

            if (m_context->m_startup_fallback && !cache_dir.empty()) {
                const auto properties =
                    m_context->m_ov_core->create_compile_config(ov::DeviceIDParser(device).get_device_name(),
                                                                device_config);
                std::string blobId;
                if (m_context->m_model)
                    blobId = ov::ModelCache::compute_hash(std::const_pointer_cast<const ov::Model>(m_context->m_model),
                                                          properties);
                else
                    blobId = ov::ModelCache::compute_hash(m_context->m_model_path, properties);
                std::string cached_model_path = ov::util::make_path(cache_dir, blobId + ".blob");
                m_compile_context[CPU].m_is_enabled = !ov::util::file_exists(cached_model_path);
                LOG_DEBUG_TAG("device: %s %s cached blob: %s ",
                              device.c_str(),
                              m_compile_context[CPU].m_is_enabled ? "not found" : "found",
                              cached_model_path.c_str());
            }
        }
        if (m_compile_context[CPU].m_is_enabled) {
            m_compile_context[CPU].m_device_info = *cpu_iter;
            m_compile_context[CPU].m_device_info.config[ov::hint::performance_mode.name()] =
                ov::hint::PerformanceMode::LATENCY;
            if (m_compile_context[ACTUALDEVICE].m_device_info.config.count(ov::cache_dir.name()) &&
                (m_context->m_startup_fallback || m_context->m_runtime_fallback)) {
                m_compile_context[CPU].m_device_info.config[ov::cache_dir.name()] = "";
                LOG_INFO_TAG("Clear cache dir setting for CPU accelerator");
            }
            m_compile_context[CPU].m_worker_name = "CPU_HELP";
            LOG_INFO_TAG("will load CPU for accelerator");
        }
    };
    if (m_compile_context[ACTUALDEVICE].m_is_enabled) {
        LOG_INFO_TAG("select device:%s", m_compile_context[ACTUALDEVICE].m_device_info.device_name.c_str());
        bool is_actual_cpu = m_compile_context[ACTUALDEVICE].m_device_info.device_name.find("CPU") != std::string::npos;
        // if Actual device is CPU or perf_hint is cumulative, disabled m_compile_context[CPU], only use
        // m_compile_context[ACTUALDEVICE]
        if (is_actual_cpu || !m_context->m_startup_fallback) {
            m_compile_context[CPU].m_is_enabled = false;
        } else {
            customize_helper_context_from_cache_setting(is_actual_cpu, m_compile_context, m_context);
        }
        // initialize the rest members of load context
        for (int i = 0; i < CONTEXTNUM; i++) {
            if (m_compile_context[i].m_is_enabled) {
                m_compile_context[i].m_future = m_compile_context[i].m_promise.get_future();
                auto* context_ptr = &m_compile_context[i];
                auto model = m_context->m_model;
                m_compile_context[i].m_task = std::bind(load_device_task, context_ptr, model);
            }
        }
    }
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, openvino::itt::handle(profilingTask));
    if (m_compile_context[CPU].m_is_enabled) {
        m_firstload_future = m_firstload_promise.get_future();
        // will not wait for compiling accelerator model,
        // so the executor can't be destroyed before finished the task,
        // so use executor as a member of AutoSchedule.
        m_executor =
            m_plugin->get_executor_manager()->get_idle_cpu_streams_executor(ov::threading::IStreamsExecutor::Config{
                "AutoDeviceAsyncCompile",
                static_cast<int>(std::thread::hardware_concurrency()) /* max possible #streams*/,
                0 /*default threads per stream, workaround for ticket 62376*/});
        for (auto&& device : m_context->m_device_priorities) {
            // initialize containers before run async task
            m_idle_worker_requests[device.device_name];
            m_worker_requests[device.device_name];
            m_infer_pipeline_tasks_device_specific[device.device_name] = nullptr;
        }
        m_idle_worker_requests["CPU_HELP"];
        m_worker_requests["CPU_HELP"];
        m_infer_pipeline_tasks_device_specific["CPU_HELP"] = nullptr;
        m_executor->run(m_compile_context[CPU].m_task);
        m_executor->run(m_compile_context[ACTUALDEVICE].m_task);
        auto recycleTask = [this]() mutable {
            wait_actual_compiled_model_ready();
            while (!m_exitflag && m_compile_context[ACTUALDEVICE].m_is_already) {
                // handle the case of ACTUAL faster than CPU
                m_compile_context[CPU].m_future.wait();
                // clean up helper infer requests
                // first, wait for all the remaining requests to finish
                for (auto& iter : m_worker_requests["CPU_HELP"]) {
                    try {
                        iter.m_inferrequest._ptr->wait();
                    } catch (const ov::Exception& iie) {
                        LOG_DEBUG_TAG("No infer results expected, infer in CPU_HELP throw some errors: %s", iie.what());
                    }
                }
                // late enough to check the idle queue now
                // second, check the idle queue if all requests are in place
                size_t destroynum = 0;
                std::pair<int, WorkerInferRequest*> worker;
                std::list<Time> cpuhelp_all_start_times;
                std::list<Time> cpuhelp_all_end_times;
                auto first_infer_time = std::chrono::duration<double, std::milli>(0.0);
                while (m_idle_worker_requests["CPU_HELP"].try_pop(worker)) {
                    destroynum++;
                    INFO_RUN([&cpuhelp_all_start_times, &cpuhelp_all_end_times, &worker]() {
                        cpuhelp_all_start_times.splice(cpuhelp_all_start_times.end(), worker.second->m_start_times);
                        cpuhelp_all_end_times.splice(cpuhelp_all_end_times.end(), worker.second->m_end_times);
                    });
                }
                INFO_RUN([this, &first_infer_time, &cpuhelp_all_start_times, &cpuhelp_all_end_times]() {
                    m_cpuhelp_infer_count = cpuhelp_all_start_times.size();
                    OPENVINO_ASSERT(m_cpuhelp_infer_count == cpuhelp_all_end_times.size());
                    if (m_cpuhelp_infer_count != 0) {
                        first_infer_time = cpuhelp_all_end_times.front() - cpuhelp_all_start_times.front();
                    }
                    cpuhelp_all_start_times.sort(std::less<Time>());
                    cpuhelp_all_end_times.sort(std::less<Time>());
                });
                if (destroynum == m_worker_requests["CPU_HELP"].size()) {
                    std::lock_guard<std::mutex> lock(m_context->m_mutex);
                    INFO_RUN([this, first_infer_time, &cpuhelp_all_start_times, &cpuhelp_all_end_times, &destroynum]() {
                        m_cpuhelp_release_time = std::chrono::steady_clock::now();
                        if (cpuhelp_all_start_times.size() >= destroynum + 1) {
                            // remove last worksize num requests, so the fps will be more accuracy
                            cpuhelp_all_start_times.resize(m_cpuhelp_infer_count - destroynum);
                            cpuhelp_all_end_times.resize(m_cpuhelp_infer_count - destroynum);
                            auto duration = m_cpuhelp_infer_count != 0
                                                ? std::chrono::duration<double, std::milli>(0.0)
                                                : cpuhelp_all_end_times.back() - cpuhelp_all_start_times.front();
                            m_cpuhelp_fps = cpuhelp_all_start_times.size() * 1000 / duration.count();
                            LOG_INFO_TAG("CPU_HELP: first inference time:%lf ms", first_infer_time.count());
                            LOG_INFO_TAG("CPU_HELP:infer:%ld", m_cpuhelp_infer_count);
                            LOG_INFO_TAG("CPU_HELP:fps:%lf", m_cpuhelp_fps);
                        }
                    });
                    LOG_INFO_TAG("release all work requests of CPU_HELP");
                    m_worker_requests["CPU_HELP"].clear();
                    m_compile_context[CPU].m_compiled_model._ptr.reset();
                    m_compile_context[CPU].m_compiled_model._so.reset();
                    m_compile_context[CPU].m_is_already = false;
                    LOG_INFO_TAG("helper released!!");
                    break;
                }
            }
        };
        m_executor->run(std::move(recycleTask));
    } else if (m_context->m_device_priorities.size() != 1 && m_context->m_runtime_fallback) {
        // The performance will has some drop then m_passthrough_compiled_model when enable ENABLE_RUNTIME_FALLBACK
        for (auto&& device : m_context->m_device_priorities) {
            // initialize containers before run async task
            m_idle_worker_requests[device.device_name];
            m_worker_requests[device.device_name];
            m_infer_pipeline_tasks_device_specific[device.device_name] = nullptr;
        }
        m_compile_context[ACTUALDEVICE].m_task();
    } else {
        // only one device need to compile model, do not need to compile it async
        m_compile_context[ACTUALDEVICE].m_task();
        m_passthrough_compiled_model = m_compile_context[ACTUALDEVICE].m_compiled_model;
    }
    m_context->m_hw_compiled_model = wait_first_compiled_model_ready();
}

void AutoSchedule::try_to_compile_model(AutoCompileContext& context, const std::shared_ptr<ov::Model>& model) {
    auto& device = context.m_device_info.device_name;
    auto& device_config = context.m_device_info.config;
    auto& device_list = context.m_meta_devices;
    bool cur_dev_is_cpu = (device.find("CPU") != std::string::npos);
    bool cur_dev_is_gpu = (device.find("GPU") != std::string::npos);
    {
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        // user does not set the compiling threads
        // limit the threads num for compiling
        bool is_already_set_gpu =
            (device_config.find(ov::intel_gpu::hint::host_task_priority.name()) != device_config.end() ||
             device_config.find(ov::compilation_num_threads.name()) != device_config.end());
        if (cur_dev_is_gpu && m_compile_context[CPU].m_is_enabled && !is_already_set_gpu) {
            device_config.insert(ov::intel_gpu::hint::host_task_priority(ov::hint::Priority::HIGH));
            int max_threads = 0;
            try {
                m_context->m_ov_core->get_property(device, ov::compilation_num_threads);
                auto proc_type_table = get_org_proc_type_table();
                max_threads = proc_type_table[0][MAIN_CORE_PROC] != 0 ? proc_type_table[0][MAIN_CORE_PROC]
                                                                      : proc_type_table[0][EFFICIENT_CORE_PROC];
                if (device_config.insert(ov::compilation_num_threads(max_threads)).second)
                    LOG_DEBUG_TAG("gpu streams number for compiling: %d", max_threads);
                else
                    LOG_DEBUG_TAG("user defined compiling threads: %d",
                                  device_config[ov::compilation_num_threads.name()].as<int32_t>());
            } catch (const ov::Exception&) {
                LOG_DEBUG_TAG("cannot get MAX_NUM_THREADS from GPU");
            }
        }
    }
    try {
        auto compile_start_time = std::chrono::high_resolution_clock::now();
        if (!(m_context->m_model_path.empty())) {
            context.m_compiled_model =
                m_context->m_ov_core->compile_model(m_context->m_model_path, device, device_config);
        } else {
            context.m_compiled_model = m_context->m_ov_core->compile_model(model, device, device_config);
        }
        context.m_is_load_success = true;
        auto compile_end_time = std::chrono::high_resolution_clock::now();
        auto compiled_time =
            std::chrono::duration_cast<std::chrono::nanoseconds>(compile_end_time - compile_start_time).count() *
            0.000001;
        LOG_INFO_TAG("Device: [%s]: Compile model took %lf ms", device.c_str(), compiled_time);
    } catch (const ov::Exception& e) {
        context.m_err_message += device + ":" + e.what();
        LOG_WARNING_TAG("Device: [%s]: Compile model failure: %s", device.c_str(), e.what());
        context.m_is_load_success = false;
    } catch (const std::exception& e) {
        context.m_err_message += device + ":" + e.what();
        LOG_WARNING_TAG("Device: [%s]: Compile model failure: %s", device.c_str(), e.what());
        context.m_is_load_success = false;
    }
    if (context.m_is_load_success || cur_dev_is_cpu) {
        return;
    }
    // need to recompile model, unregister it's priority
    // there maybe potential issue.
    // for example they are dGPU, NPU, iGPU, customer want to compile model with
    // configure 0 dGPU, 1 NPU, if dGPU compile failed,
    // the result will be not sure, maybe two models are compiled into NPU,
    // maybe 0 is compiled to NPU, 1 is compiled to iGPU
    m_plugin->unregister_priority(m_context->m_model_priority, context.m_device_info.unique_name);
    // remove the current device from device_list
    auto erase_device = deviceChecker().check_and_return_if_device_in_list(device, device_list, true);
    if (erase_device != device_list.end())
        device_list.erase(erase_device);
    if (device_list.empty()) {
        return;
    }
    // select next candidate device
    try {
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        context.m_device_info = m_plugin->select_device(device_list,
                context.m_model_precision, m_context->m_model_priority);
    } catch (const std::exception&) {
        return;
    }
    // if the select device is CPU, need to check the config of m_compile_context[CPU]
    // if they are same, do not need to compile again
    cur_dev_is_cpu = (context.m_device_info.device_name.find("CPU") != std::string::npos);
    if (cur_dev_is_cpu) {
        auto compare = [](ov::AnyMap& a, ov::AnyMap& b) -> bool {
            if (a.size() != b.size()) {
                return false;
            }
            for (auto& item : a) {
                auto bIter = b.find(item.first);
                if (bIter != b.end()) {
                    if (bIter->second != item.second) {
                        return false;
                    }
                } else {
                    return false;
                }
            }
            return true;
        };
        if (compare(context.m_device_info.config, m_compile_context[CPU].m_device_info.config)) {
            return;
        }
    }
    LOG_DEBUG_TAG("try to compile %s", context.m_device_info.device_name.c_str());
    // try to compile this candidate device
    try_to_compile_model(context, model);
}

SoCompiledModel AutoSchedule::wait_first_compiled_model_ready() {
    if (m_firstload_future.valid()) {
        // wait for the first compiling finished
        m_firstload_future.wait();
    }
    // check if there is any device that have compiled model successfully
    for (int i = CONTEXTNUM - 2; i >= 0; i--) {
        if (m_compile_context[i].m_is_enabled && m_compile_context[i].m_is_already) {
            return m_compile_context[i].m_compiled_model;
        }
    }
    // the first compiling is failed, wait for another compiling
    for (int i = CONTEXTNUM - 2; i >= 0; i--) {
        if (m_compile_context[i].m_is_enabled) {
            m_compile_context[i].m_future.wait();
            // check if compiling is successful
            if (m_compile_context[i].m_is_already) {
                return m_compile_context[i].m_compiled_model;
            }
        }
    }
    std::ostringstream result;
    //print m_err_message
    result << "compile model failed, ";
    for (int i = CONTEXTNUM - 2; i >= 0; i--) {
        if (m_compile_context[i].m_is_enabled) {
            result << m_compile_context[i].m_err_message.c_str();
            result << "; ";
            LOG_ERROR_TAG("load failed, %s", m_compile_context[i].m_err_message.c_str());
        }
    }
    OPENVINO_THROW("[", get_log_tag(), "] ", result.str());
}

void AutoSchedule::wait_actual_compiled_model_ready() const {
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, "AutoSchedule::wait_actual_compiled_model_ready");
    // Maybe different API will call this function, so add call once here
    // for every AutoSchedule instance
    std::call_once(m_oc, [this]() {
        if (m_compile_context[ACTUALDEVICE].m_future.valid()) {
            m_compile_context[ACTUALDEVICE].m_future.wait();
        }
    });
}

bool AutoSchedule::schedule_to_worker_infer_request(ov::threading::Task pipeline_task, DeviceName preferred_device) {
    std::vector<DeviceInformation> devices;
    // AUTO work mode
    // Devices that fail infer will be removed from the priority list in the callback, need lock here
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        if (!preferred_device.empty()) {
            // if the device needed by customer is not ready, need to wait for it
            wait_actual_compiled_model_ready();
            devices.push_back(m_compile_context[ACTUALDEVICE].m_device_info);
            if (!deviceChecker().check_if_device_in_list<DeviceInformation>(preferred_device, devices)) {
                OPENVINO_THROW("The preferred device should be the selected device");
            }
        } else {
            // _acceleratorDevice could be the same as _cpuDevice, such as AUTO:CPU
            if (m_compile_context[FALLBACKDEVICE].m_is_already) {
                devices.push_back(m_compile_context[FALLBACKDEVICE].m_device_info);
            } else {
                if (m_compile_context[ACTUALDEVICE].m_is_already) {
                    devices.push_back(m_compile_context[ACTUALDEVICE].m_device_info);
                } else {
                    // replace deviceName with m_worker_name, so schedule can select correct
                    // idleWorkerQueue
                    auto m_device_info = m_compile_context[CPU].m_device_info;
                    m_device_info.device_name = m_compile_context[CPU].m_worker_name;
                    devices.push_back(std::move(m_device_info));
                }
            }
        }
    }
    for (auto&& device : devices) {
        if (!preferred_device.empty() && (device.device_name != preferred_device)) {
            continue;
        }
        if (run_pipeline_task(pipeline_task, m_idle_worker_requests[device.device_name], preferred_device)) {
            return true;
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

AutoSchedule::~AutoSchedule() {
    // this is necessary to guarantee member destroyed after getting future
    if (m_compile_context[CPU].m_is_enabled) {
        m_exitflag = true;
        m_compile_context[CPU].m_future.wait();
        wait_actual_compiled_model_ready();
        // it's necessary to wait the compile model threads to stop here.
        m_plugin->get_executor_manager()->clear("AutoDeviceAsyncCompile");
        m_executor.reset();
    }
    if (m_plugin)
        m_plugin->unregister_priority(m_context->m_model_priority,
                                      m_compile_context[ACTUALDEVICE].m_device_info.unique_name);
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
