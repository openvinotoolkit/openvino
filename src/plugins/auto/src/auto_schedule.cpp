// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#include <algorithm>
#include <chrono>
#include <utility>

#include "auto_schedule.hpp"

#include <stdexcept>

#include "async_infer_request.hpp"
#include "plugin.hpp"

// ------------------------------AutoSchedule----------------------------
namespace ov {
namespace auto_plugin {
bool AutoSchedule::select_other_device(const ModelInstanceKey& failed_instance) {
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        // a recursive function to select other devices
        std::function<bool(const ModelInstanceKey&)> get_execution_devices;
        get_execution_devices = [&](const ModelInstanceKey& failed) {
            const auto& real_device_name = failed.device;
            const bool is_cpuhelp = failed.is_startup_helper();
            m_compile_context[FALLBACKDEVICE].request.m_model_precision = m_context->m_model_precision;
            if (is_cpuhelp) {
                // if infer failed in the startup helper, we will remove CPU from m_device_priorities
                // and re-run infer request when m_compile_context[ACTUALDEVICE] is ready
                wait_actual_compiled_model_ready();
            }
            const bool low_power_first_failure = check_and_disable_if_low_power_device_failed(real_device_name);
            const auto current_device_iter = deviceChecker().check_and_return_if_device_in_list<DeviceInformation>
                                             (real_device_name, m_context->m_device_priorities, true);
            if (current_device_iter != m_context->m_device_priorities.end()) {
                if (m_context->m_device_priorities.size() == 1) {
                    LOG_INFO_TAG("No other devices in m_device_priorities");
                    return false;
                }
                m_context->m_device_priorities.erase(current_device_iter);
                if (is_cpuhelp) {
                    return true;
                }
            } else if (!low_power_first_failure) {
                // an out-of-list low power target is absent from m_device_priorities, but its first failure
                // still needs a fallback
                LOG_DEBUG_TAG("Already selected the fallback device");
                return m_compile_context[FALLBACKDEVICE].is_ready();
            }
            m_compile_context[FALLBACKDEVICE].request.m_meta_devices = m_context->m_device_priorities;
            m_compile_context[FALLBACKDEVICE].request.m_profile = ModelInstanceKey::Profile::Main;
            m_compile_context[FALLBACKDEVICE].m_device_context = nullptr;
            // a previous fallback device is being replaced, drop its reservation
            if (!m_compile_context[FALLBACKDEVICE].request.m_priority_released &&
                !m_compile_context[FALLBACKDEVICE].request.m_device_info.unique_name.empty()) {
                m_plugin->unregister_priority(m_context->m_model_priority,
                                              m_compile_context[FALLBACKDEVICE].request.m_device_info.unique_name);
            }
            m_compile_context[FALLBACKDEVICE].request.m_priority_released = false;
            m_compile_context[FALLBACKDEVICE].request.m_device_info =
                m_plugin->select_device(m_context->m_device_priorities,
                                        m_compile_context[FALLBACKDEVICE].request.m_model_precision,
                                        m_context->m_model_priority,
                                        m_context->m_selection_policy,
                                        (m_context->m_low_power_device_unavailable ? std::nullopt : m_context->m_low_power_device));
            try {
                m_compile_context[FALLBACKDEVICE].m_task();
                // FALLBACKDEVICE need to be load again if infer failed, so reset promise here
                m_compile_context[FALLBACKDEVICE].m_promise = {};
                m_compile_context[FALLBACKDEVICE].m_future = m_compile_context[FALLBACKDEVICE].m_promise.get_future();
            } catch (const ov::Exception& iie) {
                LOG_DEBUG_TAG("Load context in FALLBACKDEVICE with error: %s", iie.what());
            }
            if (m_compile_context[FALLBACKDEVICE].is_ready()) {
                m_compile_context[ACTUALDEVICE].m_is_enabled = false;
                m_compile_context[ACTUALDEVICE].m_device_context = nullptr;
                publish_active_instance();
                LOG_INFO_TAG("Select fallback device:%s", m_compile_context[FALLBACKDEVICE].request.m_device_info.device_name.c_str());
                return true;
            } else {
                // load failed or generate works failed, so reselect other devices
                return get_execution_devices(ModelInstanceKey(m_compile_context[FALLBACKDEVICE].request.m_device_info.device_name));
            }
        };

        return get_execution_devices(failed_instance);
    }
}

bool AutoSchedule::check_and_disable_if_low_power_device_failed(const std::string& failed_device_name) const {
    if (m_context->m_low_power_device && m_context->m_low_power_device->device_name == failed_device_name &&
        !m_context->m_low_power_device_unavailable.exchange(true)) {
        LOG_WARNING_TAG("low_power_device:%s failed, excluding it from future selection",
                        failed_device_name.c_str());
        return true;
    }
    return false;
}

void AutoSchedule::publish_active_instance() {
    m_instances.publish([this]() -> std::optional<ModelInstanceKey> {
        for (const auto role : {FALLBACKDEVICE, ACTUALDEVICE, CPU}) {
            const auto& compile_context = m_compile_context[role];
            const auto* context = compile_context.m_device_context.load();
            if (context && context->is_ready()) {
                return context->key();
            }
        }
        return std::nullopt;
    });
}

void AutoSchedule::init() {
    if (m_context->m_bind_buffer) {
        LOG_INFO_TAG("bind buffer supported only under cumulative mode, ignoring");
    }
    if (m_context->m_dynamic_device_selection) {
        // snapshot of the graph for on-demand compilation
        m_dynamic_model = m_context->m_model->clone();
        // compile_for_all leaves a cache blob for every candidate, so the inactive ones can be released; without it
        // every compiled model is kept
        if (m_context->m_compile_for_all) {
            m_eviction_policy.emplace();
        }
        m_dynamic_executor = m_plugin->get_executor_manager()->get_idle_cpu_streams_executor(
            ov::threading::IStreamsExecutor::Config{"AutoDynamicSchedule", 1, 0});
        LOG_INFO_TAG("[dynamic] device is re-selected for every inference, inference is serialized per compiled model");
    }
    // initialize cpuHelpReleasetime
    m_cpuhelp_release_time = std::chrono::steady_clock::now();
    std::string profilingTask = "AutoSchedule::AutoSchedule:AutoMode";
    // loadContext[ACTUALDEVICE] is always enabled,
    // when there is CPU and there are more than two devices, loadContext[CPU] is enabled
    m_compile_context[ACTUALDEVICE].m_is_enabled = true;
    m_compile_context[ACTUALDEVICE].request.m_model_precision = m_context->m_model_precision;
    m_compile_context[ACTUALDEVICE].request.m_meta_devices = m_context->m_device_priorities;
    m_compile_context[ACTUALDEVICE].request.m_device_info =
        m_plugin->select_device(m_context->m_device_priorities,
                                m_compile_context[ACTUALDEVICE].request.m_model_precision,
                                m_context->m_model_priority,
                                m_context->m_selection_policy,
                                (m_context->m_low_power_device_unavailable ? std::nullopt : m_context->m_low_power_device));
    if (m_context->m_dynamic_device_selection) {
        // nothing was reserved by select_device() in this mode
        m_compile_context[ACTUALDEVICE].request.m_priority_released = true;
    }
    if (m_context->m_compile_for_all) {
        const auto& actual_device = m_compile_context[ACTUALDEVICE].request.m_device_info.device_name;
        const auto& candidates = m_compile_context[ACTUALDEVICE].request.m_meta_devices;
        const bool has_device_to_compile = std::any_of(candidates.begin(), candidates.end(), [&](const auto& device) {
            return device.device_name != actual_device && ov::DeviceIDParser(device.device_name).get_device_name() != "CPU";
        });
        if (!has_device_to_compile) {
            LOG_INFO_TAG("compile_for_all cache pre-compilation is skipped: no other non-CPU candidate device");
        } else {
            // Fallbacks switch devices non-deterministically and would overlap with the background pre-compilation.
            LOG_INFO_TAG("compile_for_all disables startup and runtime fallback");
            m_context->m_startup_fallback = false;
            m_context->m_runtime_fallback = false;
            // Capture before the actual-device task releases the model.
            m_compile_service->enable_precompile(m_plugin->get_cache_manager(), m_context->m_model);
        }
    }
    if (m_context->m_runtime_fallback) {
        m_compile_context[FALLBACKDEVICE].m_is_enabled = true;
    }

    auto load_device_task = [&](RoleSlot* context_ptr, const std::shared_ptr<ov::Model>& model) {
        auto result = try_to_compile_model(context_ptr->request, model);
        context_ptr->m_err_message += result.m_error;
        if (result.m_succeeded) {
            // release cloned model here
            const_cast<std::shared_ptr<ov::Model>&>(model).reset();
            context_ptr->m_device_context = &register_compile_result(context_ptr->request, std::move(result.m_compiled_model));
            if (!m_context->m_dynamic_device_selection) {
                publish_active_instance();
            }
            auto& device_name = context_ptr->request.m_device_info.device_name;
            LOG_INFO_TAG("device:%s compiling model finished", device_name.c_str());
            DEBUG_RUN([this, &context_ptr, &device_name] {
                auto supported_config_keys = context_ptr->compiled_model()->get_property(ov::supported_properties.name()).as<std::vector<ov::PropertyName>>();
                std::lock_guard<std::mutex> lock(m_context->m_mutex);
                for (const auto& cfg : supported_config_keys) {
                    try {
                        LOG_DEBUG_TAG("device:%s, GetConfig:%s=%s",
                                      device_name.c_str(),
                                      cfg.c_str(),
                                      context_ptr->compiled_model()->get_property(cfg).as<std::string>().c_str());
                    } catch (const ov::Exception&) {
                    }
                }
            });
        } else {
        }
        context_ptr->m_promise.set_value();
        // the first compile model process finished
        std::call_once(m_firstload_oc, [this]() {
            m_firstload_promise.set_value();
        });
    };
    auto customize_helper_context_from_cache_setting = [this](bool is_actual_cpu,
                                                              RoleSlot m_compile_context[],
                                                              ScheduleContext::Ptr& m_context) {
        m_compile_context[CPU].m_is_enabled = true;
        const auto cpu_iter = deviceChecker().check_and_return_if_device_in_list("CPU", m_context->m_device_priorities);
        if (cpu_iter == m_context->m_device_priorities.end() || is_actual_cpu) {
            m_compile_context[CPU].m_is_enabled = false;
            return;
        }
        m_compile_context[CPU].request.m_device_info = *cpu_iter;
        m_compile_context[CPU].request.m_device_info.config[ov::hint::performance_mode.name()] =
            ov::hint::PerformanceMode::LATENCY;
        if (CacheManager::disable_caching(*m_context->m_ov_core,
                                          m_compile_context[CPU].request.m_device_info.device_name,
                                          m_compile_context[CPU].request.m_device_info.config)) {
            LOG_INFO_TAG("Clear cache dir setting for CPU accelerator");
        }
        m_compile_context[CPU].request.m_profile = ModelInstanceKey::Profile::StartupHelper;
        LOG_INFO_TAG("will load CPU for accelerator");
    };
    if (m_compile_context[ACTUALDEVICE].m_is_enabled) {
        LOG_INFO_TAG("select device:%s", m_compile_context[ACTUALDEVICE].request.m_device_info.device_name.c_str());
        bool is_actual_cpu = m_compile_context[ACTUALDEVICE].request.m_device_info.device_name.find("CPU") != std::string::npos;
        // if Actual device is CPU or perf_hint is cumulative, disabled m_compile_context[CPU], only use
        // m_compile_context[ACTUALDEVICE]
        if (is_actual_cpu || !m_context->m_startup_fallback) {
            m_compile_context[CPU].m_is_enabled = false;
        } else {
            customize_helper_context_from_cache_setting(is_actual_cpu, m_compile_context, m_context);
        }
        std::shared_ptr<ov::Model> model;
        // initialize the rest members of load context
        for (int i = 0; i < CONTEXTNUM; i++) {
            if (m_compile_context[i].m_is_enabled) {
                m_compile_context[i].m_future = m_compile_context[i].m_promise.get_future();
                auto* context_ptr = &m_compile_context[i];
                // clone this model if multi HW plugins need to load model in a background thread
                model = !model ? m_context->m_model : m_context->m_model->clone();
                m_compile_context[i].m_task = std::bind(load_device_task, context_ptr, model);
            }
        }
    }
    OV_ITT_SCOPED_TASK(itt::domains::AutoPlugin, openvino::itt::handle(profilingTask));
    // the contexts must exist before the devices are compiled concurrently, the map is not modified afterwards
    for (auto&& device : m_context->m_device_priorities) {
        add_device_context(device.device_name);
    }
    // an out-of-list low power device can be compiled concurrently with the startup helper or selected per inference
    if (m_context->m_low_power_device) {
        add_device_context(m_context->m_low_power_device->device_name);
    }
    if (m_compile_context[CPU].m_is_enabled) {
        add_device_context(m_compile_context[CPU].request.instance_key());
    }
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
        m_executor->run(m_compile_context[CPU].m_task);
        m_executor->run(m_compile_context[ACTUALDEVICE].m_task);
        auto recycleTask = [this]() mutable {
            wait_actual_compiled_model_ready();
            auto& cpu_help = get_device_context(m_compile_context[CPU].request.instance_key());
            while (!m_exitflag && m_compile_context[ACTUALDEVICE].is_ready()) {
                // handle the case of ACTUAL faster than CPU
                m_compile_context[CPU].m_future.wait();
                // clean up helper infer requests
                // first, wait for all the remaining requests to finish
                cpu_help.pool().wait_all([this](const ov::Exception& iie) {
                    LOG_DEBUG_TAG("No infer results expected, infer in the startup helper throw some errors: %s",
                                  iie.what());
                });
                // late enough to check the idle queue now
                // second, release the helper once all its requests are in place, nothing to wait for if it never loaded
                const auto destroynum = cpu_help.pool().size();
                // no lock is held, the model queries cope with a helper that is released under them
                if (!cpu_help.is_ready() || cpu_help.try_release()) {
                    auto released_stats = cpu_help.take_released_stats();
                    auto& cpuhelp_all_start_times = released_stats.first;
                    auto& cpuhelp_all_end_times = released_stats.second;
                    auto first_infer_time = std::chrono::duration<double, std::milli>(0.0);
                    INFO_RUN([this, &first_infer_time, &cpuhelp_all_start_times, &cpuhelp_all_end_times]() {
                        m_cpuhelp_infer_count = cpuhelp_all_start_times.size();
                        OPENVINO_ASSERT(m_cpuhelp_infer_count == cpuhelp_all_end_times.size());
                        if (m_cpuhelp_infer_count != 0) {
                            first_infer_time = cpuhelp_all_end_times.front() - cpuhelp_all_start_times.front();
                        }
                        cpuhelp_all_start_times.sort(std::less<Time>());
                        cpuhelp_all_end_times.sort(std::less<Time>());
                    });
                    INFO_RUN([this, first_infer_time, &cpuhelp_all_start_times, &cpuhelp_all_end_times, destroynum]() {
                        m_cpuhelp_release_time = std::chrono::steady_clock::now();
                        if (cpuhelp_all_start_times.size() >= destroynum + 1) {
                            // remove last worksize num requests, so the fps will be more accuracy
                            cpuhelp_all_start_times.resize(m_cpuhelp_infer_count - destroynum);
                            cpuhelp_all_end_times.resize(m_cpuhelp_infer_count - destroynum);
                            auto duration = m_cpuhelp_infer_count != 0
                                                ? std::chrono::duration<double, std::milli>(0.0)
                                                : cpuhelp_all_end_times.back() - cpuhelp_all_start_times.front();
                            m_cpuhelp_fps = cpuhelp_all_start_times.size() * 1000 / duration.count();
                            LOG_INFO_TAG("CPU(startup helper): first inference time:%lf ms", first_infer_time.count());
                            LOG_INFO_TAG("CPU(startup helper):infer:%lu", static_cast<unsigned long>(m_cpuhelp_infer_count));
                            LOG_INFO_TAG("CPU(startup helper):fps:%lf", m_cpuhelp_fps);
                        }
                    });
                    LOG_INFO_TAG("release all work requests of the startup helper");
                    m_compile_context[CPU].m_device_context = nullptr;
                    publish_active_instance();
                    LOG_INFO_TAG("helper released!!");
                    break;
                }
            }
        };
        m_executor->run(std::move(recycleTask));
    } else if (m_context->m_dynamic_device_selection ||
               (m_context->m_runtime_fallback &&
                ((m_context->m_device_priorities.size() != 1 && m_context->m_str_devices_initial.size() != 1) ||
                 // an out-of-list low power device can still fall back onto the candidates at runtime
                 !deviceChecker().check_if_device_in_list<DeviceInformation>(
                     m_compile_context[ACTUALDEVICE].request.m_device_info.device_name,
                     m_context->m_device_priorities,
                     true)))) {
        // The performance will has some drop then the passthrough compiled model when enable ENABLE_RUNTIME_FALLBACK
        m_compile_context[ACTUALDEVICE].m_task();
        if (m_context->m_dynamic_device_selection && m_compile_context[ACTUALDEVICE].is_ready()) {
            // devices that failed to compile at startup were pruned from this copy only
            m_context->m_device_priorities = m_compile_context[ACTUALDEVICE].request.m_meta_devices;
            const auto& initial_device = m_compile_context[ACTUALDEVICE].request.m_device_info.device_name;
            m_gate_current_device = initial_device;
            m_instances.publish(ModelInstanceKey(initial_device));
            LOG_INFO_TAG("[dynamic] initial target device:%s", initial_device.c_str());
        }
    } else {
        // Only one device, or multiple devices of the same type (e.g., all GPU devices, including iGPU and dGPU), can
        // use passthrough model; no need to compile asynchronously
        LOG_INFO_TAG("Only one device or multiple devices of the same type will use passthrough compiled model");
        m_model_only = !m_context->m_bind_buffer;
        m_compile_context[ACTUALDEVICE].m_task();
        m_passthrough_context = m_compile_context[ACTUALDEVICE].m_device_context.load();
        if (m_passthrough_context) {
            m_passthrough_context->keep_alive();
        }
    }
    capture_hw_metadata(wait_first_compiled_model_ready());
    // compile_for_all disables the CPU helper, so the pre-compilation can start as soon as the actual device is ready.
    if (m_compile_service->has_precompile() && m_compile_context[ACTUALDEVICE].is_ready()) {
        // Failed startup devices were already removed from the actual context's candidates.
        m_compile_service->start_precompile(m_compile_context[ACTUALDEVICE].request.m_device_info.device_name,
                                            m_compile_context[ACTUALDEVICE].request.m_meta_devices,
                                            [this](const std::string& device_name) {
                                                const auto* context = find_device_context(device_name);
                                                return context && context->is_ready();
                                            });
    }
}

CompileResult AutoSchedule::try_to_compile_model(CompileRequest& request,
                                                 const std::shared_ptr<ov::Model>& model) {
    CompileResult result;
    auto& device = request.m_device_info.device_name;
    auto& device_config = request.m_device_info.config;
    auto& device_list = request.m_meta_devices;
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
        result.m_compiled_model = m_compile_service->compile(model, device, device_config);
        result.m_succeeded = true;
    } catch (const ov::Exception& e) {
        result.m_error += device + ":" + e.what();
        LOG_WARNING_TAG("Device: [%s]: Compile model failure: %s", device.c_str(), e.what());
        result.m_succeeded = false;
        mark_compile_failure(request, e.what());
    } catch (const std::exception& e) {
        result.m_error += device + ":" + e.what();
        LOG_WARNING_TAG("Device: [%s]: Compile model failure: %s", device.c_str(), e.what());
        result.m_succeeded = false;
        mark_compile_failure(request, e.what());
    }
    const bool low_power_first_failure =
        !result.m_succeeded && check_and_disable_if_low_power_device_failed(device);
    // a failed low power CPU picked at startup must still cascade to the candidates
    const bool retry_actual_low_power_device =
        low_power_first_failure && &request == &m_compile_context[ACTUALDEVICE].request;
    if (result.m_succeeded || (cur_dev_is_cpu && !retry_actual_low_power_device)) {
        return result;
    }
    // need to recompile model, unregister it's priority
    // there maybe potential issue.
    // for example they are dGPU, NPU, iGPU, customer want to compile model with
    // configure 0 dGPU, 1 NPU, if dGPU compile failed,
    // the result will be not sure, maybe two models are compiled into NPU,
    // maybe 0 is compiled to NPU, 1 is compiled to iGPU
    if (!request.m_priority_released) {
        m_plugin->unregister_priority(m_context->m_model_priority, request.m_device_info.unique_name);
        request.m_priority_released = true;
    }
    // remove the current device from device_list
    auto erase_device = deviceChecker().check_and_return_if_device_in_list(device, device_list, true);
    if (erase_device != device_list.end())
        device_list.erase(erase_device);
    if (device_list.empty()) {
        return result;
    }
    // select next candidate device
    try {
        std::lock_guard<std::mutex> lock(m_context->m_mutex);
        request.m_device_info = m_plugin->select_device(device_list,
                                                        request.m_model_precision,
                                                        m_context->m_model_priority,
                                                        m_context->m_selection_policy,
                                                        (m_context->m_low_power_device_unavailable ? std::nullopt : m_context->m_low_power_device));
    } catch (const ov::Exception&) {
        return result;
    }
    // select_device() registered the reselected device, so a failure on it has to unregister it again, unless
    // the dynamic selection reserved nothing
    request.m_priority_released = m_context->m_dynamic_device_selection;
    if (m_context->m_dynamic_device_selection) {
        if (const auto* ready_context = find_device_context(request.m_device_info.device_name);
            ready_context && ready_context->is_ready()) {
            result.m_succeeded = true;
            return result;
        }
    }
    // if the select device is CPU, need to check the config of m_compile_context[CPU]
    // if they are same, do not need to compile again (only when the CPU helper is enabled)
    cur_dev_is_cpu = (request.m_device_info.device_name.find("CPU") != std::string::npos);
    if (cur_dev_is_cpu && m_compile_context[CPU].m_is_enabled) {
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
        if (compare(request.m_device_info.config, m_compile_context[CPU].request.m_device_info.config)) {
            return result;
        }
    }
    LOG_DEBUG_TAG("try to compile %s", request.m_device_info.device_name.c_str());
    // try to compile this candidate device
    auto next = try_to_compile_model(request, model);
    next.m_error = result.m_error + next.m_error;
    return next;
}

SoCompiledModel AutoSchedule::wait_first_compiled_model_ready() {
    if (m_firstload_future.valid()) {
        // wait for the first compiling finished
        m_firstload_future.wait();
    }
    // check if there is any device that have compiled model successfully
    for (int i = CONTEXTNUM - 2; i >= 0; i--) {
        if (m_compile_context[i].m_is_enabled && m_compile_context[i].is_ready()) {
            return m_compile_context[i].compiled_model();
        }
    }
    // the first compiling is failed, wait for another compiling
    for (int i = CONTEXTNUM - 2; i >= 0; i--) {
        if (m_compile_context[i].m_is_enabled) {
            m_compile_context[i].m_future.wait();
            // check if compiling is successful
            if (m_compile_context[i].is_ready()) {
                return m_compile_context[i].compiled_model();
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
    if (m_context->m_dynamic_device_selection) {
        return schedule_dynamic_task(std::move(pipeline_task), preferred_device);
    }
    std::vector<ModelInstanceKey> instances;
    // AUTO work mode
    // Devices that fail infer will be removed from the priority list in the callback, need lock here
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        if (!preferred_device.empty()) {
            // if the device needed by customer is not ready, need to wait for it
            wait_actual_compiled_model_ready();
            if (!deviceChecker().check_if_device_in_list<DeviceInformation>(
                    preferred_device,
                    std::vector<DeviceInformation>{m_compile_context[ACTUALDEVICE].request.m_device_info})) {
                OPENVINO_THROW("The preferred device should be the selected device");
            }
            instances.push_back(m_compile_context[ACTUALDEVICE].request.instance_key());
        } else {
            // _acceleratorDevice could be the same as _cpuDevice, such as AUTO:CPU
            if (m_compile_context[FALLBACKDEVICE].is_ready()) {
                instances.push_back(m_compile_context[FALLBACKDEVICE].request.instance_key());
            } else if (m_compile_context[ACTUALDEVICE].is_ready()) {
                instances.push_back(m_compile_context[ACTUALDEVICE].request.instance_key());
            } else {
                instances.push_back(m_compile_context[CPU].request.instance_key());
            }
        }
    }
    for (auto&& instance : instances) {
        if (!preferred_device.empty() && (instance.device != preferred_device)) {
            continue;
        }
        if (run_pipeline_task(pipeline_task, get_device_context(instance).pool())) {
            return true;
        }
    }
    // no vacant requests this time, storing the task to the respective queue
    if (!preferred_device.empty()) {
        get_device_context(preferred_device).pool().enqueue(std::move(pipeline_task));
    } else {
        m_infer_pipeline_tasks.push(std::move(pipeline_task));
    }
    return false;
}

bool AutoSchedule::schedule_dynamic_task(ov::threading::Task pipeline_task, const DeviceName& preferred_device) {
    std::shared_ptr<ov::threading::IStreamsExecutor> executor;
    {
        std::lock_guard<std::mutex> lock(m_gate_mutex);
        if (m_dynamic_shutdown) {
            // AutoSchedule is being torn down, do not touch m_dynamic_executor anymore
            return false;
        }
        if (m_gate_busy) {
            m_gate_pending_tasks.emplace_back(std::move(pipeline_task), preferred_device);
            LOG_DEBUG_TAG("[dynamic] an inference is still running, request queued, queue size:%lu",
                          static_cast<unsigned long>(m_gate_pending_tasks.size()));
            return false;
        }
        m_gate_busy = true;
        executor = m_dynamic_executor;
        ++m_active_dispatch;
    }
    // dispatching is offloaded so that neither start_async() nor the completion callback of a device is blocked
    // by the device re-selection and by the compilation of the model on a newly selected device
    submit_dynamic_dispatch(executor, std::move(pipeline_task), preferred_device);
    return true;
}

void AutoSchedule::submit_dynamic_dispatch(const std::shared_ptr<ov::threading::IStreamsExecutor>& executor,
                                           ov::threading::Task pipeline_task,
                                           const DeviceName& preferred_device) {
    executor->run([this, task = std::move(pipeline_task), preferred_device]() mutable {
        // runs last in this body, so ~AutoSchedule() waiting on it never frees this under a live dispatch;
        // the task stays owned by the closure so dropping its last reference cannot run ~AutoSchedule() here
        struct DispatchDone {
            AutoSchedule* self;
            ~DispatchDone() {
                self->end_dynamic_dispatch();
            }
        } done{this};
        dispatch_dynamic_task(task, preferred_device);
    });
}

void AutoSchedule::end_dynamic_dispatch() {
    std::lock_guard<std::mutex> lock(m_gate_mutex);
    if (--m_active_dispatch == 0) {
        m_dispatch_done.notify_all();
    }
}

void AutoSchedule::dispatch_dynamic_task(ov::threading::Task& pipeline_task, const DeviceName& preferred_device) {
    try {
        DeviceInformation device;
        if (!preferred_device.empty()) {
            std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
            const auto iter = deviceChecker().check_and_return_if_device_in_list<DeviceInformation>(
                preferred_device, m_context->m_device_priorities, true);
            if (iter != m_context->m_device_priorities.end()) {
                device = *iter;
            } else {
                // a remote tensor of the low power device pins it even though it is not a candidate
                const auto& low_power = m_context->m_low_power_device;
                OPENVINO_ASSERT(low_power && !m_context->m_low_power_device_unavailable &&
                                    low_power->device_name == preferred_device,
                                "The preferred device should be one of the candidate devices or the low power device");
                device = *low_power;
            }
            LOG_DEBUG_TAG("[dynamic] request is pinned to device:%s by a remote tensor, skip device re-selection",
                          device.device_name.c_str());
        } else {
            device = select_dynamic_device();
        }
        const bool is_pinned = !preferred_device.empty();
        const auto requested_device_name = device.device_name;
        DeviceName previous_device;
        {
            std::lock_guard<std::mutex> lock(m_gate_mutex);
            previous_device = m_gate_current_device;
        }
        if (const auto previous_infer = m_instance_stats.last_inference(ModelInstanceKey(previous_device))) {
            LOG_DEBUG_TAG("[dynamic] previous inference on device:%s took %lf ms",
                          previous_device.c_str(),
                          std::chrono::duration<double, std::milli>(*previous_infer).count());
        }
        OPENVINO_ASSERT(ensure_device_ready(device),
                        get_log_tag(),
                        ": failed to compile the model on the selected device ",
                        device.device_name);
        // reject if a pinned request ended up compiled on a different device
        OPENVINO_ASSERT(!is_pinned || device.device_name == requested_device_name,
                        get_log_tag(),
                        ": failed to compile the model on the device pinned by the remote tensor ",
                        requested_device_name);
        const auto& device_name = device.device_name;
        {
            std::lock_guard<std::mutex> lock(m_gate_mutex);
            if (m_gate_current_device == device_name) {
                LOG_DEBUG_TAG("[dynamic] target device:%s is unchanged, reuse its idle worker", device_name.c_str());
            } else {
                LOG_INFO_TAG("[dynamic] target device switched from %s to %s",
                             m_gate_current_device.empty() ? "none" : m_gate_current_device.c_str(),
                             device_name.c_str());
                m_gate_current_device = device_name;
                // published before the previous device can be released by release_inactive_devices()
                m_instances.publish(ModelInstanceKey(device_name));
                m_decision_log.count_switch();
                m_instance_stats.set_active(ModelInstanceKey(device_name));
            }
        }
        m_decision_log.count_inference();
        get_device_context(device_name).mark_used(std::chrono::steady_clock::now());
        const bool dispatched =
            run_pipeline_task(pipeline_task, get_device_context(device_name).pool());
        OPENVINO_ASSERT(dispatched, get_log_tag(), ": no idle infer request available on device ", device_name);
        // a failure to start the request is swallowed by the pipeline and no callback returns the worker, so give
        // it back here and open the gate
        if (m_this_worker_infer_request && m_this_worker_infer_request->m_dispatch_failed) {
            auto* worker = m_this_worker_infer_request;
            worker->m_dispatch_failed = false;
            // the start was recorded but the final stage never runs to record its end
            INFO_RUN([worker]() {
                if (worker->m_start_times.size() > worker->m_end_times.size()) {
                    worker->m_start_times.pop_back();
                }
            });
            get_device_context(device_name).pool().adopt(worker).give_back();
            release_execution_slot();
        }
        // off the critical path: the next dispatch runs on this single stream executor, so it waits for the release
        try {
            release_inactive_devices(device_name);
        } catch (const std::exception& ex) {
            // releasing is only a memory optimization and must not fail a dispatched inference
            LOG_WARNING_TAG("[dynamic] failed to release inactive devices: %s", ex.what());
        }
    } catch (const std::exception&) {
        m_this_worker_infer_request = nullptr;
        release_execution_slot();
        // report the failure through the pipeline, otherwise the request would never complete; the task is gone
        // once it has been dispatched, so the exception must not be left behind on this thread in that case
        if (pipeline_task) {
            m_this_scheduling_exception = std::current_exception();
            pipeline_task();
        }
    }
}

DeviceInformation AutoSchedule::select_dynamic_device() {
    const auto start_time = std::chrono::steady_clock::now();
    DeviceInformation device;
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        const auto low_power_device =
            m_context->m_low_power_device_unavailable ? std::nullopt : m_context->m_low_power_device;
        // the low power device can run on its own once every other candidate has been excluded
        OPENVINO_ASSERT(!m_context->m_device_priorities.empty() ||
                            (low_power_device && m_plugin->get_low_power_mode().value_or(false)),
                        get_log_tag(), ": no candidate device left to run the inference");
        auto candidates = available_candidates();
        if (candidates.empty()) {
            // every candidate failed recently, retrying one of them early beats failing the inference
            candidates = m_context->m_device_priorities;
        }
        device = m_plugin->select_device(candidates,
                                        m_context->m_model_precision,
                                        m_context->m_model_priority,
                                        m_context->m_selection_policy,
                                        low_power_device);
    }
    // select_device() reserves nothing in the dynamic selection
    LOG_DEBUG_TAG("[dynamic] selected device:%s in %lf ms",
                  device.device_name.c_str(),
                  std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count());
    return device;
}

bool AutoSchedule::ensure_device_ready(DeviceInformation& device) {
    const auto* device_context = find_device_context(device.device_name);
    if (device_context && device_context->is_ready()) {
        return true;
    }
    const auto start_time = std::chrono::steady_clock::now();
    const auto requested_device_name = device.device_name;
    CompileRequest request;
    request.m_device_info = device;
    request.m_model_precision = m_context->m_model_precision;
    {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        request.m_meta_devices = available_candidates();
    }
    const auto candidate_devices = request.m_meta_devices;
    const auto state = device_context ? device_context->state() : DeviceExecutionContext::State::Unloaded;
    if (state == DeviceExecutionContext::State::Released) {
        LOG_INFO_TAG("[dynamic] device:%s was released, compiling the model again", device.device_name.c_str());
    } else if (state == DeviceExecutionContext::State::Faulted) {
        LOG_INFO_TAG("[dynamic] device:%s failed to compile before (%s), compiling the model again",
                     device.device_name.c_str(),
                     device_context->last_error().c_str());
    } else {
        LOG_INFO_TAG("[dynamic] device:%s is used for the first time, compiling the model", device.device_name.c_str());
    }
    // select_dynamic_device() reserved nothing for the requested device
    request.m_priority_released = true;
    auto result = try_to_compile_model(request, m_dynamic_model ? m_dynamic_model->clone() : nullptr);
    const bool fell_back_to_other_device = request.m_device_info.device_name != requested_device_name;
    // try_to_compile_model() drops every device that failed to compile from its own candidate list
    std::vector<std::string> failed_devices;
    for (const auto& candidate : candidate_devices) {
        const auto remaining = std::find_if(request.m_meta_devices.begin(),
                                            request.m_meta_devices.end(),
                                            [&candidate](const DeviceInformation& remaining_device) {
                                                return remaining_device.device_name == candidate.device_name;
                                            });
        if (remaining == request.m_meta_devices.end()) {
            failed_devices.push_back(candidate.device_name);
        }
    }
    const auto add_failed_device = [&failed_devices](const std::string& name) {
        if (std::find(failed_devices.begin(), failed_devices.end(), name) == failed_devices.end()) {
            failed_devices.push_back(name);
        }
    };
    if (fell_back_to_other_device || !result.m_succeeded) {
        add_failed_device(requested_device_name);
    }
    if (!result.m_succeeded) {
        // a failed CPU is never erased from the candidate list by try_to_compile_model(), so it is not found above
        add_failed_device(request.m_device_info.device_name);
    }
    if (!failed_devices.empty()) {
        handle_compile_failures(failed_devices);
    }
    if (!result.m_succeeded) {
        LOG_WARNING_TAG("[dynamic] compiling the model on device:%s failed, %s",
                        device.device_name.c_str(),
                        result.m_error.c_str());
        return false;
    }
    device = request.m_device_info;
    if (!get_device_context(device.device_name).is_ready()) {
        try {
            register_compile_result(request, std::move(result.m_compiled_model));
        } catch (const std::exception& ex) {
            // the compilation worked but the workers could not be created; without this the device is retried forever
            mark_compile_failure(request, ex.what());
            handle_compile_failures({device.device_name});
            LOG_WARNING_TAG("[dynamic] creating the workers on device:%s failed, %s",
                            device.device_name.c_str(),
                            ex.what());
            return false;
        }
    } else {
        // the fallback landed on a device this schedule already compiled and has a worker for;
        // reuse it instead of generating a second (duplicate) worker for the same device
        LOG_DEBUG_TAG("[dynamic] fallback device:%s is already cached, reusing its compiled model/worker",
                      device.device_name.c_str());
    }
    m_failure_policy.record_success(device.device_name);
    LOG_INFO_TAG("[dynamic] device:%s is ready in %lf ms",
                 device.device_name.c_str(),
                 std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_time).count());
    return true;
}

void AutoSchedule::release_inactive_devices(const DeviceName& active_device) {
    if (!m_eviction_policy || active_device.empty()) {
        return;
    }
    const auto* active = find_device_context(active_device);
    // get_active_device() relies on this: the active device is published before the previous one is released
    if (!active || !active->is_ready()) {
        return;
    }
    std::vector<InstanceUsage> inactive;
    m_instances.for_each([&](DeviceExecutionContext& context) {
        // the startup helper is released by its own recycle task
        if (context.key() == active_device || context.key().is_startup_helper() || !context.is_ready() ||
            context.is_kept_alive()) {
            return;
        }
        InstanceUsage usage;
        usage.instance = context.key();
        usage.reloadable = is_reloadable(context.name());
        usage.stats = m_compile_service->stats(context.name());
        usage.last_used = context.last_used();
        inactive.push_back(std::move(usage));
    });
    if (inactive.empty()) {
        return;
    }
    const auto decision = m_eviction_policy->decide(inactive, std::chrono::steady_clock::now());
    for (const auto& instance : decision.keep_resident) {
        LOG_INFO_TAG("[dynamic] device:%s cannot be reloaded from a cache, keeping its compiled model",
                     instance.to_string().c_str());
        get_device_context(instance).keep_alive();
    }
    for (const auto& instance : decision.release) {
        if (get_device_context(instance).try_release()) {
            LOG_INFO_TAG("[dynamic] device:%s is inactive, its compiled model is released",
                         instance.to_string().c_str());
        }
    }
}

bool AutoSchedule::is_reloadable(const DeviceName& device_name) {
    const auto& candidates = m_context->m_device_priorities_initial;
    const auto device = std::find_if(candidates.begin(), candidates.end(), [&](const DeviceInformation& info) {
        return info.device_name == device_name;
    });
    if (device == candidates.end()) {
        return false;
    }
    return m_compile_service->has_cache_blob(device_name, m_dynamic_model);
}

std::vector<DeviceInformation> AutoSchedule::available_candidates() const {
    const auto now = std::chrono::steady_clock::now();
    std::vector<DeviceInformation> candidates;
    for (const auto& device : m_context->m_device_priorities) {
        if (m_failure_policy.is_available(device.device_name, now)) {
            candidates.push_back(device);
        }
    }
    return candidates;
}

void AutoSchedule::handle_compile_failures(const std::vector<DeviceName>& devices) {
    const auto now = std::chrono::steady_clock::now();
    std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
    for (const auto& name : devices) {
        if (!m_failure_policy.record_failure(name, now)) {
            LOG_WARNING_TAG("[dynamic] device:%s failed to compile, it is not selected for a while", name.c_str());
            continue;
        }
        const auto iter = deviceChecker().check_and_return_if_device_in_list<DeviceInformation>(
            name, m_context->m_device_priorities, true);
        if (iter != m_context->m_device_priorities.end()) {
            m_context->m_device_priorities.erase(iter);
            LOG_WARNING_TAG("[dynamic] device:%s failed to compile too many times, excluding it from future selection",
                            name.c_str());
        }
    }
}

void AutoSchedule::release_execution_slot() {
    std::pair<ov::threading::Task, DeviceName> next;
    std::shared_ptr<ov::threading::IStreamsExecutor> executor;
    {
        std::unique_lock<std::mutex> lock(m_gate_mutex);
        if (m_dynamic_shutdown) {
            // AutoSchedule is being torn down, do not touch m_dynamic_executor anymore; the queued requests can no
            // longer run, so they are failed instead of being left to hang their callers.
            // The gate stays busy meanwhile so that ~AutoSchedule() does not finish under them.
            auto abandoned_tasks = std::move(m_gate_pending_tasks);
            m_gate_pending_tasks.clear();
            lock.unlock();
            if (!abandoned_tasks.empty()) {
                // it is surfaced to the caller as an ov::Exception by the scheduling stage of the pipeline
                const auto cancelled =
                    std::make_exception_ptr(std::runtime_error("Model is being destroyed, the request is cancelled"));
                for (auto& abandoned : abandoned_tasks) {
                    m_this_scheduling_exception = cancelled;
                    abandoned.first();
                }
            }
            lock.lock();
            m_gate_busy = false;
            // notified under the lock: it is the last access of the device callback to this object
            m_dispatch_done.notify_all();
            return;
        }
        m_gate_busy = false;
        if (m_gate_pending_tasks.empty()) {
            return;
        }
        next = std::move(m_gate_pending_tasks.front());
        m_gate_pending_tasks.pop_front();
        m_gate_busy = true;
        executor = m_dynamic_executor;
        ++m_active_dispatch;
    }
    LOG_DEBUG_TAG("[dynamic] inference finished, dispatching the next queued request");
    submit_dynamic_dispatch(executor, std::move(next.first), next.second);
}

AutoSchedule::~AutoSchedule() {
    std::shared_ptr<ov::threading::IStreamsExecutor> dynamic_executor;
    {
        // serialize with schedule_dynamic_task()/release_execution_slot(), which read/snapshot
        // m_dynamic_executor under the same lock, so none of them races with resetting it below
        std::unique_lock<std::mutex> lock(m_gate_mutex);
        m_dynamic_shutdown = true;
        dynamic_executor = std::move(m_dynamic_executor);
        // in-flight dispatch tasks hold a raw this and may still finish a failed request, and the device completion
        // callback of a running inference still uses this after it has notified the user request, so drain both
        m_dispatch_done.wait(lock, [this] {
            return m_active_dispatch == 0 && !m_gate_busy;
        });
    }
    if (dynamic_executor) {
        LOG_INFO_TAG("[dynamic] %s", m_decision_log.summary(m_instance_stats.snapshot()).c_str());
        // do not clear the executor manager's "AutoDynamicSchedule" entry here: clear() works by name and would
        // also evict the executors of other alive AutoSchedule instances. get_idle_cpu_streams_executor() only
        // hands out an executor nobody else holds, so instances never share one; just drop our own reference
        // and let the executor manager keep it for reuse.
        dynamic_executor.reset();
    }
    // this is necessary to guarantee member destroyed after getting future
    if (m_compile_context[CPU].m_is_enabled) {
        m_exitflag = true;
        m_compile_context[CPU].m_future.wait();
        wait_actual_compiled_model_ready();
        // it's necessary to wait the compile model threads to stop here.
        m_plugin->get_executor_manager()->clear("AutoDeviceAsyncCompile");
        m_executor.reset();
    }
    // After the recycle task has stopped, so no new pre-compilation can be queued.
    m_compile_service->stop_precompile();
    if (m_plugin && !m_context->m_dynamic_device_selection) {
        m_plugin->unregister_priority(m_context->m_model_priority,
                                      m_compile_context[ACTUALDEVICE].request.m_device_info.unique_name);
        const auto& fallback = m_compile_context[FALLBACKDEVICE];
        if (!fallback.request.m_priority_released && !fallback.request.m_device_info.unique_name.empty()) {
            m_plugin->unregister_priority(m_context->m_model_priority, fallback.request.m_device_info.unique_name);
        }
    }
    if (m_context) {
        std::lock_guard<std::mutex> lock(m_context->m_fallback_mutex);
        m_context->m_device_priorities.clear();
    }
    /* NOTE: The only threads that use `MultiSchedule` worker infer requests' threads.
     *       But AsyncInferRequest destructor should wait for all asynchronous tasks by the request
     */
    // stop accepting any idle requests back (for re-scheduling)
    m_instances.for_each([](DeviceExecutionContext& context) {
        context.stop_scheduling();
    });
}
}  // namespace auto_plugin
}  // namespace ov
