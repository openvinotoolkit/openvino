// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "schedule.hpp"
#include "async_infer_request.hpp"

namespace ov {
namespace auto_plugin {
thread_local WorkerInferRequest* Schedule::m_this_worker_infer_request = nullptr;
// TODO: revert to the plain variable (see header file), when we moved to the next CentOS 8.x in our support matrix
thread_local const char* Schedule::m_this_preferred_device_name = "";
thread_local std::exception_ptr Schedule::m_this_scheduling_exception = nullptr;

void Schedule::launch(const ScheduleContext::Ptr& context) {
    m_context = context;
    m_log_tag = context->m_log_tag;
    m_plugin = std::const_pointer_cast<Plugin>(std::dynamic_pointer_cast<const Plugin>(context->m_plugin));
    m_compile_service = std::make_unique<CompileService>(context->m_ov_core, context->m_model_path, m_log_tag);
    LOG_INFO_TAG("scheduler starting");
    init();
}

ISyncInferPtr Schedule::create_sync_infer_request() {
    auto compiled_model = std::dynamic_pointer_cast<CompiledModel>(
            m_context->m_compiled_model.lock());
    SoAsyncInferRequest request_to_share_tensors_with;
    auto request_id = m_request_id.fetch_add(1);
    if (m_context->m_bind_buffer) {
        size_t sum = 0;
        for (const auto& device : m_context->m_device_priorities_initial) {
            const auto context = find_device_context(device.device_name);
            if (!context) {
                continue;
            }
            auto& pool = context->pool();
            const auto pool_size = pool.size();
            if ((request_id - sum) < pool_size) {
                auto* worker = &pool.worker(request_id - sum);
                request_to_share_tensors_with = worker->m_inferrequest;
                INFO_RUN([&]() {
                    std::lock_guard<std::mutex> lock(m_dev_infer_mutex);
                    m_dev_infer.insert(std::make_pair(request_to_share_tensors_with._ptr, worker));
                });
                break;
            }
            sum += pool_size;
        }
        if (!request_to_share_tensors_with) {
            OPENVINO_THROW("binder mode does not allow oversubsciption of infer requests, please use optimal infer request");
        }
    } else if (m_passthrough_context) {
        const auto& passthrough_model = m_passthrough_context->compiled_model();
        request_to_share_tensors_with = {passthrough_model->create_infer_request(), passthrough_model._so};
    }
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const CompiledModel>(compiled_model), request_to_share_tensors_with);
}

void Schedule::run(ov::threading::Task pipeline_task) {
    schedule_to_worker_infer_request(std::move(pipeline_task), m_this_preferred_device_name);
}

bool Schedule::run_pipeline_task(ov::threading::Task& pipeline_task, WorkerPool& pool) {
    auto lease = pool.try_acquire();
    if (!lease) {
        return false;
    }
    m_this_worker_infer_request = lease.get();
    {
        auto captured_task = std::move(pipeline_task);
        captured_task();
    }
    // the inference has been started, its completion callback gives the worker back
    lease.detach();
    return true;
}

void Schedule::capture_hw_metadata(const SoCompiledModel& first_compiled_model) {
    auto metadata = std::make_shared<CompiledModelMetadata>();
    metadata->inputs = first_compiled_model->inputs();
    metadata->outputs = first_compiled_model->outputs();
    metadata->so = first_compiled_model._so;
    try {
        metadata->model_name = first_compiled_model->get_property(ov::model_name.name()).as<std::string>();
    } catch (const ov::Exception&) {
        LOG_DEBUG_TAG("underlying hardware does not report the model name");
    }
    try {
        metadata->context = first_compiled_model->get_context();
        if (!metadata->context._so) {
            metadata->context._so = first_compiled_model._so;
        }
    } catch (ov::NotImplemented&) {
        LOG_INFO_TAG("underlying hardware does not support hardware context");
    }
    if (metadata->context) {
        // the default context belongs to the device plugin, so it does not depend on the first compiled model, which
        // the dynamic selection may release
        try {
            if (auto default_context = m_context->m_ov_core->get_default_context(metadata->context->get_device_name())) {
                metadata->context = std::move(default_context);
            }
        } catch (const ov::Exception&) {
            LOG_DEBUG_TAG("no default context for device %s, keeping the context of the first compiled model",
                          metadata->context->get_device_name().c_str());
        }
    }
    m_context->m_hw_metadata = std::move(metadata);
}

DeviceExecutionContext& Schedule::add_device_context(const ModelInstanceKey& key) {
    return m_instances.add(key);
}

DeviceExecutionContext& Schedule::get_device_context(const ModelInstanceKey& key) {
    const auto context = find_device_context(key);
    OPENVINO_ASSERT(context, get_log_tag(), ": device ", key.to_string(), " has no execution context");
    return *context;
}

DeviceExecutionContext* Schedule::find_device_context(const ModelInstanceKey& key) {
    return m_instances.find(key);
}

ActiveDevice Schedule::get_active_device() const {
    return m_instances.active_device();
}

DeviceExecutionContext& Schedule::register_compile_result(const CompileRequest& request, SoCompiledModel compiled_model) {
    const auto key = request.instance_key();
    OPENVINO_ASSERT(compiled_model, get_log_tag(), ": no compiled model to register for ", key.to_string());
    generate_workers(key, compiled_model);
    return get_device_context(key);
}

void Schedule::mark_compile_failure(const CompileRequest& request, const std::string& reason) {
    if (auto* device_context = find_device_context(request.instance_key())) {
        device_context->mark_faulted(reason);
    }
}
void Schedule::generate_workers(const ModelInstanceKey& key, const SoCompiledModel& compiled_model) {
    auto it_numrequests = deviceChecker().check_and_return_if_device_in_list<DeviceInformation>(key.device, m_context->m_device_priorities, true);
    size_t num_requests = 0;
    if (m_context->m_dynamic_device_selection) {
        // inference is serialized per compiled model in this mode, so a single worker per device is enough and it
        // guarantees the device carries no other inference when the next target device is re-selected
        num_requests = 1;
        LOG_DEBUG_TAG("device:%s, worker infer request number:%d", key.to_string().c_str(), static_cast<int>(num_requests));
    } else {
        unsigned int optimal_num = 0;
        try {
            optimal_num = compiled_model->get_property(ov::optimal_number_of_infer_requests.name()).as<unsigned int>();
        } catch (const ov::Exception& iie) {
            OPENVINO_THROW("Every device used with AUTO should support query optimal_number_of_infer_requests property from compiled model ",
                        iie.what());
        }
        num_requests =
            (m_context->m_device_priorities.end() == it_numrequests || it_numrequests->num_requests_per_devices == -1)
                ? optimal_num
                : it_numrequests->num_requests_per_devices;
        num_requests = (num_requests == 1) ? 2 : num_requests;
    }
    if (m_model_only) {
        num_requests = 0;
    }
    auto setup_worker = [this](DeviceExecutionContext& context, WorkerInferRequest& worker_request) {
        auto* worker_request_ptr = &worker_request;
        auto* device_context = &context;
        worker_request.m_is_cpu_helper = context.key().is_startup_helper();
        worker_request.m_inferrequest->set_callback(
            [worker_request_ptr, this, device_context](std::exception_ptr exception_ptr) mutable {
                auto lease = device_context->pool().adopt(worker_request_ptr);
                worker_request_ptr->m_exception_ptr = std::move(exception_ptr);
                {
                    auto stop_retry_and_continue = [worker_request_ptr]() {
                        auto captured_task = std::move(worker_request_ptr->m_task);
                        captured_task();
                    };
                    // will fallback to other devices if enable m_runtime_fallback
                    if (worker_request_ptr->m_exception_ptr != nullptr && m_context->m_runtime_fallback) {
                        bool select_other_device_flag = false;
                        // select other device
                        try {
                            select_other_device_flag = select_other_device(device_context->key());
                        } catch (const ov::Exception&) {
                            select_other_device_flag = false;
                        }
                        if (select_other_device_flag) {
                            // Add end time to current workerRequest and restart the task in pipeline
                            worker_request_ptr->m_end_times.push_back(std::chrono::steady_clock::now());
                            worker_request_ptr->m_fallback_exec->immediate_task();
                        } else {
                            // continue to run the task in pipeline
                            stop_retry_and_continue();
                        }
                    } else {
                        stop_retry_and_continue();
                    }
                    // try to return the request to the idle list (fails if the overall object destruction has began)
                    const bool returned_to_idle = lease.give_back();
                    if (m_context->m_dynamic_device_selection) {
                        // only recorded here, it is logged by the next dispatch so the slot is released without delay
                        INFO_RUN([&] {
                            if (!worker_request_ptr->m_start_times.empty()) {
                                m_last_infer_ms = std::chrono::duration<double, std::milli>(
                                                      std::chrono::steady_clock::now() -
                                                      worker_request_ptr->m_start_times.back())
                                                      .count();
                            }
                        });
                        // released unconditionally, otherwise queued requests would hang once destruction has began
                        release_execution_slot();
                    } else if (returned_to_idle) {
                        // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                        // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                        ov::threading::Task t;
                        do {
                            m_infer_pipeline_tasks.try_pop(t);
                        } while (t && schedule_to_worker_infer_request(std::move(t)));
                        while (device_context->pool().try_dequeue(t) &&
                               schedule_to_worker_infer_request(std::move(t), device_context->name())) {
                        }
                    }
                }
            });
    };
    get_device_context(key).make_ready(compiled_model, num_requests, setup_worker);
}

Pipeline Schedule::get_async_pipeline(const ISyncInferPtr& infer_request, WorkerInferRequest** worker_infer_request) {
    Pipeline pipeline;
    if (m_passthrough_context || std::static_pointer_cast<InferRequest>(infer_request)->get_shared_request()) {
        struct RequestExecutor : ov::threading::ITaskExecutor {
            explicit RequestExecutor(const SoAsyncInferRequest& infer_request,
                                     WorkerInferRequest* worker)
                : m_inferrequest(infer_request),
                  m_worker(worker) {
                m_inferrequest->set_callback([this](std::exception_ptr exceptionPtr) mutable {
                    m_exceptionptr = std::move(exceptionPtr);
                    auto capturedTask = std::move(m_task);
                    capturedTask();
                    INFO_RUN([&]() {
                        if (m_worker) {
                            m_worker->m_end_times.push_back(std::chrono::steady_clock::now());
                        }
                    });
                });
            }
            void run(ov::threading::Task task) override {
                m_task = std::move(task);
                INFO_RUN([&]() {
                if (m_worker) {
                    m_worker->m_start_times.push_back(std::chrono::steady_clock::now());
                }
            });
                m_inferrequest->start_async();
            };
            const SoAsyncInferRequest& m_inferrequest;
            std::exception_ptr m_exceptionptr;
            ov::threading::Task m_task;
            WorkerInferRequest* m_worker;
        };
        auto& shared_request = std::static_pointer_cast<InferRequest>(infer_request)->get_shared_request();
        WorkerInferRequest* worker = nullptr;
        INFO_RUN([&]() {
            std::lock_guard<std::mutex> lock(m_dev_infer_mutex);
            auto iter = m_dev_infer.find(shared_request._ptr);
            if (iter != m_dev_infer.end()) {
                worker = iter->second;
            }
        });
        auto requestExecutor =
            std::make_shared<RequestExecutor>(shared_request, worker);
        pipeline.emplace_back(requestExecutor, [requestExecutor] {
            if (nullptr != requestExecutor->m_exceptionptr) {
                std::rethrow_exception(requestExecutor->m_exceptionptr);
            }
        });
    } else {
        AutoImmediateExecutor::Ptr first_executor = std::make_shared<AutoImmediateExecutor>();
        pipeline = {
            // if the request is coming with device-specific remote tensors make sure it is scheduled to the specific device only:
            Stage {
                /*TaskExecutor*/ first_executor, /*task*/ [this, &infer_request]() {
                    std::static_pointer_cast<InferRequest>(infer_request)->reset_scheduled_profiling();
                    // by default, no preferred device:
                    m_this_preferred_device_name = "";
                    auto compiled_model = m_context->m_compiled_model.lock();
                    auto find_remote_device = [this](const ov::SoPtr<ov::ITensor>& tensor) -> const char* {
                        auto remote_tensor = std::dynamic_pointer_cast<ov::IRemoteTensor>(tensor._ptr);
                        if (!remote_tensor) {
                            return nullptr;
                        }
                        const auto name = remote_tensor->get_device_name();
                        auto matches = [&name](const DeviceInformation& d) {
                            return (d.default_device_id.empty() ? d.device_name
                                                                : (d.device_name + "." + d.default_device_id)) == name;
                        };
                        const auto res = std::find_if(m_context->m_device_priorities_initial.cbegin(),
                                                      m_context->m_device_priorities_initial.cend(),
                                                      matches);
                        if (res != m_context->m_device_priorities_initial.cend()) {
                            // it is ok to take the c_str() here (as pointed in the schedule.hpp we need to use
                            // const char*) as the original strings are from the "persistent" vector (with the right
                            // lifetime)
                            return res->device_name.c_str();
                        }
                        // the low power device can sit outside the candidates, it is only used when a tensor pins it
                        const auto& low_power = m_context->m_low_power_device;
                        if (low_power && !m_context->m_low_power_device_unavailable && matches(*low_power)) {
                            return low_power->device_name.c_str();
                        }
                        OPENVINO_THROW("None of the devices supports a remote tensor created on the device named ",
                                       name);
                    };
                    // if any input is remote (e.g. was set with set_tensor), let's use the corresponding device
                    for (const auto& it : compiled_model->inputs()) {
                        if (const char* device = find_remote_device(infer_request->get_tensor(it))) {
                            m_this_preferred_device_name = device;
                            break;
                        }
                    }
                    if (m_context->m_dynamic_device_selection) {
                        // the target device is re-selected per inference, so a remote output pins the request as well
                        for (const auto& it : compiled_model->outputs()) {
                            if (const char* device = find_remote_device(infer_request->get_tensor(it))) {
                                OPENVINO_ASSERT(m_this_preferred_device_name[0] == '\0' ||
                                                    std::string(m_this_preferred_device_name) == device,
                                                "Remote tensors of the inputs and outputs are created on different devices");
                                m_this_preferred_device_name = device;
                            }
                        }
                    }
                }},
            // as the scheduling algo may select any device, this stage accepts the scheduling decision (actual workerRequest)
            // then sets the device-agnostic tensors to the actual (device-specific) request
            Stage {
                /*TaskExecutor*/std::dynamic_pointer_cast<ov::threading::ITaskExecutor>(shared_from_this()), /*task*/ [&infer_request, worker_infer_request]() {
                    if (m_this_scheduling_exception) {
                        auto scheduling_exception = m_this_scheduling_exception;
                        m_this_scheduling_exception = nullptr;
                        // the failure can come from a device plugin, so make sure an ov::Exception is surfaced
                        try {
                            std::rethrow_exception(scheduling_exception);
                        } catch (const ov::Exception&) {
                            throw;
                        } catch (const std::exception& ex) {
                            OPENVINO_THROW(ex.what());
                        }
                    }
                    *worker_infer_request = m_this_worker_infer_request;
                    auto auto_request = std::dynamic_pointer_cast<InferRequest>(infer_request);
                    DispatchFailureGuard guard{m_this_worker_infer_request};
                    auto_request->set_tensors_to_another_request(m_this_worker_infer_request->m_inferrequest);
                    guard.m_dismissed = true;
                    INFO_RUN([worker_infer_request]() {
                        (*worker_infer_request)->m_start_times.push_back(std::chrono::steady_clock::now());
                        });
                }},
            // final task in the pipeline:
            Stage {
                /*TaskExecutor*/std::make_shared<ThisRequestExecutor>(worker_infer_request, first_executor), /*task*/
                [this, &infer_request, worker_infer_request]() {
                    INFO_RUN([worker_infer_request]() {
                        (*worker_infer_request)->m_end_times.push_back(std::chrono::steady_clock::now());
                    });
                    std::exception_ptr eptr = (*worker_infer_request)->m_exception_ptr;
                    if (nullptr != eptr) {
                        std::rethrow_exception(eptr);
                    }
                    if (m_context->m_need_perf_counters) {
                        auto auto_request = std::dynamic_pointer_cast<InferRequest>
                            (infer_request);
                        // the CPU helper is released once the actual device is ready, so its request is not pinned
                        if (!m_context->m_dynamic_device_selection && !(*worker_infer_request)->m_is_cpu_helper) {
                            auto_request->set_scheduled_request((*worker_infer_request)->m_inferrequest);
                            return;
                        }
                        // the device may be released, so snapshot instead of pinning it
                        try {
                            auto_request->set_scheduled_profiling_info(
                                (*worker_infer_request)->m_inferrequest->get_profiling_info());
                        } catch (...) {
                            // surfaced from get_profiling_info(), the inference itself succeeded
                            auto_request->set_scheduled_profiling_error(std::current_exception());
                        }
                    }
                }}
        };
    }
    return pipeline;
}

std::string Schedule::get_log_tag() const noexcept {
    return m_log_tag;
}

Schedule::~Schedule() {
    INFO_RUN([this] {
        const bool serialized_inference = m_context && m_context->m_dynamic_device_selection;
        m_instances.for_each([&](DeviceExecutionContext& context) {
            context.log_summary(m_log_tag, m_cpuhelp_release_time, serialized_inference);
        });
    });
    m_instances.clear();
    LOG_INFO_TAG("scheduler ending");
}
}  // namespace auto_plugin
}  // namespace ov
