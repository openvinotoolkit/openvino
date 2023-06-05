// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "schedule.hpp"
#include "async_infer_request.hpp"

namespace ov {
namespace auto_plugin {
thread_local WorkerInferRequest* Schedule::m_this_worker_inferrequest = nullptr;
// TODO: revert to the plain variable (see header file), when we moved to the next CentOS 8.x in our support matrix
thread_local const char* Schedule::m_this_preferred_devicename = "";

void Schedule::launch(ScheduleContext::Ptr context) {
    m_context = context;
    m_log_tag = context->m_log_tag;
    m_plugin = std::const_pointer_cast<Plugin>(std::dynamic_pointer_cast<const Plugin>(context->m_plugin));
    LOG_INFO_TAG("scheduler starting");
    init();
}

ISyncInferPtr Schedule::create_sync_infer_request() {
    auto compiled_model = std::dynamic_pointer_cast<CompiledModel>(
            m_context->m_compiled_model.lock());
    SoAsyncInferRequest request_to_share_blobs_with;
    auto request_id = m_request_id.fetch_add(1);
    if (m_passthrough_exenet) {
        request_to_share_blobs_with = {m_passthrough_exenet->create_infer_request(), m_passthrough_exenet._so};
    } else if (m_context->m_bind_buffer) {
        size_t sum = 0;
        for (const auto& device : m_context->m_device_priorities_initial) {
            auto& dev_requests = m_workerrequests[device.device_name];
            if ((request_id - sum) <  dev_requests.size()) {
                request_to_share_blobs_with = dev_requests.at(request_id - sum).m_inferrequest;
                break;
            }
            sum += dev_requests.size();
        }
        if (!request_to_share_blobs_with) {
            OPENVINO_THROW("binder mode does not allow oversubsciption of infer requests, please use optimal infer request");
        }
    }
    return std::make_shared<InferRequest>(
        std::static_pointer_cast<const CompiledModel>(compiled_model), request_to_share_blobs_with);
}

void Schedule::run(ov::threading::Task pipeline_task) {
    schedule_to_worker_inferrequest(std::move(pipeline_task), m_this_preferred_devicename);
}

bool Schedule::run_pipeline_task(ov::threading::Task& pipeline_task,
    NotBusyPriorityWorkerRequests& idle_workerrequests,
    const DeviceName& preferred_device) {
    WorkerInferRequest* worker_request_ptr = nullptr;
    std::pair<int, WorkerInferRequest*> worker;
    if (idle_workerrequests.try_pop(worker)) {
        worker_request_ptr = worker.second;
        IdleGuard<NotBusyPriorityWorkerRequests> idle_guard{worker_request_ptr, idle_workerrequests};
        m_this_worker_inferrequest = worker_request_ptr;
        {
            auto captured_task = std::move(pipeline_task);
            captured_task();
        }
        idle_guard.release();
        return true;
    }
    return false;
}

void Schedule::generate_workers(const std::string& device, const SoCompiledModel& executable_network) {
    std::string real_devicename;
    if (device == "CPU_HELP") {
        real_devicename = "CPU";
    } else {
        real_devicename = device;
    }
    auto it_numrequests = deviceChecker().check_and_return_if_device_in_list<DeviceInformation>(real_devicename, m_context->m_device_priorities, true);
    unsigned int optimal_num = 0;
    try {
        optimal_num = executable_network->get_property(ov::optimal_number_of_infer_requests.name()).as<unsigned int>();
    } catch (const ov::Exception& iie) {
        OPENVINO_THROW("Every device used with the Multi-Device should support OPTIMAL_NUMBER_OF_INFER_REQUESTS ExecutableNetwork metric",
                    iie.what());
    }
    const auto num_requests = (m_context->m_device_priorities.end() == it_numrequests ||
                              it_numrequests->num_requests_per_devices == -1) ? optimal_num : it_numrequests->num_requests_per_devices;
    auto& worker_requests = m_workerrequests[device];
    auto& idle_worker_requests = m_idle_workerrequests[device];
    worker_requests.resize(num_requests);
    m_infer_pipelinetasks_devicespecific[device] = std::unique_ptr<TaskQueue>(new TaskQueue);
    auto* idle_workerrequests_ptr = &(idle_worker_requests);
    idle_worker_requests.set_capacity(num_requests);
    int num = 0;
    for (auto&& worker_request : worker_requests) {
        worker_request.m_inferrequest = {executable_network->create_infer_request(), executable_network._so};
        auto* worker_request_ptr = &worker_request;
        worker_request_ptr->m_index = num++;
        IE_ASSERT(idle_worker_requests.try_push(std::make_pair(worker_request_ptr->m_index, worker_request_ptr)) == true);
        worker_request.m_inferrequest->set_callback(
            [worker_request_ptr, this, device, idle_workerrequests_ptr](std::exception_ptr exception_ptr) mutable {
                IdleGuard<NotBusyPriorityWorkerRequests> idleGuard{worker_request_ptr, *idle_workerrequests_ptr};
                worker_request_ptr->m_exception_ptr = exception_ptr;
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
                            select_other_device_flag = select_other_device(device);
                        } catch (const ov::Exception& iie) {
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
                    if (idleGuard.release()->try_push(std::make_pair(worker_request_ptr->m_index, worker_request_ptr))) {
                        // let's try to pop a task, as we know there is at least one idle request, schedule if succeeded
                        // if no device-agnostic tasks, let's try pop the device specific task, schedule if succeeded
                        ov::threading::Task t;
                        do {
                            m_infer_pipelinetasks.try_pop(t);
                        } while (t && schedule_to_worker_inferrequest(std::move(t)));
                        do {
                            m_infer_pipelinetasks_devicespecific[device]->try_pop(t);
                        } while (t && schedule_to_worker_inferrequest(std::move(t), device));
                    }
                }
            });
    }
}

Pipeline Schedule::get_async_pipeline(const ISyncInferPtr& infer_request, WorkerInferRequest** worker_inferrequest) {
    Pipeline pipeline;
    if (m_passthrough_exenet || std::static_pointer_cast<InferRequest>(infer_request)->get_shared_request()) {
        struct RequestExecutor : ov::threading::ITaskExecutor {
            explicit RequestExecutor(const SoAsyncInferRequest& infer_request) : m_inferrequest(infer_request) {
                m_inferrequest->set_callback([this](std::exception_ptr exceptionPtr) mutable {
                    m_exceptionptr = exceptionPtr;
                    auto capturedTask = std::move(m_task);
                    capturedTask();
                });
            }
            void run(ov::threading::Task task) override {
                m_task = std::move(task);
                m_inferrequest->start_async();
            };
            const SoAsyncInferRequest& m_inferrequest;
            std::exception_ptr m_exceptionptr;
            ov::threading::Task m_task;
        };
        auto requestExecutor =
            std::make_shared<RequestExecutor>(std::static_pointer_cast<InferRequest>(infer_request)->get_shared_request());
        pipeline.emplace_back(requestExecutor, [requestExecutor] {
            if (nullptr != requestExecutor->m_exceptionptr) {
                std::rethrow_exception(requestExecutor->m_exceptionptr);
            }
        });
    } else {
        AutoImmediateExecutor::Ptr _firstExecutor = std::make_shared<AutoImmediateExecutor>();
        pipeline = {
            // if the request is coming with device-specific remote blobs make sure it is scheduled to the specific device only:
            Stage {
                /*TaskExecutor*/ _firstExecutor, /*task*/ [this, &infer_request]() {
                    // by default, no preferred device:
                    m_this_preferred_devicename = "";
                    auto exec_network = m_context->m_compiled_model.lock();
                    // if any input is remote (e.g. was set with SetBlob), let' use the corresponding device
                    for (const auto& it : exec_network->inputs()) {
                        auto b = infer_request->get_tensor(it);
                        auto r = b.is<ov::RemoteTensor>();
                        if (r) {
                            const auto name = b.as<ov::RemoteTensor>().get_device_name();
                            const auto res = std::find_if(
                                m_context->m_device_priorities_initial.cbegin(),
                                m_context->m_device_priorities_initial.cend(),
                            [&name](const DeviceInformation & d) {
                                return (d.default_device_id.empty() ? d.device_name : (d.device_name + "." +
                                        d.default_device_id)) == name;
                            });
                            if (m_context->m_device_priorities_initial.cend() == res) {
                                OPENVINO_THROW(
                                    "None of the devices supports a remote blob created on the device named ", name);
                            } else {
                                // it is ok to take the c_str() here (as pointed in the executable_network.hpp we need to use const char*)
                                // as the original strings are from the "persistent" vector (with the right lifetime)
                                m_this_preferred_devicename = res->device_name.c_str();
                                break;
                            }
                        }
                    }
                }},
            // as the scheduling algo may select any device, this stage accepts the scheduling decision (actual workerRequest)
            // then sets the device-agnostic blobs to the actual (device-specific) request
            Stage {
                /*TaskExecutor*/std::dynamic_pointer_cast<ov::threading::ITaskExecutor>(shared_from_this()), /*task*/ [&infer_request, worker_inferrequest]() {
                    *worker_inferrequest = m_this_worker_inferrequest;
                    auto auto_request = std::dynamic_pointer_cast<InferRequest>(infer_request);
                    auto_request->set_tensors_to_another_request(m_this_worker_inferrequest->m_inferrequest);
                    INFO_RUN([worker_inferrequest]() {
                        (*worker_inferrequest)->m_start_times.push_back(std::chrono::steady_clock::now());
                        });
                }},
            // final task in the pipeline:
            Stage {
                /*TaskExecutor*/std::make_shared<ThisRequestExecutor>(worker_inferrequest, _firstExecutor), /*task*/
                [this, &infer_request, worker_inferrequest]() {
                    INFO_RUN([worker_inferrequest]() {
                        (*worker_inferrequest)->m_end_times.push_back(std::chrono::steady_clock::now());
                    });
                    std::exception_ptr eptr = (*worker_inferrequest)->m_exception_ptr;
                    if (nullptr != eptr) {
                        std::rethrow_exception(eptr);
                    }
                    if (m_context->m_need_perf_counters) {
                        auto auto_request = std::dynamic_pointer_cast<InferRequest>
                            (infer_request);
                        auto_request->set_scheduled_request((*worker_inferrequest)->m_inferrequest);
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
        for (auto&& worker_request : m_workerrequests) {
            std::list<Time> req_all_start_times;
            std::list<Time> req_all_end_times;
            for (auto& request : worker_request.second) {
                req_all_start_times.splice(req_all_start_times.end(), request.m_start_times);
                req_all_end_times.splice(req_all_end_times.end(), request.m_end_times);
            }
            size_t count = req_all_start_times.size();
            IE_ASSERT(count == req_all_end_times.size());
            req_all_start_times.sort(std::less<Time>());
            req_all_end_times.sort(std::less<Time>());
            {
                LOG_INFO_TAG("%s:infer:%ld", worker_request.first.c_str(), count);
                auto n = req_all_start_times.size();
                Time time;
                while (!req_all_start_times.empty()) {
                    time = req_all_start_times.front();
                    if (time < m_cpuhelp_release_time) {
                        req_all_start_times.pop_front();
                        n--;
                    } else {
                        break;
                    }
                }
                if (n >= 1) {
                    std::chrono::duration<double, std::milli> durtation =
                        req_all_end_times.back() - time;
                    LOG_INFO_TAG("%s:fps:%lf", worker_request.first.c_str(),
                        n * 1000 / durtation.count());
                }
            }
        }
    });
    m_workerrequests.clear();
    LOG_INFO_TAG("scheduler ending");
}
}  // namespace auto_plugin
}  // namespace ov