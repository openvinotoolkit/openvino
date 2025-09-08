// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <string>

#include "openvino/runtime/auto/properties.hpp"
#include "openvino/runtime/iasync_infer_request.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/icore.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/remote_tensor.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "openvino/runtime/threading/thread_safe_containers.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define auto_plugin mock_auto_plugin
#else
#define MOCKTESTMACRO
#endif

namespace ov {
namespace auto_plugin {
using DeviceName = std::string;
using IASyncInferPtr = std::shared_ptr<ov::IAsyncInferRequest>;
using ISyncInferPtr = std::shared_ptr<ov::ISyncInferRequest>;
using SoAsyncInferRequest = ov::SoPtr<ov::IAsyncInferRequest>;
using SoCompiledModel = ov::SoPtr<ov::ICompiledModel>;
using Time = std::chrono::time_point<std::chrono::steady_clock>;
using Stage = std::pair<std::shared_ptr<ov::threading::ITaskExecutor>, ov::threading::Task>;
using Pipeline = std::vector<Stage>;

template<typename T>
using DeviceMap = std::unordered_map<DeviceName, T>;
// Bell to do, check if needed, or just use immediate exectutor is enough
struct AutoImmediateExecutor : public ov::threading::ITaskExecutor {
public:
    /**
     * @brief A shared pointer to a ImmediateExecutor object
     */
    using Ptr = std::shared_ptr<AutoImmediateExecutor>;

    /**
     * @brief Destroys the object.
     */
    ~AutoImmediateExecutor() override = default;

    void run(ov::threading::Task task) override {
        immediate_task = std::move(task);
        immediate_task();
    }
    ov::threading::Task immediate_task;
};

struct WorkerInferRequest {
    SoAsyncInferRequest           m_inferrequest;
    ov::threading::Task           m_task;
    std::exception_ptr            m_exception_ptr = nullptr;
    std::list<Time>               m_start_times;
    std::list<Time>               m_end_times;
    int                           m_index = 0;
    AutoImmediateExecutor::Ptr    m_fallback_exec;
};

struct ThisRequestExecutor : public ov::threading::ITaskExecutor {
    explicit ThisRequestExecutor(WorkerInferRequest** ptr, AutoImmediateExecutor::Ptr executor = nullptr):
        m_workptrptr{ptr},
        m_fallback_exec(std::move(executor)) {}
    void run(ov::threading::Task task) override {
        (*m_workptrptr)->m_task = std::move(task);
        (*m_workptrptr)->m_fallback_exec = m_fallback_exec;
        (*m_workptrptr)->m_inferrequest->start_async();
    };
    WorkerInferRequest** m_workptrptr = nullptr;
    AutoImmediateExecutor::Ptr m_fallback_exec;
};

struct DeviceInformation {
    DeviceName device_name;
    ov::AnyMap config;
    int num_requests_per_devices;
    std::string default_device_id;
    DeviceName unique_name;
    unsigned int device_priority;
    DeviceInformation(DeviceName dn = {}, ov::AnyMap conf = {},
        int n_req = -1, std::string default_id = {}, DeviceName name = {}, unsigned int priority = 0)
        : device_name(std::move(dn)), config(std::move(conf)),
        num_requests_per_devices(n_req), default_device_id(std::move(default_id)), unique_name(std::move(name)), device_priority(priority)
        {}
};

struct deviceChecker {
        template <typename T,
          typename std::enable_if<std::is_same<typename std::decay<T>::type, std::string>::value, bool>::type = true,
          typename U = typename std::vector<T>::const_iterator>
        U check_and_return_if_device_in_list(const std::string& target, const std::vector<T>& device_list, bool exact_match = false) {
            if (exact_match) {
                return std::find_if(device_list.begin(), device_list.end(),
                        [&target](const T& d) { return d == target; });
            }
            return std::find_if(device_list.begin(), device_list.end(),
                            [&target](const T & d) {
                                return d.find(target) != std::string::npos;
                            });
        }
        template <typename T,
          typename std::enable_if<std::is_same<typename std::decay<T>::type, std::string>::value, bool>::type = true>
        bool check_if_device_in_list(const std::string& target, const std::vector<T>& device_list, bool exact_match = false) {
            if (exact_match) {
                return std::find_if(device_list.begin(), device_list.end(),
                                    [&target](const T& d) { return d == target; }) != device_list.cend();
            }
            return std::find_if(device_list.begin(), device_list.end(),
                            [&target](const T& d) {
                                return d.find(target) != std::string::npos;
                            }) != device_list.end();
        }
        template <typename T,
          typename std::enable_if<std::is_same<typename std::decay<T>::type, DeviceInformation>::value, bool>::type = true,
          typename U = typename std::vector<T>::const_iterator>
        U check_and_return_if_device_in_list(const std::string& target, const std::vector<T>& device_list, bool exact_match = false) {
            if (exact_match) {
                return std::find_if(device_list.begin(), device_list.end(),
                        [&target](const T& d) { return d.device_name == target; });
            }
            return std::find_if(device_list.begin(), device_list.end(),
                            [&target](const T& d) {
                                return d.device_name.find(target) != std::string::npos;
                            });
        }
        template <typename T,
          typename std::enable_if<std::is_same<typename std::decay<T>::type, DeviceInformation>::value, bool>::type = true>
        bool check_if_device_in_list(const std::string& target, const std::vector<T>& device_list, bool exact_match = false) {
            if (exact_match) {
                return std::find_if(device_list.begin(), device_list.end(),
                                    [&target](const T& d) { return d.device_name == target; }) != device_list.end();
            }
            return std::find_if(device_list.begin(), device_list.end(),
                            [&target](const T& d) {
                                return d.device_name.find(target) != std::string::npos;
                            }) != device_list.end();
        }
};

using NotBusyPriorityWorkerRequests = ov::threading::ThreadSafeBoundedPriorityQueue<std::pair<int, WorkerInferRequest*>>;
using NotBusyWorkerRequests = ov::threading::ThreadSafeBoundedQueue<WorkerInferRequest*>;
using TaskQueue = ov::threading::ThreadSafeQueue<ov::threading::Task>;

template <typename T>
struct IdleGuard {};
template<>
struct IdleGuard<NotBusyWorkerRequests> {
    explicit IdleGuard(WorkerInferRequest* worker_inferrequest_ptr, NotBusyWorkerRequests& not_busy_worker_requests) :
        m_worker_inferrequest_ptr{worker_inferrequest_ptr},
        m_not_busy_worker_requests{&not_busy_worker_requests} {
    }
    ~IdleGuard() {
        if (nullptr != m_not_busy_worker_requests) {
            m_not_busy_worker_requests->try_push(m_worker_inferrequest_ptr);
        }
    }
    NotBusyWorkerRequests* release() {
        auto not_busy_worker_requests = m_not_busy_worker_requests;
        m_not_busy_worker_requests = nullptr;
        return not_busy_worker_requests;
    }
    WorkerInferRequest* m_worker_inferrequest_ptr = nullptr;
    NotBusyWorkerRequests*  m_not_busy_worker_requests = nullptr;
};

template<>
struct IdleGuard<NotBusyPriorityWorkerRequests> {
    explicit IdleGuard(WorkerInferRequest* worker_inferrequest_ptr, NotBusyPriorityWorkerRequests& not_busy_worker_requests) :
        m_worker_inferrequest_ptr{worker_inferrequest_ptr},
        m_not_busy_worker_requests{&not_busy_worker_requests} {
    }
    ~IdleGuard() {
        if (nullptr != m_not_busy_worker_requests) {
            m_not_busy_worker_requests->try_push(std::make_pair(m_worker_inferrequest_ptr->m_index, m_worker_inferrequest_ptr));
        }
    }
    NotBusyPriorityWorkerRequests* release() {
        auto not_busy_worker_requests_queue = m_not_busy_worker_requests;
        m_not_busy_worker_requests = nullptr;
        return not_busy_worker_requests_queue;
    }
    WorkerInferRequest* m_worker_inferrequest_ptr = nullptr;
    NotBusyPriorityWorkerRequests*  m_not_busy_worker_requests = nullptr;
};

class Plugin;
class ScheduleContext : public std::enable_shared_from_this<ScheduleContext>  {
public:
    using Ptr = std::shared_ptr<ScheduleContext>;
    std::shared_ptr<ov::ICore>                     m_ov_core;
    std::weak_ptr<ov::ICompiledModel>              m_compiled_model;
    std::string                                    m_log_tag;
    std::vector<DeviceInformation>                 m_device_priorities;
    std::vector<DeviceInformation>                 m_device_priorities_initial;
    bool                                           m_need_perf_counters;
    bool                                           m_batching_disabled = false;
    bool                                           m_startup_fallback = true;
    bool                                           m_runtime_fallback = true;
    bool                                           m_bind_buffer = false;
    std::shared_ptr<ov::Model>                     m_model;
    std::string                                    m_model_path;
    std::shared_ptr<const ov::IPlugin>             m_plugin;
    std::string                                    m_str_devices;
    unsigned int                                   m_model_priority = 0;
    ov::Any                                        m_performance_hint;
    ov::Any                                        m_schedule_policy = ov::intel_auto::SchedulePolicy::DEFAULT;
    std::mutex                                     m_mutex;
    std::mutex                                     m_fallback_mutex;
    SoCompiledModel                                m_hw_compiled_model;
    std::string                                    m_model_precision;
    // hold the resource of static variable to avoid the unexpected destruction.
    std::shared_ptr<std::mutex>                                          m_mtx;
    std::shared_ptr<std::map<unsigned int, std::list<std::string>>>      m_priority_map;
    std::shared_ptr<Log>                                                 m_logger = Log::instance();
    virtual ~ScheduleContext() = default;
};

struct AutoCompileContext {
    std::atomic<bool> m_is_enabled = {false};
    std::atomic<bool> m_is_already = {false};
    std::atomic<bool> m_is_load_success = {false};
    std::atomic<bool> m_is_reload_success = {false};
    std::future<void> m_future;
    std::promise<void> m_promise;
    SoCompiledModel m_compiled_model;
    DeviceInformation  m_device_info;
    std::vector<DeviceInformation> m_meta_devices;
    std::string m_model_precision;
    std::string m_err_message;
    ov::threading::Task m_task;
    std::string m_worker_name = "";
};

enum AutoCompileContextIndex {
    CPU = 0,
    ACTUALDEVICE = 1,
    FALLBACKDEVICE = 2,
    CONTEXTNUM = 3
};
}  // namespace auto_plugin
} // namespace ov
