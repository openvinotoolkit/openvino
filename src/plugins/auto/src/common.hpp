// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>
#include <string>
#include "ie_icore.hpp"
#include "ie_metric_helpers.hpp"
#include <ie_plugin_config.hpp>
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"
#include "threading/ie_thread_safe_containers.hpp"
#include "utils/log_util.hpp"
#include <ie_performance_hints.hpp>
#include "openvino/runtime/auto/properties.hpp"
#include "ngraph/opsets/opset1.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/log_util.hpp"
#include "itt.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define auto_plugin mock_auto_plugin
#else
#define MOCKTESTMACRO
#endif

namespace ov {
namespace auto_plugin {
namespace IE = InferenceEngine;
using DeviceName = std::string;
using IInferPtr = std::shared_ptr<ov::ISyncInferRequest>;
using IExecNetwork = std::shared_ptr<ov::ICompiledModel>;
using SoInfer = ov::SoPtr<ov::ISyncInferRequest>;
using SoExecNetwork = ov::SoPtr<ov::ICompiledModel>;
using Time = std::chrono::time_point<std::chrono::steady_clock>;

template<typename T>
using device_map = std::unordered_map<devicename, T>;
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

struct DeviceInformation {
    DeviceName device_name;
    ov::AnyMap config;
    int num_requests_per_devices;
    std::string default_device_id;
    DeviceName unique_name;
    unsigned int device_priority;
    DeviceInformation(DeviceName dn = {}, ov::AnyMap conf = {},
        int nReq = -1, std::string defaultID = {}, DeviceName uName = {}, unsigned int priority = 0)
        : device_name(dn), config(conf),
        num_requests_per_devices(nReq), default_device_id(defaultID), unique_name(uName), device_priority(priority)
        {}
};

struct WorkerInferRequest {
    SoInfer                       m_inferrequest;
    ov::threading::Task           m_task;
    std::exception_ptr            m_exception_ptr = nullptr;
    std::list<Time>               m_start_times;
    std::list<Time>               m_end_times;
    int                           m_index = 0;
    AutoImmediateExecutor::Ptr    m_fallback_exec;
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
                        [&target](const T& d) { return d.deviceName == target; });
            }
            return std::find_if(device_list.begin(), device_list.end(),
                            [&target](const T& d) {
                                return d.deviceName.find(target) != std::string::npos;
                            });
        }
        template <typename T,
          typename std::enable_if<std::is_same<typename std::decay<T>::type, DeviceInformation>::value, bool>::type = true>
        bool check_if_device_in_list(const std::string& target, const std::vector<T>& device_list, bool exact_match = false) {
            if (exact_match) {
                return std::find_if(device_list.begin(), device_list.end(),
                                    [&target](const T& d) { return d.deviceName == target; }) != device_list.end();
            }
            return std::find_if(device_list.begin(), device_list.end(),
                            [&target](const T& d) {
                                return d.deviceName.find(target) != std::string::npos;
                            }) != device_list.end();
        }
};

using NotBusyPriorityWorkerRequests = IE::ThreadSafeBoundedPriorityQueue<std::pair<int, WorkerInferRequest*>>;
using NotBusyWorkerRequests = IE::ThreadSafeBoundedQueue<WorkerInferRequest*>;
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
    NotBusyWorkerRequests* Release() {
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
    NotBusyPriorityWorkerRequests* Release() {
        auto not_busy_worker_requests_queue = m_not_busy_worker_requests;
        m_not_busy_worker_requests = nullptr;
        return not_busy_worker_requests_queue;
    }
    WorkerInferRequest* m_worker_inferrequest_ptr = nullptr;
    NotBusyPriorityWorkerRequests*  m_not_busy_worker_requests = nullptr;
};
class ScheduleContext : public std::enable_shared_from_this<ScheduleContext> {
public:
    using Ptr = std::shared_ptr<ScheduleContext>;
    std::shared_ptr<ov::ICore>  m_ov_core;
    std::weak_ptr<IExecNetwork> m_compiled_model;
    std::string m_log_tag;
    virtual ~ScheduleContext() = default;
};

class Plugin;
class AutoScheduleContext : public ScheduleContext {
public:
    using Ptr = std::shared_ptr<AutoScheduleContext>;
    std::vector<DeviceInformation>                 m_device_priorities;
    std::vector<DeviceInformation>                 m_device_priorities_initial;
    std::unordered_map<std::string, ov::Any>       m_config;
    bool                                           m_need_perf_counters;
    bool                                           m_batching_disabled = {false};
    bool                                           m_startup_fallback = true;
    bool                                           m_runtime_fallback = true;
    std::string                                    m_modelpath;
    IE::CNNNetwork                                 m_network;
    std::string                                    m_str_devices;
    unsigned int                                   m_model_priority = 0;
    std::string                                    m_performance_hint;
    std::mutex                                     m_conf_mutex;
    std::mutex                                     m_fallback_mutex;
    ov::auto_plugin::Plugin*                       m_plugin;
    SoExecNetwork                                  m_hw_compiled_model;
    virtual ~AutoScheduleContext() = default;
};
}  // namespace auto_plugin
} // namespace ov
