// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "common.hpp"
#include "itt.hpp"

namespace ov {
namespace auto_plugin {
using Stage = std::pair<std::shared_ptr<ov::threading::ITaskExecutor>, ov::threading::Task>;
using Pipeline = std::vector<Stage>;

class Schedule : public std::enable_shared_from_this<Schedule>, public ov::threading::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<Schedule>;
    virtual void launch(const ScheduleContext::Ptr& context);
    virtual Pipeline get_async_pipeline(const ISyncInferPtr& sync_request, WorkerInferRequest** Worker_infer_request);
    void run(ov::threading::Task infer_task) override;
    virtual ~Schedule();
    virtual ISyncInferPtr create_sync_infer_request();
    static thread_local WorkerInferRequest* m_this_worker_infer_request;
    // have to use the const char* ptr rather than std::string due to a bug in old gcc versions,
    // the bug is e.g. manifesting on the old CentOS (and it's 4.8.x gcc) used in our testing
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81880
    static thread_local const char*         m_this_preferred_device_name;

protected:
    virtual void init() = 0;
    static bool run_pipeline_task(ov::threading::Task& pipeline_task, NotBusyPriorityWorkerRequests& idle_worker_request,
                                  const DeviceName& preferred_device);
    virtual void generate_workers(const std::string& device, const SoCompiledModel& compiled_model);
    virtual void try_to_compile_model(AutoCompileContext& context, const std::shared_ptr<ov::Model>& model) = 0;
    virtual bool schedule_to_worker_infer_request(ov::threading::Task, DeviceName preferred_device = "") = 0;
    virtual bool select_other_device(const std::string& cur_dev_name) = 0;
    virtual SoCompiledModel wait_first_compiled_model_ready() = 0;
    std::string get_log_tag() const noexcept;
    std::shared_ptr<ov::threading::IStreamsExecutor>                     m_executor;
    DeviceMap<NotBusyPriorityWorkerRequests>                             m_idle_worker_requests;
    DeviceMap<std::vector<WorkerInferRequest>>                           m_worker_requests;
    TaskQueue                                                            m_infer_pipeline_tasks;
    DeviceMap<std::unique_ptr<TaskQueue>>                                m_infer_pipeline_tasks_device_specific;
    SoCompiledModel                                                      m_passthrough_compiled_model;
    ScheduleContext::Ptr                                                 m_context;
    std::shared_ptr<Plugin>                                              m_plugin;
    std::string                                                          m_log_tag;
    Time                                                                 m_cpuhelp_release_time;
    mutable std::atomic<std::size_t>                                     m_request_id = {0};
    std::mutex                                                           m_dev_infer_mutex;
    std::unordered_map<IASyncInferPtr, WorkerInferRequest*>              m_dev_infer;
};

}  // namespace auto_plugin
}  // namespace ov
