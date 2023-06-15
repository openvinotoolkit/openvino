// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "common.hpp"
namespace ov {
namespace auto_plugin {
using Stage = std::pair<std::shared_ptr<ov::threading::ITaskExecutor>, ov::threading::Task>;
using Pipeline = std::vector<Stage>;

class Schedule : public std::enable_shared_from_this<Schedule>, public ov::threading::ITaskExecutor {
public:
    using Ptr = std::shared_ptr<Schedule>;
    virtual void launch(ScheduleContext::Ptr context);
    virtual Pipeline get_async_pipeline(const ISyncInferPtr& syncRequestImpl, WorkerInferRequest** WorkerInferRequest);
    void run(ov::threading::Task infer_task) override;
    virtual ~Schedule();
    virtual ISyncInferPtr create_sync_infer_request();
    static thread_local WorkerInferRequest* m_this_worker_inferrequest;
    // have to use the const char* ptr rather than std::string due to a bug in old gcc versions,
    // the bug is e.g. manifesting on the old CentOS (and it's 4.8.x gcc) used in our testing
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81880
    static thread_local const char*         m_this_preferred_devicename;

protected:
    virtual void init() = 0;
    static bool run_pipeline_task(ov::threading::Task& pipeline_task, NotBusyPriorityWorkerRequests& idle_worker_request,
                                  const DeviceName& preferred_device);
    virtual void generate_workers(const std::string& device, const SoCompiledModel& executable_network);
    virtual void try_to_load_network(AutoLoadContext& context, const std::shared_ptr<ov::Model>& model) = 0;
    virtual bool schedule_to_worker_inferrequest(ov::threading::Task, DeviceName preferred_device = "") = 0;
    virtual bool select_other_device(const std::string& cur_dev_name) = 0;
    virtual SoCompiledModel wait_first_network_ready() = 0;
    std::string get_log_tag() const noexcept;
    std::shared_ptr<ov::threading::IStreamsExecutor>                     m_executor;
    DeviceMap<NotBusyPriorityWorkerRequests>                             m_idle_workerrequests;
    DeviceMap<std::vector<WorkerInferRequest>>                           m_workerrequests;
    TaskQueue                                                            m_infer_pipelinetasks;
    DeviceMap<std::unique_ptr<TaskQueue>>                                m_infer_pipelinetasks_devicespecific;
    SoCompiledModel                                                      m_passthrough_exenet;
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
