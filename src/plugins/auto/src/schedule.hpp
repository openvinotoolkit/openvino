// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "common.hpp"
#include "device_execution_context.hpp"
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
    // the context whose compiled model answers the model queries right now, null if none holds one;
    // the model may be released at any time, so the caller takes a copy through compiled_model()
    virtual DeviceExecutionContext* get_active_context() = 0;
    static thread_local WorkerInferRequest* m_this_worker_infer_request;
    // have to use the const char* ptr rather than std::string due to a bug in old gcc versions,
    // the bug is e.g. manifesting on the old CentOS (and it's 4.8.x gcc) used in our testing
    // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81880
    static thread_local const char*         m_this_preferred_device_name;
    // set when the scheduler cannot dispatch a request, rethrown by the scheduling stage of the pipeline
    static thread_local std::exception_ptr  m_this_scheduling_exception;

protected:
    virtual void init() = 0;
    virtual void release_execution_slot() {}
    static bool run_pipeline_task(ov::threading::Task& pipeline_task, NotBusyPriorityWorkerRequests& idle_worker_request,
                                  const DeviceName& preferred_device);
    virtual void generate_workers(const std::string& device_name, const SoCompiledModel& compiled_model);
    virtual void try_to_compile_model(AutoCompileContext& context, const std::shared_ptr<ov::Model>& model) = 0;
    virtual bool schedule_to_worker_infer_request(ov::threading::Task, DeviceName preferred_device = "") = 0;
    virtual bool select_other_device(const std::string& cur_dev_name) = 0;
    virtual SoCompiledModel wait_first_compiled_model_ready() = 0;
    std::string get_log_tag() const noexcept;
    // keeps the ports and the remote context of the first compiled model, the model itself is not kept
    void capture_hw_metadata(const SoCompiledModel& first_compiled_model);
    // add_device_context() is idempotent and must not be called once the scheduler serves requests
    DeviceExecutionContext& add_device_context(const DeviceName& name);
    // asserts if the device was not added by init()
    DeviceExecutionContext& get_device_context(const DeviceName& name);
    DeviceExecutionContext* find_device_context(const DeviceName& name);
    // moves a successful compilation into the context named by m_worker_name (it differs from the device name only
    // for CPU_HELP): creates the workers, records the compile config, and the context becomes the owner of the model
    void register_compile_result(AutoCompileContext& context);
    // records a failed compilation in the context of the device the compile context targets
    void mark_compile_failure(const AutoCompileContext& context, const std::string& reason);
    std::shared_ptr<ov::threading::IStreamsExecutor>                     m_executor;
    TaskQueue                                                            m_infer_pipeline_tasks;
    // created by init() only, the map is never modified afterwards so the contexts keep their address
    DeviceMap<std::unique_ptr<DeviceExecutionContext>>                   m_device_contexts;
    // the context holding the compiled model that serves every request directly, null if requests are scheduled
    DeviceExecutionContext*                                              m_passthrough_context = nullptr;
    // set before compiling: the model is only held by its context, no worker is created for it
    bool                                                                 m_model_only = false;
    ScheduleContext::Ptr                                                 m_context;
    std::shared_ptr<Plugin>                                              m_plugin;
    std::string                                                          m_log_tag;
    Time                                                                 m_cpuhelp_release_time;
    mutable std::atomic<std::size_t>                                     m_request_id = {0};
    std::mutex                                                           m_dev_infer_mutex;
    // dynamic selection only: written before the execution slot is released, read by the next dispatch
    double                                                               m_last_infer_ms = -1.0;
    std::unordered_map<IASyncInferPtr, WorkerInferRequest*>              m_dev_infer;
};

}  // namespace auto_plugin
}  // namespace ov
