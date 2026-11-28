// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "adaptive/instance_stats.hpp"
#include "common.hpp"
#include "compile_service.hpp"
#include "instance_registry.hpp"
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
    // the published instance that answers the model queries, read without any lock of the scheduler
    ActiveDevice get_active_device() const;
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
    // runs the task on an idle worker of the pool, which stays leased until the inference completes
    static bool run_pipeline_task(ov::threading::Task& pipeline_task, WorkerPool& pool);
    virtual void generate_workers(const ModelInstanceKey& key, const SoCompiledModel& compiled_model);
    // the result is empty if the compilation failed or the context reuses a ready device
    virtual CompileResult try_to_compile_model(CompileRequest& request, const std::shared_ptr<ov::Model>& model) = 0;
    virtual bool schedule_to_worker_infer_request(ov::threading::Task, DeviceName preferred_device = "") = 0;
    // picks another device after an inference failed on the given instance
    virtual bool select_other_device(const ModelInstanceKey& failed_instance) = 0;
    virtual SoCompiledModel wait_first_compiled_model_ready() = 0;
    std::string get_log_tag() const noexcept;
    // keeps the ports of the first compiled model and the default remote context of its device, not the model
    void capture_hw_metadata(const SoCompiledModel& first_compiled_model);
    // add_device_context() is idempotent and must not be called once the scheduler serves requests
    DeviceExecutionContext& add_device_context(const ModelInstanceKey& key);
    // asserts if the instance was not added by init()
    DeviceExecutionContext& get_device_context(const ModelInstanceKey& key);
    DeviceExecutionContext* find_device_context(const ModelInstanceKey& key);
    // moves a successful compilation into the context of request.instance_key(): creates the workers, and the
    // context becomes the owner of the model
    DeviceExecutionContext& register_compile_result(const CompileRequest& request, SoCompiledModel compiled_model);
    // records a failed compilation in the context of the device the compile context targets
    void mark_compile_failure(const CompileRequest& request, const std::string& reason);
    std::shared_ptr<ov::threading::IStreamsExecutor>                     m_executor;
    TaskQueue                                                            m_infer_pipeline_tasks;
    // filled by init() only, the contexts keep their address until the schedule is destroyed
    InstanceRegistry                                                     m_instances;
    // created by launch() before init(), every compilation of the schedule goes through it
    std::unique_ptr<CompileService>                                      m_compile_service;
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
    // dynamic selection only: written before the execution slot is released
    InstanceStats                                                        m_instance_stats;
    std::unordered_map<IASyncInferPtr, WorkerInferRequest*>              m_dev_infer;
};

}  // namespace auto_plugin
}  // namespace ov
