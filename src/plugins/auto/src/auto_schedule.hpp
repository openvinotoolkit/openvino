// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <condition_variable>
#include <deque>
#include <map>

#include "adaptive/decision_log.hpp"
#include "adaptive/headroom.hpp"
#include "adaptive/signal_hub.hpp"
#include "adaptive/switch_policy.hpp"
#include "async_infer_request.hpp"
#include "cache_manager.hpp"
#include "schedule.hpp"
#include "schedule_policies.hpp"

namespace ov {
namespace auto_plugin {

class AutoSchedule : public Schedule {
public:
    using Ptr = std::shared_ptr<AutoSchedule>;
    virtual ~AutoSchedule();
    RoleSlot                                                               m_compile_context[CONTEXTNUM];

private:
    void init() override;
    // release actual task
    // ov::threading::Task release_actualdevice_task;
    bool schedule_to_worker_infer_request(ov::threading::Task, DeviceName preferred_device = "") override;
    void wait_actual_compiled_model_ready() const;
    /**
     * @brief wait for one of the compiled model to finish loading.
     * @return An SoPtr object hold an available compiled model loaded to HW device.
     * @note An exception will be thrown if all loading of model to hw device fails.
     */
    SoCompiledModel wait_first_compiled_model_ready() override;
    CompileResult try_to_compile_model(CompileRequest& request, const std::shared_ptr<ov::Model>& model) override;
    bool select_other_device(const ModelInstanceKey& failed_instance) override;
    void release_execution_slot() override;
    /**
     * @brief Serialize the incoming request behind the execution gate when per inference device selection is on.
     * @return true if the request was handed over for dispatching, false if it was queued.
     */
    bool schedule_dynamic_task(ov::threading::Task pipeline_task, const DeviceName& preferred_device);
    void dispatch_dynamic_task(ov::threading::Task& pipeline_task, const DeviceName& preferred_device);
    // runs dispatch_dynamic_task() on the executor;
    // the caller must have incremented m_active_dispatch under m_gate_mutex
    void submit_dynamic_dispatch(const std::shared_ptr<ov::threading::IStreamsExecutor>& executor,
                                 ov::threading::Task pipeline_task,
                                 const DeviceName& preferred_device);
    void end_dynamic_dispatch();
    DeviceInformation select_dynamic_device();
    // keeps the current device when the policy sees no reason to leave it for the proposed one
    DeviceInformation apply_switch_policy(DeviceInformation proposed);
    /**
     * @brief Compile the model on the given device unless it is already cached, and create its workers.
     * @note device is updated in place when the compilation falls back to another candidate device.
     */
    bool ensure_device_ready(DeviceInformation& device);
    /**
     * @brief Drop the compiled model and workers of every device that is no longer the active one.
     * @note Only done with compile_for_all. The active device is already dispatched when this runs, so devices are only
     * released while the active one is loaded. A device that could not be released is tried again at the next dispatch.
     */
    void release_inactive_devices(const DeviceName& active_device);
    // whether the device has a cache blob to be recompiled from; only used on the dynamic dispatch thread
    bool is_reloadable(const DeviceName& device_name);
    // the candidates the failure policy lets the dynamic selection try now; the caller holds m_fallback_mutex
    std::vector<DeviceInformation> available_candidates() const;
    // backs the devices off, and excludes the ones that failed too often for good
    void handle_compile_failures(const std::vector<DeviceName>& devices);
    // disables the configured low_power_device override if it matches the failed device
    // @return true only for the call that disabled it, i.e. the first failure of that device
    bool check_and_disable_if_low_power_device_failed(const std::string& failed_device_name) const;
    // publishes the instance that serves the requests in the static mode: a fallback device replaces the actual
    // device it was selected for, and the actual device replaces the startup helper
    void publish_active_instance();
    size_t                                                               m_cpuhelp_infer_count = 0;
    double                                                               m_cpuhelp_fps = 0.0;
    mutable std::once_flag                                               m_oc;
    std::once_flag                                                       m_firstload_oc;
    std::future<void>                                                    m_firstload_future;
    std::promise<void>                                                   m_firstload_promise;
    bool                                                                 m_exitflag = {false};
    std::shared_ptr<ov::threading::IStreamsExecutor>                     m_dynamic_executor;
    std::shared_ptr<ov::Model>                                           m_dynamic_model;
    // the members below belong to the dynamic dispatch thread
    // null keeps every compiled model
    std::optional<ReloadablePolicy>                                      m_eviction_policy;
    FailurePolicy                                                        m_failure_policy;
    mutable std::mutex                                                   m_gate_mutex;
    bool                                                                 m_gate_busy = false;
    // guarded by m_gate_mutex; set once in ~AutoSchedule() so in-flight dynamic tasks stop touching
    // m_dynamic_executor once teardown has started
    bool                                                                 m_dynamic_shutdown = false;
    // guarded by m_gate_mutex; dispatch tasks on m_dynamic_executor still running and holding a raw this
    size_t                                                               m_active_dispatch = 0;
    std::condition_variable                                              m_dispatch_done;
    DeviceName                                                           m_gate_current_device;
    std::deque<std::pair<ov::threading::Task, DeviceName>>               m_gate_pending_tasks;
    DecisionLog                                                          m_decision_log;
    SwitchPolicy m_switch_policy;
    // guarded by m_gate_mutex: since when the current device serves
    Time m_current_since{};
    // keeps the utilization of the devices fresh in the background, only with the adaptive level set
    std::shared_ptr<SignalHub::Subscription> m_telemetry_subscription;
};
} // namespace auto_plugin
} // namespace ov
