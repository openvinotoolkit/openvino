// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <deque>

#include "schedule.hpp"
#include "async_infer_request.hpp"

namespace ov {
namespace auto_plugin {

class AutoSchedule : public Schedule {
public:
    using Ptr = std::shared_ptr<AutoSchedule>;
    virtual ~AutoSchedule();
    AutoCompileContext                                                     m_compile_context[CONTEXTNUM];

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
    void try_to_compile_model(AutoCompileContext& context, const std::shared_ptr<ov::Model>& model) override;
    bool select_other_device(const std::string& cur_dev_name) override;
    void release_execution_slot() override;
    /**
     * @brief Serialize the incoming request behind the execution gate when per inference device selection is on.
     * @return true if the request was handed over for dispatching, false if it was queued.
     */
    bool schedule_dynamic_task(ov::threading::Task pipeline_task, const DeviceName& preferred_device);
    void dispatch_dynamic_task(ov::threading::Task pipeline_task, const DeviceName& preferred_device);
    DeviceInformation select_dynamic_device();
    /**
     * @brief Compile the model on the given device unless it is already cached, and create its workers.
     * @note device is updated in place when the compilation falls back to another candidate device.
     */
    bool ensure_device_ready(DeviceInformation& device);
    size_t                                                               m_cpuhelp_infer_count = 0;
    double                                                               m_cpuhelp_fps = 0.0;
    mutable std::once_flag                                               m_oc;
    std::once_flag                                                       m_firstload_oc;
    std::future<void>                                                    m_firstload_future;
    std::promise<void>                                                   m_firstload_promise;
    bool                                                                 m_exitflag = {false};
    std::shared_ptr<ov::threading::IStreamsExecutor>                     m_dynamic_executor;
    std::shared_ptr<ov::Model>                                           m_dynamic_model;
    std::mutex                                                           m_gate_mutex;
    bool                                                                 m_gate_busy = false;
    DeviceName                                                           m_gate_current_device;
    std::deque<std::pair<ov::threading::Task, DeviceName>>               m_gate_pending_tasks;
    DeviceMap<SoCompiledModel>                                           m_dynamic_compiled_models;
    std::atomic<size_t>                                                  m_dynamic_infer_count = {0};
    std::atomic<size_t>                                                  m_dynamic_switch_count = {0};
};
} // namespace auto_plugin
} // namespace ov
