// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "schedule.hpp"
#include "async_infer_request.hpp"

namespace ov {
namespace auto_plugin {

class AutoSchedule : public Schedule {
public:
    using Ptr = std::shared_ptr<AutoSchedule>;
    AutoSchedule(std::shared_ptr<std::mutex>& mtx,
                 std::shared_ptr<std::map<unsigned int, std::list<std::string>>>& priority_map)
        : Schedule(),
          m_mtx(mtx),
          m_priority_map(priority_map) {}
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
    size_t                                                               m_cpuhelp_infer_count = 0;
    double                                                               m_cpuhelp_fps = 0.0;
    mutable std::once_flag                                               m_oc;
    std::once_flag                                                       m_firstload_oc;
    std::future<void>                                                    m_firstload_future;
    std::promise<void>                                                   m_firstload_promise;
    bool                                                                 m_exitflag = {false};
    // hold the resource of static variable to avoid the destruction of the static variable.
    std::shared_ptr<std::mutex>                                          m_mtx;
    std::shared_ptr<std::map<unsigned int, std::list<std::string>>>      m_priority_map;
};
} // namespace auto_plugin
} // namespace ov
