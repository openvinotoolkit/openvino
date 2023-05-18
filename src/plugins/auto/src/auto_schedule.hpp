// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "schedule.hpp"
#include "async_infer_request.hpp"

#ifdef  MULTIUNITTEST
#define MOCKTESTMACRO virtual
#define auto_plugin mock_auto_plugin
#else
#define MOCKTESTMACRO
#endif

namespace ov {
namespace auto_plugin {

class AutoSchedule : public Schedule {
public:
    using Ptr = std::shared_ptr<AutoSchedule>;
    virtual ~AutoSchedule();
    AutoLoadContext                                                     m_loadcontext[CONTEXTNUM];

private:
    void init() override;
    // release actual task
    // ov::threading::Task release_actualdevice_task;
    bool schedule_to_worker_inferrequest(ov::threading::Task, DeviceName preferred_device = "") override;
    void wait_actual_network_ready() const;
    /**
     * @brief wait for one of the executable network to finish loading.
     * @return An SoPtr object hold an available executable network loaded to HW device.
     * @note An exception will be thrown if all loading of network to hw device fails.
     */
    SoCompiledModel wait_first_network_ready() override;
    void try_to_load_network(AutoLoadContext& context, const std::shared_ptr<ov::Model>& model) override;
    bool select_other_device(const std::string& cur_dev_name) override;
    size_t                                                               m_cpuhelp_infer_count = 0;
    double                                                               m_cpuhelp_fps = 0.0;
    mutable std::once_flag                                               m_oc;
    std::once_flag                                                       m_firstload_oc;
    std::future<void>                                                    m_firstload_future;
    std::promise<void>                                                   m_firstload_promise;
    bool                                                                 m_exitflag = {false};
};
} // namespace auto_plugin
} // namespace ov
