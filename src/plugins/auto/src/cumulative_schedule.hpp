// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "schedule.hpp"
#include "async_infer_request.hpp"

namespace ov {
namespace auto_plugin {

class CumuSchedule : public Schedule {
public:
    using Ptr = std::shared_ptr<CumuSchedule>;
    virtual ~CumuSchedule();
    std::unique_ptr<AutoLoadContext[]>      m_p_ctput_loadcontext = nullptr;
    size_t                                  m_n_ctput_devicenums = 0;

private:
    void init() override;
    SoCompiledModel wait_first_network_ready() override;
    bool schedule_to_worker_inferrequest(ov::threading::Task, DeviceName preferred_device = "") override;
    void try_to_load_network(AutoLoadContext& context, const std::shared_ptr<ov::Model>& model) override;
    bool select_other_device(const std::string& cur_dev_name) override;
};
} // namespace auto_plugin
} // namespace ov
