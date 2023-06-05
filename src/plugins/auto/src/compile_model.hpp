// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "schedule.hpp"

namespace ov {
namespace auto_plugin {
class Schedule;
class CompiledModel : public ov::ICompiledModel {
public:
    CompiledModel(const std::shared_ptr<ov::Model>& model,
                  const std::shared_ptr<const ov::IPlugin>& plugin,
                  ScheduleContext::Ptr context,
                  Schedule::Ptr        scheduler);

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<const Plugin> get_auto_plugin();

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    std::string get_log_tag() const noexcept;
    static ov::AnyMap get_device_supported_metrics(AutoLoadContext& context);

private:
    ScheduleContext::Ptr   m_context;
    Schedule::Ptr          m_scheduler;
    std::once_flag         m_oc;
    std::vector<ov::Output<const ov::Node>> m_inputs;
    std::vector<ov::Output<const ov::Node>> m_outputs;
    void set_compilemodel_for_context();
};
}  // namespace auto_plugin
} // namespace ov
