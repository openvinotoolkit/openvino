// Copyright (C) 2018-2025 Intel Corporation
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
                  const ov::SoPtr<ov::IRemoteContext>& remote_context,
                  ScheduleContext::Ptr& schedule_context,
                  Schedule::Ptr&        scheduler);

    std::shared_ptr<ov::IAsyncInferRequest> create_infer_request() const override;
    std::shared_ptr<const Plugin> get_auto_plugin();
    const std::vector<ov::Output<const ov::Node>>& outputs() const override;
    const std::vector<ov::Output<const ov::Node>>& inputs() const override;

protected:
    std::shared_ptr<ov::ISyncInferRequest> create_sync_infer_request() const override;
    std::string get_log_tag() const noexcept;
    static ov::AnyMap get_device_supported_properties(AutoCompileContext& context);
    ScheduleContext::Ptr   m_context;

private:
    Schedule::Ptr          m_scheduler;
    std::once_flag         m_oc;
    bool m_inputs_outputs_from_hardware;
    void set_compile_model_for_context();
};
}  // namespace auto_plugin
} // namespace ov
