// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "compiled_model.hpp"
#include "cumulative_schedule.hpp"

namespace ov {
namespace auto_plugin {

class AutoCumuCompiledModel : public CompiledModel {
public:
    AutoCumuCompiledModel(const std::shared_ptr<ov::Model>& model,
                          const std::shared_ptr<const ov::IPlugin>& plugin,
                          const ov::SoPtr<ov::IRemoteContext>& remote_context,
                          ScheduleContext::Ptr& schedule_context,
                          Schedule::Ptr& scheduler);

    // implement pure virtual methods from a base class ov::ICompiledModel
    void export_model(std::ostream& model) const override;

    std::shared_ptr<const ov::Model> get_runtime_model() const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name) const override;

private:
    friend class InferRequest;
    friend class Plugin;
    CumuSchedule::Ptr          m_scheduler;
};
}  // namespace auto_plugin
} // namespace ov
