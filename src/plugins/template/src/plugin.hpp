// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend.hpp"
#include "compiled_model.hpp"
#include "config.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/threading/itask_executor.hpp"

//! [plugin:header]
namespace ov {
namespace template_plugin {

class Plugin : public ov::IPlugin {
public:
    using Ptr = std::shared_ptr<Plugin>;

    Plugin();
    ~Plugin();

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::RemoteContext& context) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::RemoteContext create_context(const ov::AnyMap& remote_properties) const override;

    ov::RemoteContext get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    friend class CompiledModel;
    friend class InferRequest;

    std::shared_ptr<ov::runtime::Backend> _backend;
    Configuration _cfg;
    std::shared_ptr<ov::threading::ITaskExecutor> _waitExecutor;
};

}  // namespace template_plugin
}  // namespace ov
   //! [plugin:header]
