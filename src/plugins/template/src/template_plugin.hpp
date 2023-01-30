// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "backend.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "template_config.hpp"
#include "template_executable_network.hpp"

//! [plugin:header]
namespace TemplatePlugin {

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
    friend class ExecutableNetwork;
    friend class TemplateInferRequest;

    std::shared_ptr<ngraph::runtime::Backend> _backend;
    Configuration _cfg;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};

}  // namespace TemplatePlugin
   //! [plugin:header]
