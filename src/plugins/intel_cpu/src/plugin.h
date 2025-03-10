// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiled_model.h"
#include "openvino/runtime/threading/cpu_message.hpp"

namespace ov {
namespace intel_cpu {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin();

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("compile_model with RemoteContext is not supported by CPU plugin!");
    };

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("import_model with RemoteContext is not supported by CPU plugin!");
    };

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("create_context is not supported by CPU plugin!");
    };
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("get_default_context is not supported by CPU plugin!");
    };

    std::shared_ptr<ov::threading::MessageManager> m_msg_manager;

private:
    ov::Any get_ro_property(const std::string& name, const ov::AnyMap& options) const;

    void get_performance_streams(Config& config, const std::shared_ptr<ov::Model>& model) const;
    void calculate_streams(Config& conf, const std::shared_ptr<ov::Model>& model, bool imported = false) const;
    Config engConfig;
    /* Explicily configured streams have higher priority than performance hints.
       So track if streams is set explicitly (not auto-configured) */
    bool streamsExplicitlySetForEngine = false;
    const std::string deviceFullName;
    ov::AnyMap m_compiled_model_runtime_properties;

    std::shared_ptr<void> specialSetup;
};

}  // namespace intel_cpu
}  // namespace ov
