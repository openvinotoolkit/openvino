// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiled_model.h"
#include "cpu_streams_calculation.hpp"

#include <string>
#include <map>
#include <memory>
#include <functional>

namespace ov {
namespace intel_cpu {

class Engine : public ov::IPlugin {
public:
    Engine();
    ~Engine();

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::RemoteContext& context) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "compile_model with RemoteContext is not supported by this plugin!");
    };

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "import_model with RemoteContext is not supported by this plugin!");
    };

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "create_context  is not supported by this plugin!");
    };
    std::shared_ptr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "get_default_context  is not supported by this plugin!");
    };

private:
    bool is_legacy_api() const;

    ov::Any GetMetric(const std::string& name, const ov::AnyMap& options) const;
    ov::Any GetMetricLegacy(const std::string& name, const ov::AnyMap& options) const;

    ov::Any get_property_legacy(const std::string& name, const ov::AnyMap& options) const;
    void ApplyPerformanceHints(ov::AnyMap &config, const std::shared_ptr<ov::Model>& model) const;
    void GetPerformanceStreams(Config &config, const std::shared_ptr<ov::Model>& model) const;
    StreamCfg GetNumStreams(ov::threading::IStreamsExecutor::ThreadBindingType thread_binding_type,
                            int stream_mode,
                            const bool enable_hyper_thread = true) const;

    Config engConfig;
    ExtensionManager::Ptr extensionManager = std::make_shared<ExtensionManager>();
    /* Explicily configured streams have higher priority than performance hints.
       So track if streams is set explicitly (not auto-configured) */
    bool streamsExplicitlySetForEngine = false;
    const std::string deviceFullName;

    std::shared_ptr<void> specialSetup;
};

}   // namespace intel_cpu
}   // namespace ov
