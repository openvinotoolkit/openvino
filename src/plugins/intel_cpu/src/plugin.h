// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "compiled_model.h"
#include "cpu_streams_calculation.hpp"

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
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "compile_model with RemoteContext is not supported by CPU plugin!");
    };

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "import_model with RemoteContext is not supported by CPU plugin!");
    };

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "create_context  is not supported by CPU plugin!");
    };
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override {
        OPENVINO_ASSERT_HELPER(::ov::NotImplemented,
                               "",
                               false,
                               "Not Implemented",
                               "get_default_context  is not supported by CPU plugin!");
    };

private:
    bool is_legacy_api() const;

    ov::Any get_ro_property(const std::string& name, const ov::AnyMap& options) const;
    ov::Any get_metric_legacy(const std::string& name, const ov::AnyMap& options) const;

    ov::Any get_property_legacy(const std::string& name, const ov::AnyMap& options) const;
    void apply_performance_hints(ov::AnyMap &config, const std::shared_ptr<ov::Model>& model) const;
    void get_performance_streams(Config &config, const std::shared_ptr<ov::Model>& model) const;
    StreamCfg get_streams_num(ov::threading::IStreamsExecutor::ThreadBindingType thread_binding_type,
                              int stream_mode,
                              const bool enable_hyper_thread = true) const;
    void calculate_streams(Config& conf, const std::shared_ptr<ov::Model>& model, bool imported = false) const;

    Config engConfig;
    ExtensionManager::Ptr extensionManager = std::make_shared<ExtensionManager>();
    /* Explicily configured streams have higher priority than performance hints.
       So track if streams is set explicitly (not auto-configured) */
    bool streamsExplicitlySetForEngine = false;
    const std::string deviceFullName;

    std::shared_ptr<void> specialSetup;

#if defined(OV_CPU_WITH_ACL)
    struct SchedulerGuard {
        SchedulerGuard();
        ~SchedulerGuard();
        static std::shared_ptr<SchedulerGuard> instance();
        static std::mutex mutex;
        // separate mutex for saving ACLScheduler state in destructor
        mutable std::mutex dest_mutex;
        static std::weak_ptr<SchedulerGuard> ptr;
    };

    std::shared_ptr<SchedulerGuard> scheduler_guard;
#endif
};

}   // namespace intel_cpu
}   // namespace ov
