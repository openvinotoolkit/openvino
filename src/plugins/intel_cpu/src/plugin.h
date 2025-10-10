// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <istream>
#include <memory>
#include <string>

#include "config.h"
#include "openvino/core/any.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/model.hpp"
#include "openvino/runtime/common.hpp"
#include "openvino/runtime/icompiled_model.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/iremote_context.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "openvino/runtime/threading/cpu_message.hpp"
#include "utils/graph_serializer/deserializer.hpp"

namespace ov::intel_cpu {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin() override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(
        [[maybe_unused]] const std::shared_ptr<const ov::Model>& model,
        [[maybe_unused]] const ov::AnyMap& properties,
        [[maybe_unused]] const ov::SoPtr<ov::IRemoteContext>& context) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("compile_model with RemoteContext is not supported by CPU plugin!");
    };

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model([[maybe_unused]] const ov::Tensor& model,
                                                     [[maybe_unused]] const ov::SoPtr<ov::IRemoteContext>& context,
                                                     [[maybe_unused]] const ov::AnyMap& properties) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("import_model with RemoteContext is not supported by CPU plugin!");
    };
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model([[maybe_unused]] std::istream& model,
                                                     [[maybe_unused]] const ov::SoPtr<ov::IRemoteContext>& context,
                                                     [[maybe_unused]] const ov::AnyMap& properties) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("import_model with RemoteContext is not supported by CPU plugin!");
    };

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
    ov::SoPtr<ov::IRemoteContext> create_context([[maybe_unused]] const ov::AnyMap& remote_properties) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("create_context is not supported by CPU plugin!");
    };
    ov::SoPtr<ov::IRemoteContext> get_default_context(
        [[maybe_unused]] const ov::AnyMap& remote_properties) const override {
        OPENVINO_THROW_NOT_IMPLEMENTED("get_default_context is not supported by CPU plugin!");
    };

    std::shared_ptr<ov::threading::MessageManager> m_msg_manager;

private:
    std::shared_ptr<ov::ICompiledModel> deserialize_model(ModelDeserializer& deserializer,
                                                          const ov::AnyMap& config) const;

    ov::Any get_ro_property(const std::string& name, const ov::AnyMap& options) const;

    static void get_performance_streams(Config& config, const std::shared_ptr<ov::Model>& model);
    static void calculate_streams(Config& conf, const std::shared_ptr<ov::Model>& model, bool imported = false);
    Config engConfig;
    /* Explicily configured streams have higher priority than performance hints.
       So track if streams is set explicitly (not auto-configured) */
    bool streamsExplicitlySetForEngine = false;
    const std::string deviceFullName;
    ov::AnyMap m_compiled_model_runtime_properties;

    std::shared_ptr<void> specialSetup;
};

}  // namespace ov::intel_cpu
