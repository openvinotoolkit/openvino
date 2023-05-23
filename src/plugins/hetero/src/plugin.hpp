// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/runtime/iplugin.hpp"
#include "config.hpp"

#include "cpp_interfaces/interface/ie_iplugin_internal.hpp"
#include "description_buffer.hpp"
#include "ie_icore.hpp"

namespace ov {
namespace hetero {

using Configs = std::map<std::string, std::string>;

template <typename T>
struct ParsedConfig {
    Configs hetero_config;
    T device_config;
};

class Plugin : public ov::IPlugin {
public:
    using DeviceMetaInformationMap = std::unordered_map<std::string, ov::AnyMap>;

    Plugin();
    ~Plugin() = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::RemoteContext& context) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    std::shared_ptr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
    

    

    // TODO vurusovs: THINK TWICE ABOUT NECESSITY OF FUNCTION
    DeviceMetaInformationMap GetDevicePlugins(const std::string& targetFallback, const ov::AnyMap& properties) const;

    // std::string GetTargetFallback(const Configs& config, bool raise_exception = true) const;
    // std::string GetTargetFallback(const ov::AnyMap& config, bool raise_exception = true) const;

    // FROM OLD HETERO PLUGIN

private:
    friend class CompiledModel;
    friend class InferRequest;

    Configuration m_cfg;

    // FROM TEMPLATE PLUGIN
    // std::shared_ptr<ov::runtime::Backend> m_backend;
    
    // std::shared_ptr<ov::threading::ITaskExecutor> m_waitExecutor;


    std::string DeviceCachingProperties(const std::string& targetFallback) const;

    Configs _device_config;
};

}  // namespace hetero
}  // namespace ov
