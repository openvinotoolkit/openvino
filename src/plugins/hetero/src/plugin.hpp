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

#include "../executable_network.hpp"
#include "config.hpp"
#include "openvino/runtime/iplugin.hpp"

namespace HeteroPlugin {
class HeteroExecutableNetwork;
}

namespace ov {
namespace hetero {

class Plugin : public ov::IPlugin {
public:
    using DeviceProperties = std::unordered_map<std::string, ov::AnyMap>;

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

    // FROM OLD HETERO PLUGIN
    // std::string GetTargetFallback(const Configs& config, bool raise_exception = true) const;
    // std::string GetTargetFallback(const ov::AnyMap& config, bool raise_exception = true) const;

private:
    // friend class CompiledModel;
    friend class HeteroPlugin::HeteroExecutableNetwork;
    // friend class InferRequest;

    std::string DeviceCachingProperties(const std::string& targetFallback) const;

    DeviceProperties get_device_properties(const std::string& device_priorities, const ov::AnyMap& properties) const;

    Configuration m_cfg;
};

}  // namespace hetero
}  // namespace ov