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

#include "config.hpp"
#include "executable_network.hpp"
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

    ov::Any get_property(const std::string& name, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    friend class HeteroPlugin::HeteroExecutableNetwork;

    ov::Any caching_device_properties(const std::string& device_priorities) const;

    bool device_supports_model_caching(const std::string& device_name) const;

    DeviceProperties get_properties_per_device(const std::string& device_priorities,
                                               const ov::AnyMap& properties) const;

    Configuration m_cfg;
};

}  // namespace hetero
}  // namespace ov