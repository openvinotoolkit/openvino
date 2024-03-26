// Copyright (C) 2018-2024 Intel Corporation
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
#include "openvino/runtime/iplugin.hpp"
#include "subgraph_collector.hpp"

namespace ov {
namespace hetero {

class CompiledModel;

class Plugin : public ov::IPlugin {
public:
    using DeviceProperties = std::unordered_map<std::string, ov::AnyMap>;

    Plugin();

    ~Plugin() = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& properties) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    friend class CompiledModel;

    ov::Any caching_device_properties(const std::string& device_priorities) const;

    DeviceProperties get_properties_per_device(const std::string& device_priorities,
                                               const ov::AnyMap& properties) const;

    void get_device_memory_map(const std::vector<std::string>& device_names,
                               std::map<std::string, size_t>& device_mem_map) const;

    std::pair<ov::SupportedOpsMap, ov::hetero::SubgraphsMappingInfo> query_model_update(
        std::shared_ptr<ov::Model>& model,
        const ov::AnyMap& properties,
        bool allow_exception = false) const;

    Configuration m_cfg;

    mutable size_t independent_submodel_size = 0;
};

}  // namespace hetero
}  // namespace ov
