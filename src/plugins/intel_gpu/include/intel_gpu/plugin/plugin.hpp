// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/iplugin.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include <map>
#include <string>
#include <memory>

namespace ov {
namespace intel_gpu {

class Plugin : public ov::IPlugin {
private:
    std::string m_default_device_id = "0";
    std::map<std::string, cldnn::device::ptr> m_device_map;
    std::map<std::string, ExecutionConfig> m_configs_map;
    ov::AnyMap m_compiled_model_runtime_properties;

    mutable std::map<std::string, std::shared_ptr<RemoteContextImpl>> m_default_contexts;
    mutable std::once_flag m_default_contexts_once;

    std::map<std::string, std::shared_ptr<RemoteContextImpl>> get_default_contexts() const;

    std::shared_ptr<ov::Model> clone_and_transform_model(const std::shared_ptr<const ov::Model>& network,
                                                         const ExecutionConfig& config,
                                                         const std::shared_ptr<RemoteContextImpl>& context) const;
    void transform_model(std::shared_ptr<ov::Model>& model, const ExecutionConfig& config, const std::shared_ptr<RemoteContextImpl>& context) const;
    void register_primitives() const;
    std::string get_device_id_from_config(const ov::AnyMap& config) const;
    std::string get_device_id(const ov::AnyMap& config) const;
    std::shared_ptr<RemoteContextImpl> get_default_context(const std::string& device_id) const;

    std::vector<ov::PropertyName> get_caching_properties() const;
    std::vector<ov::PropertyName> get_supported_properties() const;
    std::vector<ov::PropertyName> get_supported_internal_properties() const;
    std::vector<std::string> get_device_capabilities(const cldnn::device_info& info) const;
    uint32_t get_optimal_batch_size(const ov::AnyMap& options) const;
    uint32_t get_max_batch_size(const ov::AnyMap& options) const;

    bool is_metric(const std::string& name) const;
    ov::Any get_metric(const std::string& name, const ov::AnyMap& arguments) const;

    void set_cache_info(const std::shared_ptr<const ov::Model>& model, ExecutionConfig& properties) const;

public:
    Plugin();

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;
    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;
    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;
    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;
    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;
};

}  // namespace intel_gpu
}  // namespace ov
