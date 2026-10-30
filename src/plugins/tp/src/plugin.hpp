// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/runtime/iplugin.hpp"
#include "tp/tp_config.hpp"

namespace ov {
namespace tp {

class Plugin : public ov::IPlugin {
public:
    Plugin();
    ~Plugin() = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(const ov::Tensor& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

private:
    /// One remote context per rank, created by the device plugin with the TP
    /// parameters of the collective contract: the device group and this rank's
    /// endpoint of an in-process exchange. The device plugin builds the
    /// communicator; every rank model compiled or imported with its context is
    /// bound to it. Used by both compile and import.
    std::vector<ov::SoPtr<ov::IRemoteContext>> create_rank_contexts(const std::vector<std::string>& device_names,
                                                                    const TPConfig& config) const;

    /// Restores a compiled model from a TP blob, shared by all four
    /// `import_model` overloads.
    std::shared_ptr<ov::ICompiledModel> import_blob(std::istream& blob, const ov::AnyMap& properties) const;

    /// The rank device list for property queries, which arrive without a
    /// compile config. Empty when the arguments name no devices.
    std::vector<std::string> query_devices(const ov::AnyMap& arguments) const;

    /// The rank device plugin's cache-defining properties, which are also
    /// supported: a TP blob is a container of rank blobs.
    std::vector<ov::PropertyName> rank_caching_properties(const ov::AnyMap& arguments) const;

    /// Answers a device-plugin property on behalf of the whole rank set.
    ov::Any aggregate_rank_property(const std::string& name, const ov::AnyMap& arguments) const;

    /// Builds the configuration of one call: what the plugin was configured
    /// with, overridden by what the call passed, finalized so the environment
    /// and the config file get their say. Everything that is not supported by TP lands
    /// in `forwarded`, ready for the rank compilations.
    void build_call_config(const ov::AnyMap& properties, TPConfig& config, ov::AnyMap& forwarded) const;

    /// Asks the rank devices' plugin whether it can run the collectives of
    /// `device_names` with the configured backend; throws its reason if not.
    void check_collective_support(const std::vector<std::string>& device_names, const TPConfig& config) const;

    /// Only ever holds what was explicitly set -- defaults, environment and
    /// config file are resolved per call.
    TPConfig m_config;

    /// Everything else set on the plugin, forwarded verbatim to the rank
    /// compilations.
    ov::AnyMap m_device_config;
};

}  // namespace tp
}  // namespace ov
