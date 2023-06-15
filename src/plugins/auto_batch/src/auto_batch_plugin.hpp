// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ie/ie_plugin_config.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"

namespace ov {
namespace autobatch_plugin {

struct DeviceInformation {
    std::string deviceName;
    ov::AnyMap config;
    uint32_t batchForDevice;
};

class Plugin : public ov::IPlugin {
public:
    Plugin();

    virtual ~Plugin() = default;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::RemoteContext& context) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::RemoteContext& context,
                                                     const ov::AnyMap& properties) const override;

private:
    ov::AnyMap m_plugin_config;
    std::vector<std::string> supported_configKeys = {CONFIG_KEY(AUTO_BATCH_DEVICE_CONFIG),
                                                     ov::device::priorities.name(),
                                                     ov::auto_batch_timeout.name(),
                                                     ov::cache_dir.name()};

protected:
    static DeviceInformation parse_batch_device(const std::string& deviceWithBatch);

    DeviceInformation parse_meta_device(const std::string& devicesBatchCfg, const ov::AnyMap& config) const;
};

}  // namespace autobatch_plugin
}  // namespace ov