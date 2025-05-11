// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <map>

#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/properties.hpp"

#ifdef AUTOBATCH_UNITTEST
#    define autobatch_plugin mock_autobatch_plugin
#endif

namespace ov {
namespace autobatch_plugin {

struct DeviceInformation {
    std::string device_name;
    ov::AnyMap device_config;
    uint32_t device_batch_size;
};

class Plugin : public ov::IPlugin {
public:
    Plugin();

    virtual ~Plugin() = default;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& model,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

#ifdef AUTOBATCH_UNITTEST

public:
#else

protected:
#endif
    DeviceInformation parse_meta_device(const std::string& devices_batch_config, const ov::AnyMap& user_config) const;

    static DeviceInformation parse_batch_device(const std::string& device_with_batch);

private:
    mutable ov::AnyMap m_plugin_config;
};
}  // namespace autobatch_plugin
}  // namespace ov
