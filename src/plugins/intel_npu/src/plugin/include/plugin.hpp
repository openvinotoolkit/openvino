// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>

#include "backends.hpp"
#include "intel_npu/al/config/config.hpp"
#include "intel_npu/al/icompiler.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "metrics.hpp"
#include "npu.hpp"
#include "openvino/runtime/iplugin.hpp"
#include "openvino/runtime/so_ptr.hpp"
#include "properties.hpp"

namespace intel_npu {

class Plugin : public ov::IPlugin {
public:
    Plugin();

    Plugin(const Plugin&) = delete;

    Plugin& operator=(const Plugin&) = delete;

    virtual ~Plugin() = default;

    void set_property(const ov::AnyMap& properties) override;

    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> compile_model(const std::shared_ptr<const ov::Model>& model,
                                                      const ov::AnyMap& properties,
                                                      const ov::SoPtr<ov::IRemoteContext>& context) const override;

    ov::SoPtr<ov::IRemoteContext> create_context(const ov::AnyMap& remote_properties) const override;

    ov::SoPtr<ov::IRemoteContext> get_default_context(const ov::AnyMap& remote_properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream, const ov::AnyMap& properties) const override;

    std::shared_ptr<ov::ICompiledModel> import_model(std::istream& stream,
                                                     const ov::SoPtr<ov::IRemoteContext>& context,
                                                     const ov::AnyMap& properties) const override;

    ov::SupportedOpsMap query_model(const std::shared_ptr<const ov::Model>& model,
                                    const ov::AnyMap& properties) const override;

private:
    ov::SoPtr<ICompiler> getCompiler(const Config& config) const;
    void init_options(compilerVersion comp_ver);
    void init_properties();

    std::shared_ptr<NPUBackends> _backends;

    std::shared_ptr<OptionsDesc> _options;
    Config _globalConfig;
    mutable Logger _logger;
    std::shared_ptr<Metrics> _metrics;
    std::unique_ptr<Properties> _properties;

    static std::atomic<int> _compiledModelLoadCounter;
};

}  // namespace intel_npu
