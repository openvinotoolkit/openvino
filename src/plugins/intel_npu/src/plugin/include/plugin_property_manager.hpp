// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>

#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "metrics.hpp"
#include "properties.hpp"

namespace intel_npu {

class PluginPropertyManager final {
public:
    PluginPropertyManager(const FilteredConfig& config,
                          const std::shared_ptr<Metrics>& metrics,
                          const ov::SoPtr<IEngineBackend>& backend,
                          Logger& logger);

    void setProperty(const ov::AnyMap& properties);
    ov::Any getProperty(const std::string& name, const ov::AnyMap& arguments = {}) const;
    bool isPropertySupported(const std::string& name, const ov::AnyMap& arguments = {}) const;

    const FilteredConfig& getConfig() const {
        return _properties.getConfig();
    }

    FilteredConfig getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties) const;
    FilteredConfig getConfigForSpecificCompiler(const ov::AnyMap& properties, const ICompilerAdapter* compiler) const;

    std::string determinePlatform(const ov::AnyMap& properties) const;
    std::string determineDeviceId(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerType(const ov::AnyMap& properties) const;

private:
    ov::CompatibilityCheck validateCompatibilityDescriptor(ov::intel_npu::CompilerType compilerType,
                                                           const ov::AnyMap& arguments) const;

    ov::SoPtr<IEngineBackend> _backend;
    Logger& _logger;
    mutable Properties _properties;
};

}  // namespace intel_npu