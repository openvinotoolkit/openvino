// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "compiler_option_support_helper.hpp"
#include "intel_npu/common/filtered_config.hpp"
#include "intel_npu/common/icompiler_adapter.hpp"
#include "intel_npu/common/npu.hpp"
#include "intel_npu/config/npuw.hpp"
#include "intel_npu/utils/logger/logger.hpp"
#include "property_registration.hpp"

namespace intel_npu {

class PluginPropertyManager final {
public:
    PluginPropertyManager(const FilteredConfig& config,
                          const ov::SoPtr<IEngineBackend>& backend,
                          const std::shared_ptr<CompilerOptionSupportHelper>& optionSupportHelper,
                          Logger& logger);

    PluginPropertyManager& operator=(const PluginPropertyManager& other) = delete;

    void setProperty(const ov::AnyMap& properties);
    ov::Any getProperty(const std::string& name, const ov::AnyMap& arguments = {});
    bool isPropertySupported(const std::string& name, const ov::AnyMap& arguments = {});

    FilteredConfig deriveConfigForProperties(const ov::AnyMap& properties);

    const FilteredConfig& getConfig() const {
        return _config;
    }

    std::string determinePlatform(const ov::AnyMap& properties) const;
    std::string determineDeviceId(const ov::AnyMap& properties) const;
    ov::intel_npu::CompilerType determineCompilerType(const ov::AnyMap& properties) const;

private:
    void registerProperties();
    bool isPropertyRegistered(const std::string& propertyName) const;

    FilteredConfig _config;

    ov::SoPtr<IEngineBackend> _backend;
    std::shared_ptr<CompilerOptionSupportHelper> _compilerOptionSupportHelper;
    Logger& _logger;

    std::map<std::string, PropertyDescriptor> _properties;

    mutable std::mutex _mutex;
};

}  // namespace intel_npu
