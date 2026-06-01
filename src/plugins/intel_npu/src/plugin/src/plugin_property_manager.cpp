// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "plugin_property_manager.hpp"

#include "intel_npu/common/compiler_adapter_factory.hpp"
#include "metadata.hpp"

namespace {

std::shared_ptr<const ov::Model> exclude_model_ptr_from_map(ov::AnyMap& properties) {
    std::shared_ptr<const ov::Model> modelPtr = nullptr;
    if (properties.count(ov::hint::model.name())) {
        try {
            modelPtr = properties.at(ov::hint::model.name()).as<std::shared_ptr<const ov::Model>>();
        } catch (const ov::Exception&) {
            try {
                modelPtr = std::const_pointer_cast<const ov::Model>(
                    properties.at(ov::hint::model.name()).as<std::shared_ptr<ov::Model>>());
            } catch (const ov::Exception&) {
                OPENVINO_THROW("The value of the \"ov::hint::model\" configuration option (\"MODEL_PTR\") has the "
                               "wrong data type. Expected: std::shared_ptr<const ov::Model>.");
            }
        }
        properties.erase(ov::hint::model.name());
    }
    return modelPtr;
}

}  // namespace

namespace intel_npu {

PluginPropertyManager::PluginPropertyManager(const FilteredConfig& config,
                                             const std::shared_ptr<Metrics>& metrics,
                                             const ov::SoPtr<IEngineBackend>& backend,
                                             Logger& logger)
    : _backend(backend),
      _logger(logger),
      _properties(PropertiesType::PLUGIN, config, metrics, backend) {}

void PluginPropertyManager::setProperty(const ov::AnyMap& properties) {
    _properties.setProperty(properties);
}

ov::Any PluginPropertyManager::getProperty(const std::string& name, const ov::AnyMap& arguments) const {
    if (name == ov::compatibility_check.name()) {
        _properties.getProperty(name);
        auto compilerType = _properties.determineCompilerTypeForCompatibilityCheck();
        return validateCompatibilityDescriptor(compilerType, arguments);
    }

    if (!arguments.empty()) {
        auto pluginArguments = arguments;
        exclude_model_ptr_from_map(pluginArguments);

        auto copyProperties = Properties(_properties);
        copyProperties.setProperty(pluginArguments);
        return copyProperties.getProperty(name);
    }

    return _properties.getProperty(name);
}

bool PluginPropertyManager::isPropertySupported(const std::string& name, const ov::AnyMap& arguments) const {
    if (!arguments.empty()) {
        auto pluginArguments = arguments;
        exclude_model_ptr_from_map(pluginArguments);

        auto copyProperties = Properties(_properties);
        try {
            copyProperties.setProperty(pluginArguments);
        } catch (...) {
            return false;
        }

        return copyProperties.isPropertySupported(name);
    }

    return _properties.isPropertySupported(name);
}

FilteredConfig PluginPropertyManager::getConfigWithCompilerPropertiesDisabled(const ov::AnyMap& properties) const {
    return _properties.getConfigWithCompilerPropertiesDisabled(properties);
}

FilteredConfig PluginPropertyManager::getConfigForSpecificCompiler(const ov::AnyMap& properties,
                                                                   const ICompilerAdapter* compiler) const {
    return _properties.getConfigForSpecificCompiler(properties, compiler);
}

std::string PluginPropertyManager::determinePlatform(const ov::AnyMap& properties) const {
    return _properties.determinePlatform(properties);
}

std::string PluginPropertyManager::determineDeviceId(const ov::AnyMap& properties) const {
    return _properties.determineDeviceId(properties);
}

ov::intel_npu::CompilerType PluginPropertyManager::determineCompilerType(const ov::AnyMap& properties) const {
    return _properties.determineCompilerType(properties);
}

ov::CompatibilityCheck PluginPropertyManager::validateCompatibilityDescriptor(
    ov::intel_npu::CompilerType compilerType,
    const ov::AnyMap& arguments) const {
    if (arguments.empty() || arguments.find(ov::runtime_requirements.name()) == arguments.end()) {
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }

    const auto& runtimeRequirements = arguments.at(ov::runtime_requirements.name()).as<const std::string&>();
    _logger.debug("Received runtime_requirements: %s length: %zu",
                  runtimeRequirements.c_str(),
                  runtimeRequirements.length());

    std::unique_ptr<MetadataBase> metadata = nullptr;
    try {
        metadata = read_as_text(runtimeRequirements);
    } catch (const std::exception& ex) {
        _logger.debug("Failed to read metadata from the runtime requirements. The requirements are not met. %s",
                      ex.what());
        return ov::CompatibilityCheck::UNSUPPORTED;
    }

    const auto descriptorView = metadata->get_compatibility_descriptor();
    std::string compatibilityDescriptor = descriptorView.has_value() ? std::string(descriptorView.value()) : "";
    _logger.debug("Retrieved compatibility descriptor from metadata: %s length: %zu",
                  compatibilityDescriptor.c_str(),
                  compatibilityDescriptor.length());

    std::unique_ptr<ICompilerAdapter> compiler = nullptr;
    CompilerAdapterFactory factory;
    try {
        compiler = factory.getCompiler(_backend, compilerType, std::string_view{});

        auto result = compiler->validate_compatibility_descriptor(compatibilityDescriptor);
        _logger.debug("Compatibility check result: %s", result ? "met" : "not met");
        return result ? ov::CompatibilityCheck::SUPPORTED : ov::CompatibilityCheck::UNSUPPORTED;
    } catch (const std::exception&) {
        _logger.error("Failed to create the recommended compiler type for the compatibility check %d. The requirements "
                      "are not met.",
                      static_cast<int>(compilerType));
        return ov::CompatibilityCheck::NOT_APPLICABLE;
    }
}

}  // namespace intel_npu