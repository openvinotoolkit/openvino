// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/utils/error.hpp"
#include "vpu/configuration/plugin_configuration.hpp"

#include "ie_plugin_config.hpp"

namespace vpu {

namespace details {

ConfigurationOptionConcept& ConfigurationEntry::get() {
    return *m_value;
}

const ConfigurationOptionConcept& ConfigurationEntry::get() const {
    return *m_value;
}

bool ConfigurationEntry::isPrivate() const {
    return m_access == Access::Private;
}

bool ConfigurationEntry::isDeprecated() const {
    return m_deprecation == Deprecation::On;
}

Category ConfigurationEntry::getCategory() const {
    return m_category;
}

std::string ConfigurationEntry::key() const {
    return m_value->key();
}

}  // namespace details

PluginConfiguration::PluginConfiguration() : logger(std::make_shared<Logger>("Configuration", LogLevel::Warning, consoleOutput())) {}


std::unordered_set<std::string> PluginConfiguration::getPublicKeys() const {
    auto publicKeys = std::unordered_set<std::string>{};
    for (const auto& entry : concepts) {
        const auto& key    = entry.first;
        const auto& option = entry.second;
        if (option.isPrivate()) {
            continue;
        }
        publicKeys.insert(key);
    }
    return publicKeys;
}

bool PluginConfiguration::supports(const std::string& key) const {
    return concepts.count(key) != 0;
}

void PluginConfiguration::from(const std::map<std::string, std::string>& config) {
    create(config);
}

void PluginConfiguration::fromAtRuntime(const std::map<std::string, std::string>& config) {
    create(config, Mode::RunTime);
}

void PluginConfiguration::validate() const {
    for (const auto& option : concepts) {
        option.second.get().validate(*this);
    }
}

void PluginConfiguration::create(const std::map<std::string, std::string>& config, Mode mode) {
    for (const auto& entry : config) {
        const auto& key = entry.first;
        validate(key);

        const auto& optionConcept = concepts.at(key);
        if (mode == Mode::RunTime && optionConcept.getCategory() == details::Category::CompileTime) {
            logger->warning("Configuration option \"{}\" is used after network is loaded. Its value is going to be ignored.", key);
            continue;
        }

        const auto& value = entry.second;
        set(key, value);
    }
}

InferenceEngine::Parameter PluginConfiguration::asParameter(const std::string& key) const {
    const auto& value = operator[](key);
    return concepts.at(key).get().asParameter(value);
}

void PluginConfiguration::validate(const std::string& key) const {
    VPU_THROW_UNSUPPORTED_OPTION_UNLESS(supports(key), "Encountered an unsupported key {}, supported keys are {}", key, getPublicKeys());
    if (concepts.at(key).isDeprecated()) {
        if (concepts.at(key).key() != key) {
            logger->warning("Encountered deprecated option {} usage, consider replacing it with {} option", key, concepts.at(key).key());
        }
    }
}

const std::string& PluginConfiguration::operator[](const std::string& key) const {
    validate(key);
    return values.at(concepts.at(key).key());
}

void PluginConfiguration::set(const std::string& key, const std::string& value) {
    validate(key);
    const auto& optionConcept = concepts.at(key).get();
    optionConcept.validate(value);
    values[optionConcept.key()] = value;
}

}  // namespace vpu
