// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <memory>

#include "ie_parameter.hpp"

#include "vpu/utils/logger.hpp"

namespace vpu {

class PluginConfiguration;

struct ConfigurationOptionConcept {
    virtual std::string key() const = 0;
    virtual void validate(const std::string&) const = 0;
    virtual void validate(const PluginConfiguration&) const = 0;
    virtual InferenceEngine::Parameter asParameter(const std::string&) const = 0;
};

namespace details {

template<class Option>
struct ConfigurationOptionModel : public ConfigurationOptionConcept {
    std::string key() const override { return Option::key(); }
    void validate(const std::string& value) const override { return Option::validate(value); }
    void validate(const PluginConfiguration& options) const override { Option::validate(options); }
    InferenceEngine::Parameter asParameter(const std::string& value) const override { return Option::asParameter(value); }
};

enum class Deprecation {
    Off,
    On
};

enum class Access {
    Private,
    Public
};

enum class Category {
    CompileTime,
    RunTime
};

class ConfigurationEntry {
public:
    template<class Option>
    ConfigurationEntry(Option, details::Deprecation deprecation)
        : m_access(Option::access())
        , m_deprecation(deprecation)
        , m_category(Option::category())
        , m_value(std::make_shared<ConfigurationOptionModel<Option>>())
        {}

    ConfigurationOptionConcept& get();
    const ConfigurationOptionConcept& get() const;

    std::string key() const;
    bool isPrivate() const;
    bool isDeprecated() const;
    Category getCategory() const;

private:
    Access m_access = Access::Public;
    Deprecation m_deprecation = Deprecation::Off;
    Category m_category = Category::CompileTime;
    std::shared_ptr<ConfigurationOptionConcept> m_value;
};

}  // namespace details

class PluginConfiguration {
public:
    PluginConfiguration();

    void from(const std::map<std::string, std::string>& config);
    void fromAtRuntime(const std::map<std::string, std::string>& config);
    std::unordered_set<std::string> getPublicKeys() const;
    bool supports(const std::string& key) const;

    template<class Option>
    void registerOption() {
        const auto& key = Option::key();
        concepts.emplace(key, details::ConfigurationEntry(Option{}, details::Deprecation::Off));
        if (values.count(key) == 0) {
            // option could be registered more than once if there are deprecated versions of it
            values.emplace(key, Option::defaultValue());
        }
    }

    template<class Option>
    void registerDeprecatedOption(const std::string& deprecatedKey) {
        const auto& key = Option::key();
        concepts.emplace(deprecatedKey, details::ConfigurationEntry(Option{}, details::Deprecation::On));
        if (values.count(key) == 0) {
            // option could be registered more than once if there are deprecated versions of it
            values.emplace(key, Option::defaultValue());
        }
    }

    template<class Option>
    typename Option::value_type get() const {
        const auto& key = Option::key();
        validate(key);
        return Option::parse(values.at(key));
    }

    void set(const std::string& key, const std::string& value);

    const std::string& operator[](const std::string& key) const;

    InferenceEngine::Parameter asParameter(const std::string& key) const;

    virtual void validate() const;

private:
    std::unordered_map<std::string, details::ConfigurationEntry> concepts;
    std::unordered_map<std::string, std::string> values;

    Logger::Ptr logger;

    enum class Mode {
        Default,
        RunTime
    };
    void create(const std::map<std::string, std::string>& config, Mode mode = Mode::Default);

    void validate(const std::string& key) const;
};

}  // namespace vpu
