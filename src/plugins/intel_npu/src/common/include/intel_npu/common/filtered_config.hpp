// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Plugin Config Engine

#pragma once

#include "intel_npu/config/config.hpp"

namespace intel_npu {

/**
 * @class FilteredConfig
 * @brief A derived configuration class that extends the base `Config` class with filtering capabilities.
 *
 * This class provides additional functionality to manage and filter configuration options,
 * including enabling/disabling specific options, managing internal configurations, and
 * generating filtered configuration strings for compiler-specific needs.
 */
class FilteredConfig final : public Config {
public:
    using EnableMap =
        std::unordered_map<std::string, bool>;  ///< Map to track enabled/disabled state of configuration keys.

    /**
     * @brief Constructs a `FilteredConfig` object with a given options descriptor.
     * @param desc Shared pointer to an `OptionsDesc` object that describes available options.
     */
    explicit FilteredConfig(const std::shared_ptr<const OptionsDesc>& desc) : Config(desc) {}

    /**
     * @brief Updates the configuration with new options if the key is enabled state
     * @param options A map of key-value pairs representing the new configuration options.
     * @param mode Specifies the mode in which the options should be updated (default is `OptionMode::Both`).
     */
    void update(const ConfigMap& options, OptionMode mode = OptionMode::Both) override;

    /**
     * @brief Checks if a specific option exists in the configuration's descriptorDesc.
     * @param key The key of the option to check.
     * @return True if the option exists, false otherwise.
     */
    bool hasOpt(std::string_view key) const;

    /**
     * @brief Checks if a specific option is public (publishable in supported_properties).
     * @param key The key of the option to check.
     * @return True if the option is public, false otherwise.
     */
    bool isOptPublic(std::string_view key) const;

    /**
     * @brief Retrieves the OptionBase concept associated with a specific option. Used to check option details.
     * @param key The key of the option to retrieve.
     * @return The `OptionConcept` object representing the option's details.
     */
    details::OptionConcept getOpt(std::string_view key) const;

    /**
     * @brief Checks if a specific option is available (enabled and valid).
     * @param key The key of the option to check.
     * @return True if the option is available, false otherwise.
     */
    bool isAvailable(std::string key) const;

    /**
     * @brief Enables or disables a specific configuration option.
     * @param key The key of the option to enable/disable.
     * @param enable True to enable the option, false to disable it.
     */
    void enable(std::string key, bool enable);

    /**
     * @brief Enables all available configuration options.
     */
    void enableAll();

    /**
     * @brief Iterates over all enabled options and applies a callback function to each enabled key.
     * @param cb A callback function that takes a string (key) as input and performs an operation on it.
     */
    void walkEnables(std::function<void(const std::string&)> cb) const;

    /**
     * @brief Adds or updates an internal configuration value for compiler-specific needs.
     * @param key The key of the internal configuration to add or update.
     * @param value The value to set for the internal configuration.
     */
    void addOrUpdateInternal(std::string key, std::string value);

    /**
     * @brief Retrieves an internal configuration value by its key.
     * @param key The key of the internal configuration to retrieve.
     * @return The value associated with the specified internal configuration key.
     */
    std::string getInternal(std::string key) const;

    /**
     * @brief Generates a string representation of all internal configurations for compiler use.
     * @return A string containing all internal configurations formatted for compiler use.
     */
    std::string toStringForCompilerInternal() const;

    /**
     * @brief Iterates over all internal configurations and applies a callback function to each entry's key.
     * @param cb A callback function that takes a string (key) as input and performs an operation on it.
     */
    void walkInternals(std::function<void(const std::string&)> cb) const;

    /**
     * @brief Generates a string representation of all config keys with set values
     *        that are relevant for compilation (`OptionMode::Compile` or `OptionMode::Both`).
     * @return A string containing filtered configuration keys formatted for compiler use.
     */
    std::string toStringForCompiler() const;

private:
    EnableMap _enabled;  ///< Map to track whether specific configuration keys are enabled or disabled.

    ConfigMap _internal_compiler_configs;  ///< Map to store internal (hidden) configurations used for compiler.
};

}  // namespace intel_npu
