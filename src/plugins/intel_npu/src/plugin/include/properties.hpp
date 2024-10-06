// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_npu/al/config/config.hpp"
#include "metrics.hpp"

namespace intel_npu {

enum class PropertiesType { PLUGIN, COMPILED_MODEL };

class Properties final {
public:
    /**
     * @brief Properties handler constructor
     * @param pType - type of object this handler gets attached to: PLUGIN or COMPILED_MODEL
     * @param config - reference to the global configuration table of the parent object
     * @param metrics - reference ptr to the metrics object of the parent object (PLUGIN only)
     */
    Properties(const PropertiesType pType, Config& config, const std::shared_ptr<Metrics>& metrics = nullptr);

    /**
     * @brief Initialize the properties map and try registering the properties for npu-plugin and compiled-model
     * Can be used for both plugin and compiled-model properties maps, based on the provided pType param to the
     * constructor of this object
     * @details
     * - it will reset the properties map
     * - it will try registering config-backed option-based properties, with data from global configuration (supported,
     * visibilty, mutability, value)
     * - if an option is not present in the global config, it assumes it is not supported and will skip it
     * - it will register metric-based properties, with data from the metrics interface
     * - at the end it populates supported_properties with the now dynamically registered public properties
     */
    void registerProperties();

    /**
     * @brief Get the values of a property in a map
     */
    ov::Any get_property(const std::string& name, const ov::AnyMap& arguments) const;

    /**
     * @brief Set the values of a subset of properties, provided as a map
     * @details
     * - checks if the property exists, will report if unsupported
     * - checks if the property is Read-only, will report error if so
     */
    void set_property(const ov::AnyMap& properties);

private:
    PropertiesType _pType;
    Config& _config;
    std::shared_ptr<Metrics> _metrics;

    // properties map: {name -> [supported, mutable, eval function]}
    std::map<std::string, std::tuple<bool, ov::PropertyMutability, std::function<ov::Any(const Config&)>>> _properties;
    std::vector<ov::PropertyName> _supportedProperties;
};

}  // namespace intel_npu