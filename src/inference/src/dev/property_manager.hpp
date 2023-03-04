// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief OpenVINO Runtime Config Manager
 * @file openvino/runtime/property_manager.hpp
 */

#pragma once

#include "dev/plugin.hpp"
#include "ie_cache_manager.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {

/**
 * @brief cache_manager
 */
static constexpr Property<std::shared_ptr<ov::ICacheManager>> cache_manager{"CACHE_MANAGER"};

/**
 * @interface PropertyManager
 * @brief Interface for property manager.
 * This is global point for setting and getting core's properties.
 * @ingroup ov_dev_api
 */
class OPENVINO_RUNTIME_API PropertyManager {
    class Impl;
    std::shared_ptr<Impl> _impl;

public:
    /**
     * @brief Constructs an PropertyManager Instance
     */
    PropertyManager();

    /**
     * @brief Destructor
     */

    ~PropertyManager() = default;

    /**
     * @brief Merge core property items of input_property into core_properties_cache per plugin
     * @param property is the input property that will be merged
     * @param plugin_name point to the plugin that core properties will be merged into, it can be empty
     * @return void
     */
    void merge_property(const ov::AnyMap& property, const std::string& plugin_name = {});

    /**
     * @brief Subtract core property items from input_property
     * @param property is the input property that need exclude core properties
     * @return properties have been excluded core_property items
     */
    ov::AnyMap exclude_property(const ov::AnyMap& property);

    /**
     * @brief Get property from property manager
     * @param property_name is the property name
     * @param plugin_name points to the plugin that will be operated
     * @return property value
     */
    ov::Any get_property(const std::string& property_name, const std::string& plugin_name = {});

    /**
     * @brief Check whether this property belongs core property
     * @param property_name is the property name
     * @return True if it is core property, else false
     */
    bool is_core_property(const std::string& property_name);
};

}  // namespace ov