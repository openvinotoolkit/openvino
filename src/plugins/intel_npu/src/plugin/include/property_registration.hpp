// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>
#include <utility>

#include "intel_npu/common/filtered_config.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

struct PropertyDescriptor final {
    bool isPublic;
    ov::PropertyMutability mutability;
    std::function<bool(const ov::AnyMap&)> isSupported;
    std::function<ov::Any(const ov::AnyMap&)> get;
};

class PropertyRegistrationBase {
protected:
    /**
     * @brief Register a simple property backed directly by a config option.
     *
     * Use this when the property name maps to an option stored in the provided config and the getter is just
     * config.get<OptionType>(). The descriptor is always registered, while runtime availability is gated by
     * config.hasOpt(propertyName).
     */
    template <typename OptionType>
    void register_property(const FilteredConfig& config, bool isPublic, ov::PropertyMutability mutability) {
        const auto propertyName = std::string(OptionType::key());
        _properties.emplace(
            propertyName,
            PropertyDescriptor{isPublic,
                               mutability,
                               std::function<bool(const ov::AnyMap&)>(
                                   [propertyName = std::string(propertyName), &config](const ov::AnyMap&) {
                                       return config.hasOpt(propertyName);
                                   }),
                               std::function<ov::Any(const ov::AnyMap&)>([&config](const ov::AnyMap&) {
                                   return config.get<OptionType>();
                               })});
    }

    /**
     * @brief Register a config-backed property with explicit getter function.
     *
     * Use this when a custom getter function is required. Visibility and mutability are provided explicitly by the
     * caller. The descriptor is always registered, and runtime availability is gated by config.hasOpt(propertyName).
     */
    template <typename OptionType>
    void register_property_with_custom_function(const FilteredConfig& config,
                                                bool isPublic,
                                                ov::PropertyMutability mutability,
                                                std::function<ov::Any(const ov::AnyMap&)> getter) {
        const auto propertyName = std::string(OptionType::key());
        _properties.emplace(
            propertyName,
            PropertyDescriptor{isPublic,
                               mutability,
                               std::function<bool(const ov::AnyMap&)>(
                                   [propertyName = std::string(propertyName), &config](const ov::AnyMap&) {
                                       return config.hasOpt(propertyName);
                                   }),
                               std::move(getter)});
    }

    /**
     * @brief Register a config-backed property with support check.
     *
     * Registers the descriptor unconditionally, but exposes the property only when the supplied `isSupported` predicate
     * returns true at runtime. Getter is config.get<OptionType>(). Use this when a property's availability depends on
     * dynamic runtime conditions.
     */
    template <typename OptionType>
    void register_property_with_support(const FilteredConfig& config,
                                        bool isPublic,
                                        ov::PropertyMutability mutability,
                                        std::function<bool(const ov::AnyMap&)> isSupported) {
        const auto propertyName = std::string(OptionType::key());
        _properties.emplace(propertyName,
                            PropertyDescriptor{isPublic,
                                               mutability,
                                               std::move(isSupported),
                                               std::function<ov::Any(const ov::AnyMap&)>([&config](const ov::AnyMap&) {
                                                   return config.get<OptionType>();
                                               })});
    }

    /**
     * @brief Register a config-backed property with support check and custom getter.
     *
     * Registers the descriptor unconditionally, but exposes the property only when the supplied `isSupported` predicate
     * returns true at runtime. A custom getter function is required.
     */
    template <typename OptionType>
    void register_property_with_support_and_custom_function(const FilteredConfig& config,
                                                            bool isPublic,
                                                            ov::PropertyMutability mutability,
                                                            std::function<bool(const ov::AnyMap&)> isSupported,
                                                            std::function<ov::Any(const ov::AnyMap&)> getter) {
        const auto propertyName = std::string(OptionType::key());
        _properties.emplace(propertyName,
                            PropertyDescriptor{isPublic, mutability, std::move(isSupported), std::move(getter)});
    }

    /**
     * @brief Register an exposed NPUW option property backed by config.
     *
     * Equivalent to register_property, but derives the property name from OptionType::key().
     */
    template <typename OptionType>
    void register_npuw_property(const FilteredConfig& config) {
        const auto propertyName = std::string(OptionType::key());
        _properties.emplace(
            propertyName,
            PropertyDescriptor{false,  // NPUW options are not public
                               ov::PropertyMutability::RW,
                               std::function<bool(const ov::AnyMap&)>(
                                   [propertyName = std::string(propertyName), &config](const ov::AnyMap&) {
                                       return config.hasOpt(propertyName);
                                   }),
                               std::function<ov::Any(const ov::AnyMap&)>([&config](const ov::AnyMap&) {
                                   return config.get<OptionType>();
                               })});
    }

    /**
     * @brief Register a simple property with custom getter.
     *
     * The property is read-only and always available.
     * The getter receives query-time arguments and returns the property value.
     */
    void register_property_with_custom_function(const std::string& propertyName,
                                                bool isPublic,
                                                ov::PropertyMutability mutability,
                                                std::function<ov::Any(const ov::AnyMap&)> getter) {
        _properties.emplace(propertyName,
                            PropertyDescriptor{isPublic,
                                               mutability,
                                               std::function<bool(const ov::AnyMap&)>([](const ov::AnyMap&) {
                                                   return true;
                                               }),
                                               std::move(getter)});
    }

    /**
     * @brief Register a property with support check and custom getter.
     *
     * Registers a property that is always added to the descriptor but gated by an `isSupported` condition at runtime
     * and a custom getter function is required.
     */
    void register_property_with_support_and_custom_function(const std::string& propertyName,
                                                            bool isPublic,
                                                            ov::PropertyMutability mutability,
                                                            std::function<bool(const ov::AnyMap&)> isSupported,
                                                            std::function<ov::Any(const ov::AnyMap&)> getter) {
        _properties.emplace(propertyName,
                            PropertyDescriptor{isPublic, mutability, std::move(isSupported), std::move(getter)});
    }

    std::map<std::string, PropertyDescriptor> _properties;
};

}  // namespace intel_npu
