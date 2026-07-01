// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "intel_npu/common/filtered_config.hpp"
#include "openvino/runtime/properties.hpp"

namespace intel_npu {

struct PropertyDescriptor final {
    bool isPublic;
    ov::PropertyMutability mutability;
    std::function<bool(const FilteredConfig&)> isSupported;
    std::function<ov::Any(const FilteredConfig&)> get;
    std::function<ov::Any(const FilteredConfig&, const ov::AnyMap&)> getWithArgs;
};

using PropertyMap = std::map<std::string, PropertyDescriptor>;

inline void ensure_option_exists_in_config(const FilteredConfig& config, const std::string& propertyName) {
    if (!config.hasOpt(propertyName)) {
        OPENVINO_THROW("Property '", propertyName, "' is not backed by a registered config option");
    }
}

/**
 * @brief Register a simple property backed directly by a config option.
 *
 * Use this when the property name maps to an option stored in the provided config and the getter is just
 * config.get<OptionType>(). The property is registered unconditionally, while runtime support is checked via
 * config.isAvailable(propertyName).
 */
template <typename OptionType>
inline void register_property(const FilteredConfig& config, PropertyMap& properties, const std::string& propertyName) {
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(
        propertyName,
        PropertyDescriptor{
            option.isPublic(),
            option.mutability(),
            std::function<bool(const FilteredConfig&)>([propertyName](const FilteredConfig& configValue) {
                return configValue.isAvailable(propertyName);
            }),
            std::function<ov::Any(const FilteredConfig&)>([](const FilteredConfig& configValue) {
                return configValue.get<OptionType>();
            }),
            {}});
}

/**
 * @brief Register a config-backed property with explicit public/private visibility.
 *
 * Use this when the callback is the standard config.get<OptionType>() but the visibility (public/private) must be
 * provided by the caller rather than taken from the option descriptor. The property is always registered, while
 * support is gated at query time via config.isAvailable(propertyName).
 */
template <typename OptionType>
inline void register_property_with_custom_visibility(const FilteredConfig& config,
                                                     PropertyMap& properties,
                                                     const std::string& propertyName,
                                                     bool isPublic) {
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(
        propertyName,
        PropertyDescriptor{
            isPublic,
            option.mutability(),
            std::function<bool(const FilteredConfig&)>([propertyName](const FilteredConfig& configValue) {
                return configValue.isAvailable(propertyName);
            }),
            std::function<ov::Any(const FilteredConfig&)>([](const FilteredConfig& configValue) {
                return configValue.get<OptionType>();
            }),
            {}});
}

/**
 * @brief Register a config-backed property with explicit getter function.
 *
 * Use this when a custom getter function is required. Visibility and mutability are taken from the option descriptor.
 * The property is available only if the underlying config option is available.
 */
template <typename Getter>
inline void register_property_with_custom_function(const FilteredConfig& config,
                                                   PropertyMap& properties,
                                                   const std::string& propertyName,
                                                   Getter&& getter) {
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(propertyName,
                       PropertyDescriptor{option.isPublic(),
                                          option.mutability(),
                                          std::function<bool(const FilteredConfig&)>(
                                              [propertyName](const FilteredConfig& configValue) {
                                                  return configValue.isAvailable(propertyName);
                                              }),
                                          std::function<ov::Any(const FilteredConfig&)>(std::forward<Getter>(getter)),
                                          {}});
}

/**
 * @brief Register a config-backed property with support check.
 *
 * Registers a property that is always added to the descriptor but gated by an `isSupported` predicate at runtime.
 * Getter is config.get<OptionType>(). Use this when a property's availability depends on different runtime conditions.
 */
template <typename OptionType, typename IsSupportedFn>
inline void register_property_with_support(const FilteredConfig& config,
                                           PropertyMap& properties,
                                           const std::string& propertyName,
                                           IsSupportedFn&& isSupported) {
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(
        propertyName,
        PropertyDescriptor{option.isPublic(),
                           option.mutability(),
                           std::function<bool(const FilteredConfig&)>(std::forward<IsSupportedFn>(isSupported)),
                           std::function<ov::Any(const FilteredConfig&)>([](const FilteredConfig& configValue) {
                               return configValue.get<OptionType>();
                           }),
                           {}});
}

/**
 * @brief Register a config-backed property with support check and custom getter.
 *
 * Registers a property that is always added to the descriptor but gated by an `isSupported` condition at runtime and a
 * custom getter function is required.
 */
template <typename IsSupportedFn, typename Getter>
inline void register_property_with_support_and_custom_function(const FilteredConfig& config,
                                                               PropertyMap& properties,
                                                               const std::string& propertyName,
                                                               IsSupportedFn&& isSupported,
                                                               Getter&& getter) {
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(
        propertyName,
        PropertyDescriptor{option.isPublic(),
                           option.mutability(),
                           std::function<bool(const FilteredConfig&)>(std::forward<IsSupportedFn>(isSupported)),
                           std::function<ov::Any(const FilteredConfig&)>(std::forward<Getter>(getter)),
                           {}});
}

/**
 * @brief Register a property backed directly by the current config as read-only.
 *
 * The property is read-only. Getter is config.get<OptionType>(). Runtime support is checked via
 * config.isAvailable(propertyName).
 */
template <typename OptionType>
inline void register_property_as_read_only(const FilteredConfig& config,
                                           PropertyMap& properties,
                                           const std::string& propertyName) {
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(
        propertyName,
        PropertyDescriptor{
            option.isPublic(),
            ov::PropertyMutability::RO,
            std::function<bool(const FilteredConfig&)>([propertyName](const FilteredConfig& configValue) {
                return configValue.isAvailable(propertyName);
            }),
            std::function<ov::Any(const FilteredConfig&)>([](const FilteredConfig& configValue) {
                return configValue.get<OptionType>();
            }),
            {}});
}

/**
 * @brief Register a config-backed property with support check for value presence.
 *
 * Default option values are not materialized into the config, so this form advertises a property only when the user or
 * upper layer actually set it. The property is read-only.
 */
template <typename OptionType>
inline void register_property_as_read_only_mark_supported_if_set(const FilteredConfig& config,
                                                                 PropertyMap& properties,
                                                                 const std::string& propertyName) {
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(
        propertyName,
        PropertyDescriptor{
            option.isPublic(),
            ov::PropertyMutability::RO,
            std::function<bool(const FilteredConfig&)>([propertyName](const FilteredConfig& configValue) {
                return (configValue.isAvailable(propertyName) && configValue.has(propertyName));
            }),
            std::function<ov::Any(const FilteredConfig&)>([](const FilteredConfig& configValue) {
                return configValue.get<OptionType>();
            }),
            {}});
}

/**
 * @brief Register an exposed NPUW option property backed by config.
 *
 * Equivalent to register_property, but derives the property name from OptionType::key().
 */
template <typename OptionType>
inline void register_npuw_property(const FilteredConfig& config, PropertyMap& properties) {
    const auto propertyName = std::string(OptionType::key());
    ensure_option_exists_in_config(config, propertyName);
    const auto& option = config.getOpt(propertyName);
    properties.emplace(
        propertyName,
        PropertyDescriptor{
            option.isPublic(),
            option.mutability(),
            std::function<bool(const FilteredConfig&)>([propertyName](const FilteredConfig& configValue) {
                return configValue.isAvailable(propertyName);
            }),
            std::function<ov::Any(const FilteredConfig&)>([](const FilteredConfig& configValue) {
                return configValue.get<OptionType>();
            }),
            {}});
}

template <typename GetterOrValue>
inline auto normalize_getter(GetterOrValue&& getterOrValue) {
    using GetterOrValueType = std::decay_t<GetterOrValue>;

    if constexpr (std::is_invocable_v<GetterOrValueType, const FilteredConfig&>) {
        return std::forward<GetterOrValue>(getterOrValue);
    } else {
        return [value = std::forward<GetterOrValue>(getterOrValue)](const FilteredConfig&) {
            return value;
        };
    }
}

/**
 * @brief Register a simple property with custom getter.
 *
 * The property is read-only and always available.
 * Accepts either:
 * - a callable with signature compatible with ov::Any(const FilteredConfig&)
 * - a plain value, which is returned as-is regardless of config
 */
template <typename GetterOrValue>
inline void register_property_with_custom_function(PropertyMap& properties,
                                                   const std::string& propertyName,
                                                   bool isPublic,
                                                   GetterOrValue&& getterOrValue) {
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          ov::PropertyMutability::RO,
                                          std::function<bool(const FilteredConfig&)>([](const FilteredConfig&) {
                                              return true;
                                          }),
                                          std::function<ov::Any(const FilteredConfig&)>(
                                              normalize_getter(std::forward<GetterOrValue>(getterOrValue))),
                                          {}});
}

/**
 * @brief Register a property with support check and custom getter.
 *
 * Registers a property that is always added to the descriptor but gated by an `isSupported` condition at runtime and a
 * custom getter function is required.
 */
template <typename IsSupportedFn, typename Getter>
inline void register_property_with_support_and_custom_function(PropertyMap& properties,
                                                               const std::string& propertyName,
                                                               IsSupportedFn&& isSupported,
                                                               bool isPublic,
                                                               Getter&& getter) {
    properties.emplace(
        propertyName,
        PropertyDescriptor{isPublic,
                           ov::PropertyMutability::RO,
                           std::function<bool(const FilteredConfig&)>(std::forward<IsSupportedFn>(isSupported)),
                           std::function<ov::Any(const FilteredConfig&)>(std::forward<Getter>(getter)),
                           {}});
}

/**
 * @brief Register a property with support check, custom getter, and query-time arguments.
 *
 * Registers a property that is always added to the descriptor but gated by an `isSupported` condition at runtime.
 * Uses getWithArgs instead of get, so the getter receives both FilteredConfig and query-time arguments.
 * Use this for properties that accept extra arguments at get_property call time (e.g. compatibility_check).
 */
template <typename IsSupportedFn, typename Getter>
inline void register_property_with_support_custom_function_and_args(PropertyMap& properties,
                                                                    const std::string& propertyName,
                                                                    IsSupportedFn&& isSupported,
                                                                    bool isPublic,
                                                                    Getter&& getter) {
    properties.emplace(
        propertyName,
        PropertyDescriptor{
            isPublic,
            ov::PropertyMutability::RO,
            std::function<bool(const FilteredConfig&)>(std::forward<IsSupportedFn>(isSupported)),
            {},
            std::function<ov::Any(const FilteredConfig&, const ov::AnyMap&)>(std::forward<Getter>(getter))});
}

}  // namespace intel_npu
