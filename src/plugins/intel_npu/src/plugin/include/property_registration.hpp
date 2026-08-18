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
    std::function<bool()> isSupported;
    std::function<ov::Any(const ov::AnyMap&)> get;
};

using PropertyMap = std::map<std::string, PropertyDescriptor>;
/**
 * @brief Register a simple property backed directly by a config option.
 *
 * Use this when the property name maps to an option stored in the provided config and the getter is just
 * config.get<OptionType>(). The property is registered unconditionally, while runtime support is checked via
 * config.isAvailable(propertyName).
 */
template <typename OptionType>
inline void register_property(const FilteredConfig& config,
                              PropertyMap& properties,
                              bool isPublic,
                              ov::PropertyMutability mutability) {
    const auto propertyName = std::string(OptionType::key());
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          mutability,
                                          std::function<bool()>([propertyName = std::string(propertyName), &config]() {
                                              return config.hasOpt(propertyName);
                                          }),
                                          std::function<ov::Any(const ov::AnyMap&)>([&config](const ov::AnyMap&) {
                                              return config.get<OptionType>();
                                          })});
}

/**
 * @brief Register a config-backed property with explicit getter function.
 *
 * Use this when a custom getter function is required. Visibility and mutability are taken from the option descriptor.
 * The property is available only if the underlying config option is available.
 */
template <typename OptionType, typename Getter>
inline void register_property_with_custom_function(const FilteredConfig& config,
                                                   PropertyMap& properties,
                                                   bool isPublic,
                                                   ov::PropertyMutability mutability,
                                                   Getter&& getter) {
    const auto propertyName = std::string(OptionType::key());
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          mutability,
                                          std::function<bool()>([propertyName = std::string(propertyName), &config]() {
                                              return config.hasOpt(propertyName);
                                          }),
                                          std::function<ov::Any(const ov::AnyMap&)>(std::forward<Getter>(getter))});
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
                                           bool isPublic,
                                           ov::PropertyMutability mutability,
                                           IsSupportedFn&& isSupported) {
    const auto propertyName = std::string(OptionType::key());
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          mutability,
                                          std::function<bool()>(std::forward<IsSupportedFn>(isSupported)),
                                          std::function<ov::Any(const ov::AnyMap&)>([&config](const ov::AnyMap&) {
                                              return config.get<OptionType>();
                                          })});
}

/**
 * @brief Register a config-backed property with support check and custom getter.
 *
 * Registers a property that is always added to the descriptor but gated by an `isSupported` condition at runtime and a
 * custom getter function is required.
 */
template <typename OptionType, typename IsSupportedFn, typename Getter>
inline void register_property_with_support_and_custom_function(const FilteredConfig& config,
                                                               PropertyMap& properties,
                                                               bool isPublic,
                                                               ov::PropertyMutability mutability,
                                                               IsSupportedFn&& isSupported,
                                                               Getter&& getter) {
    const auto propertyName = std::string(OptionType::key());
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          mutability,
                                          std::function<bool()>(std::forward<IsSupportedFn>(isSupported)),
                                          std::function<ov::Any(const ov::AnyMap&)>(std::forward<Getter>(getter))});
}

/**
 * @brief Register an exposed NPUW option property backed by config.
 *
 * Equivalent to register_property, but derives the property name from OptionType::key().
 */
template <typename OptionType>
inline void register_npuw_property(const FilteredConfig& config, PropertyMap& properties) {
    const auto propertyName = std::string(OptionType::key());
    properties.emplace(propertyName,
                       PropertyDescriptor{false,  // NPUW options are not public
                                          ov::PropertyMutability::RW,
                                          std::function<bool()>([propertyName = std::string(propertyName), &config]() {
                                              return config.hasOpt(propertyName);
                                          }),
                                          std::function<ov::Any(const ov::AnyMap&)>([&config](const ov::AnyMap&) {
                                              return config.get<OptionType>();
                                          })});
}

template <typename GetterOrValue>
inline auto normalize_getter(GetterOrValue&& getterOrValue) {
    using GetterOrValueType = std::decay_t<GetterOrValue>;

    if constexpr (std::is_invocable_v<GetterOrValueType, const ov::AnyMap&>) {
        return std::forward<GetterOrValue>(getterOrValue);
    } else {
        return [value = std::forward<GetterOrValue>(getterOrValue)](const ov::AnyMap&) {
            return value;
        };
    }
}

/**
 * @brief Register a simple property with custom getter.
 *
 * The property is read-only and always available.
 * Accepts either:
 * - a callable with signature compatible with ov::Any(const ov::AnyMap&)
 * - a plain value, which is returned as-is regardless of query-time arguments
 */
template <typename GetterOrValue>
inline void register_property_with_custom_function(PropertyMap& properties,
                                                   const std::string& propertyName,
                                                   bool isPublic,
                                                   ov::PropertyMutability mutability,
                                                   GetterOrValue&& getterOrValue) {
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          mutability,
                                          std::function<bool()>([]() {
                                              return true;
                                          }),
                                          std::function<ov::Any(const ov::AnyMap&)>(
                                              normalize_getter(std::forward<GetterOrValue>(getterOrValue)))});
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
                                                               bool isPublic,
                                                               ov::PropertyMutability mutability,
                                                               IsSupportedFn&& isSupported,
                                                               Getter&& getter) {
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          mutability,
                                          std::function<bool()>(std::forward<IsSupportedFn>(isSupported)),
                                          std::function<ov::Any(const ov::AnyMap&)>(std::forward<Getter>(getter))});
}

/**
 * @brief Register a property with support check, custom getter, and query-time arguments.
 *
 * Registers a property that is always added to the descriptor but gated by an `isSupported` condition at runtime.
 * The getter receives query-time arguments via ov::AnyMap.
 * Use this for properties that accept extra arguments at get_property call time.
 */
template <typename IsSupportedFn, typename Getter>
inline void register_property_with_support_custom_function_and_args(PropertyMap& properties,
                                                                    const std::string& propertyName,
                                                                    bool isPublic,
                                                                    ov::PropertyMutability mutability,
                                                                    IsSupportedFn&& isSupported,
                                                                    Getter&& getter) {
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          mutability,
                                          std::function<bool()>(std::forward<IsSupportedFn>(isSupported)),
                                          std::function<ov::Any(const ov::AnyMap&)>(std::forward<Getter>(getter))});
}

}  // namespace intel_npu
