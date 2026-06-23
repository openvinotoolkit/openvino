// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>
#include <type_traits>
#include <utility>

#include "openvino/runtime/properties.hpp"

namespace intel_npu {

class Config;

struct PropertyDescriptor final {
    bool isPublic;
    ov::PropertyMutability mutability;
    std::function<ov::Any(const Config&)> get;
    std::function<ov::Any(const Config&, const ov::AnyMap&)> getWithArgs;
};

using PropertyMap = std::map<std::string, PropertyDescriptor>;

/**
 * @brief Register a simple property backed directly by a config option.
 *
 * Use this when the property name maps to an option stored in the provided config and
 * the getter is just config.get<OPT_TYPE>().
 */
template <typename OptionType, typename ConfigLike>
inline void try_register_property_based_on_config(const ConfigLike& config,
                                                  PropertyMap& properties,
                                                  const std::string& propertyName) {
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    properties.emplace(propertyName,
                       PropertyDescriptor{option.isPublic(),
                                          option.mutability(),
                                          std::function<ov::Any(const Config&)>([](const Config& configValue) {
                                              return configValue.get<OptionType>();
                                          }),
                                          {}});
}

/**
 * @brief Register a config-backed property with explicit public/private visibility.
 *
 * Use this when the callback is the standard config.get<OPT_TYPE>() but the visibility (public/private)
 * must be determined at runtime rather than taken from the option descriptor.
 */
template <typename OptionType, typename ConfigLike>
inline void try_register_property_based_on_config_with_visibility(const ConfigLike& config,
                                                                  PropertyMap& properties,
                                                                  const std::string& propertyName,
                                                                  bool isPublic) {
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          option.mutability(),
                                          std::function<ov::Any(const Config&)>([](const Config& configValue) {
                                              return configValue.get<OptionType>();
                                          }),
                                          {}});
}

/**
 * @brief Register a custom property only when the property is available in the provided config.
 *
 * Use this when a custom function/implementation is required. Visibility and mutability are taken from the option
 * descriptor.
 */
template <typename ConfigLike, typename Getter>
inline void try_register_property_based_on_config_with_custom_function(const ConfigLike& config,
                                                                       PropertyMap& properties,
                                                                       const std::string& propertyName,
                                                                       Getter&& getter) {
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    properties.emplace(propertyName,
                       PropertyDescriptor{option.isPublic(),
                                          option.mutability(),
                                          std::function<ov::Any(const Config&)>(std::forward<Getter>(getter)),
                                          {}});
}

/**
 * @brief Register a compiled-model property backed directly by the current config.
 */
template <typename OptionType, typename ConfigLike>
inline void try_register_property_based_on_config_as_read_only(const ConfigLike& config,
                                                               PropertyMap& properties,
                                                               const std::string& propertyName) {
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    properties.emplace(propertyName,
                       PropertyDescriptor{option.isPublic(),
                                          ov::PropertyMutability::RO,
                                          std::function<ov::Any(const Config&)>([](const Config& configValue) {
                                              return configValue.get<OptionType>();
                                          }),
                                          {}});
}

/**
 * @brief Register a compiled-model property only if it was explicitly set before compilation.
 *
 * Default option values are not materialized into the config, so this form advertises a property only when
 * the user or upper layer actually set it.
 */
template <typename OptionType, typename ConfigLike>
inline void try_register_property_based_on_config_if_set_as_read_only(const ConfigLike& config,
                                                                      PropertyMap& properties,
                                                                      const std::string& propertyName) {
    if (!(config.has(propertyName) && config.isAvailable(propertyName))) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    properties.emplace(propertyName,
                       PropertyDescriptor{option.isPublic(),
                                          ov::PropertyMutability::RO,
                                          std::function<ov::Any(const Config&)>([](const Config& configValue) {
                                              return configValue.get<OptionType>();
                                          }),
                                          {}});
}

/**
 * @brief Register a simple exposed NPUW option property.
 *
 * Equivalent to try_register_property_based_on_config, but uses OPT_TYPE::key() for the property name.
 */
template <typename OptionType, typename ConfigLike>
inline void try_register_npuw_option_property(const ConfigLike& config, PropertyMap& properties) {
    const auto propertyName = std::string(OptionType::key());
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    properties.emplace(propertyName,
                       PropertyDescriptor{option.isPublic(),
                                          option.mutability(),
                                          std::function<ov::Any(const Config&)>([](const Config& configValue) {
                                              return configValue.get<OptionType>();
                                          }),
                                          {}});
}

template <typename GetterOrValue>
inline auto make_metric_getter(GetterOrValue&& getterOrValue) {
    using GetterOrValueType = std::decay_t<GetterOrValue>;

    if constexpr (std::is_invocable_v<GetterOrValueType, const Config&>) {
        return std::forward<GetterOrValue>(getterOrValue);
    } else {
        return [value = std::forward<GetterOrValue>(getterOrValue)](const Config&) {
            return value;
        };
    }
}

/**
 * @brief Register a simple metric property.
 *
 * Metrics are read-only properties. Accepts either a plain value or a getter lambda `(const Config&) -> T`.
 * Use this as the default form for metric registration.
 */
template <typename GetterOrValue>
inline void register_property_with_custom_function(PropertyMap& properties,
                                                   const std::string& propertyName,
                                                   bool isPublic,
                                                   GetterOrValue&& getterOrValue) {
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          ov::PropertyMutability::RO,
                                          std::function<ov::Any(const Config&)>(
                                              make_metric_getter(std::forward<GetterOrValue>(getterOrValue))),
                                          {}});
}

/**
 * @brief Conditionally register a metric property with a standard (Config-only) custom function.
 *
 * Registers the property only when @p shouldRegister is true.
 * Use this when a metric property must be conditionally exposed based on device/backend capabilities.
 */
template <typename Getter>
inline void try_register_property_with_custom_function(PropertyMap& properties,
                                                       const std::string& propertyName,
                                                       bool shouldRegister,
                                                       bool isPublic,
                                                       Getter&& getter) {
    if (!shouldRegister) {
        return;
    }
    properties.emplace(propertyName,
                       PropertyDescriptor{isPublic,
                                          ov::PropertyMutability::RO,
                                          std::function<ov::Any(const Config&)>(std::forward<Getter>(getter)),
                                          {}});
}

/**
 * @brief Conditionally register a metric property whose custom function requires additional AnyMap arguments.
 *
 * Registers the property only when @p shouldRegister is true.
 * Use this for properties that accept extra arguments at get_property call time (e.g. compatibility_check).
 */
template <typename Getter>
inline void try_register_property_with_custom_function_and_args(PropertyMap& properties,
                                                                const std::string& propertyName,
                                                                bool shouldRegister,
                                                                bool isPublic,
                                                                Getter&& getter) {
    if (!shouldRegister) {
        return;
    }
    properties.emplace(
        propertyName,
        PropertyDescriptor{isPublic,
                           ov::PropertyMutability::RO,
                           {},
                           std::function<ov::Any(const Config&, const ov::AnyMap&)>(std::forward<Getter>(getter))});
}

}  // namespace intel_npu
