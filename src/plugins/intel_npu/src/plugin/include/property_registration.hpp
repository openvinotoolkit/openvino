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
};

using PropertyMap = std::map<std::string, PropertyDescriptor>;

template <typename Getter>
inline PropertyDescriptor make_property_descriptor(bool isPublic, ov::PropertyMutability mutability, Getter&& get) {
    return PropertyDescriptor{isPublic,
                              mutability,
                              std::function<ov::Any(const Config&)>(std::forward<Getter>(get))};
}

template <typename Getter>
inline void register_named_property(PropertyMap& properties,
                                    std::string propertyName,
                                    bool isPublic,
                                    ov::PropertyMutability mutability,
                                    Getter&& getter) {
    properties.emplace(std::move(propertyName),
                       make_property_descriptor(isPublic, mutability, std::forward<Getter>(getter)));
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
 * @brief Register a simple property backed directly by a config option.
 *
 * Use this when the property name maps to an option stored in the provided config and
 * the getter is just config.get<OPT_TYPE>().
 */
template <typename OptionType, typename ConfigLike, typename Property>
inline void try_register_simple_property(const ConfigLike& config, PropertyMap& properties, const Property& property) {
    const auto propertyName = std::string(property.name());
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    register_named_property(properties,
                            propertyName,
                            option.isPublic(),
                            option.mutability(),
                            [](const Config& configValue) {
                                return configValue.get<OptionType>();
                            });
}

/**
 * @brief Register a simple exposed NPUW option property.
 *
 * Equivalent to try_register_simple_property, but uses OPT_TYPE::key() for the property name.
 */
template <typename OptionType, typename ConfigLike>
inline void try_register_npuw_option_property(const ConfigLike& config, PropertyMap& properties) {
    const auto propertyName = std::string(OptionType::key());
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    register_named_property(properties,
                            propertyName,
                            option.isPublic(),
                            option.mutability(),
                            [](const Config& configValue) {
                                return configValue.get<OptionType>();
                            });
}

/**
 * @brief Register a config-backed property with explicit public/private visibility.
 */
template <typename OptionType, typename ConfigLike, typename Property>
inline void try_register_varpub_property(const ConfigLike& config,
                                         PropertyMap& properties,
                                         const Property& property,
                                         bool isPublic) {
    const auto propertyName = std::string(property.name());
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    register_named_property(properties,
                            propertyName,
                            isPublic,
                            option.mutability(),
                            [](const Config& configValue) {
                                return configValue.get<OptionType>();
                            });
}

/**
 * @brief Register a config-backed property with a custom getter callback.
 */
template <typename ConfigLike, typename Property, typename Getter>
inline void try_register_customfunc_property(const ConfigLike& config,
                                             PropertyMap& properties,
                                             const Property& property,
                                             Getter&& getter) {
    const auto propertyName = std::string(property.name());
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    register_named_property(properties,
                            propertyName,
                            option.isPublic(),
                            option.mutability(),
                            std::forward<Getter>(getter));
}

/**
 * @brief Register a custom property only when the property is available in the provided config.
 */
template <typename ConfigLike, typename Property, typename Getter>
inline void try_register_custom_property(const ConfigLike& config,
                                         PropertyMap& properties,
                                         const Property& property,
                                         bool isPublic,
                                         ov::PropertyMutability mutability,
                                         Getter&& getter) {
    const auto propertyName = std::string(property.name());
    if (!config.isAvailable(propertyName)) {
        return;
    }

    register_named_property(properties,
                            propertyName,
                            isPublic,
                            mutability,
                            std::forward<Getter>(getter));
}

/**
 * @brief Register a custom property unconditionally.
 */
template <typename Property, typename Getter>
inline void force_register_custom_property(PropertyMap& properties,
                                           const Property& property,
                                           bool isPublic,
                                           ov::PropertyMutability mutability,
                                           Getter&& getter) {
    register_named_property(properties,
                            std::string(property.name()),
                            isPublic,
                            mutability,
                            std::forward<Getter>(getter));
}

/**
 * @brief Register a simple metric property.
 *
 * Metrics are read-only properties. Use this form when the return expression is a single value.
 */
template <typename Property, typename GetterOrValue>
inline void register_simple_metric(PropertyMap& properties,
                                   const Property& property,
                                   bool isPublic,
                                   GetterOrValue&& getterOrValue) {
    register_named_property(properties,
                            std::string(property.name()),
                            isPublic,
                            ov::PropertyMutability::RO,
                            make_metric_getter(std::forward<GetterOrValue>(getterOrValue)));
}

/**
 * @brief Register a metric property with a custom callback.
 */
template <typename Property, typename Getter>
inline void register_custom_metric(PropertyMap& properties,
                                   const Property& property,
                                   bool isPublic,
                                   Getter&& getter) {
    register_named_property(properties,
                            std::string(property.name()),
                            isPublic,
                            ov::PropertyMutability::RO,
                            std::forward<Getter>(getter));
}

/**
 * @brief Register a compiled-model property backed directly by the current config.
 */
template <typename OptionType, typename ConfigLike, typename Property>
inline void try_register_compiled_model_property(const ConfigLike& config,
                                                 PropertyMap& properties,
                                                 const Property& property) {
    const auto propertyName = std::string(property.name());
    if (!config.isAvailable(propertyName)) {
        return;
    }

    const auto& option = config.getOpt(propertyName);
    register_named_property(properties,
                            propertyName,
                            option.isPublic(),
                            ov::PropertyMutability::RO,
                            [](const Config& configValue) {
                                return configValue.get<OptionType>();
                            });
}

/**
 * @brief Register a compiled-model property only if it was explicitly set before compilation.
 *
 * Default option values are not materialized into the config, so this form advertises a property only when
 * the user or upper layer actually set it.
 */
template <typename OptionType, typename ConfigLike, typename Property>
inline void try_register_compiled_model_property_ifset(const ConfigLike& config,
                                                       PropertyMap& properties,
                                                       const Property& property) {
    const auto propertyName = std::string(property.name());
    if (!(config.has(propertyName) && config.isAvailable(propertyName))) {
        return;
    }

    register_named_property(properties,
                            propertyName,
                            true,
                            ov::PropertyMutability::RO,
                            [](const Config& configValue) {
                                return configValue.get<OptionType>();
                            });
}

}  // namespace intel_npu
