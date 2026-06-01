// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <functional>
#include <map>
#include <string>
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

inline PropertyDescriptor make_property_descriptor(bool isPublic,
                                                   ov::PropertyMutability mutability,
                                                   std::function<ov::Any(const Config&)> get) {
    return PropertyDescriptor{isPublic, mutability, std::move(get)};
}

/**
 * @brief Register a simple property backed directly by a config option.
 *
 * Use this when the property name maps to an option stored in the provided config and
 * the getter is just `config.get<OPT_TYPE>()`.
 */
#define TRY_REGISTER_SIMPLE_PROPERTY(CONFIG, PROPERTIES, OPT_NAME, OPT_TYPE)                                 \
    do {                                                                                                      \
        std::string ov_prop_name = OPT_NAME.name();                                                           \
        if ((CONFIG).isAvailable(ov_prop_name)) {                                                             \
            bool ov_is_public = (CONFIG).getOpt(ov_prop_name).isPublic();                                     \
            ov::PropertyMutability ov_mutability = (CONFIG).getOpt(ov_prop_name).mutability();               \
            (PROPERTIES).emplace(ov_prop_name,                                                                \
                                 intel_npu::make_property_descriptor(ov_is_public, ov_mutability, [](const Config& config) { \
                                     return config.get<OPT_TYPE>();                                           \
                                 }));                                                                         \
        }                                                                                                     \
    } while (0)

/**
 * @brief Register a simple exposed NPUW option property.
 *
 * Equivalent to TRY_REGISTER_SIMPLE_PROPERTY, but uses `OPT_TYPE::key()` for the property name.
 */
#define TRY_REGISTER_NPUW_OPTION_PROPERTY(CONFIG, PROPERTIES, OPT_TYPE)                                      \
    do {                                                                                                      \
        std::string ov_prop_name = std::string(OPT_TYPE::key());                                              \
        if ((CONFIG).isAvailable(ov_prop_name)) {                                                             \
            bool ov_is_public = (CONFIG).getOpt(ov_prop_name).isPublic();                                     \
            ov::PropertyMutability ov_mutability = (CONFIG).getOpt(ov_prop_name).mutability();               \
            (PROPERTIES).emplace(ov_prop_name,                                                                \
                                 intel_npu::make_property_descriptor(ov_is_public, ov_mutability, [](const Config& config) { \
                                     return config.get<OPT_TYPE>();                                           \
                                 }));                                                                         \
        }                                                                                                     \
    } while (0)

/**
 * @brief Register a config-backed property with explicit public/private visibility.
 */
#define TRY_REGISTER_VARPUB_PROPERTY(CONFIG, PROPERTIES, OPT_NAME, OPT_TYPE, PROP_VISIBILITY)               \
    do {                                                                                                      \
        std::string ov_prop_name = OPT_NAME.name();                                                           \
        if ((CONFIG).isAvailable(ov_prop_name)) {                                                             \
            ov::PropertyMutability ov_mutability = (CONFIG).getOpt(ov_prop_name).mutability();               \
            (PROPERTIES).emplace(ov_prop_name,                                                                \
                                 intel_npu::make_property_descriptor(PROP_VISIBILITY, ov_mutability, [](const Config& config) { \
                                     return config.get<OPT_TYPE>();                                           \
                                 }));                                                                         \
        }                                                                                                     \
    } while (0)

/**
 * @brief Register a config-backed property with a custom getter callback.
 */
#define TRY_REGISTER_CUSTOMFUNC_PROPERTY(CONFIG, PROPERTIES, OPT_NAME, OPT_TYPE, ...)                        \
    do {                                                                                                      \
        std::string ov_prop_name = OPT_NAME.name();                                                           \
        if ((CONFIG).isAvailable(ov_prop_name)) {                                                             \
            bool ov_is_public = (CONFIG).getOpt(ov_prop_name).isPublic();                                     \
            ov::PropertyMutability ov_mutability = (CONFIG).getOpt(ov_prop_name).mutability();               \
            (PROPERTIES).emplace(ov_prop_name,                                                                \
                                 intel_npu::make_property_descriptor(ov_is_public, ov_mutability, __VA_ARGS__)); \
        }                                                                                                     \
    } while (0)

/**
 * @brief Register a custom property only when the property is available in the provided config.
 */
#define TRY_REGISTER_CUSTOM_PROPERTY(CONFIG, PROPERTIES, OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, ...) \
    do {                                                                                                      \
        std::string ov_prop_name = OPT_NAME.name();                                                           \
        if ((CONFIG).isAvailable(ov_prop_name)) {                                                             \
            (PROPERTIES).emplace(ov_prop_name,                                                                \
                                 intel_npu::make_property_descriptor(PROP_VISIBILITY, PROP_MUTABILITY, __VA_ARGS__)); \
        }                                                                                                     \
    } while (0)

/**
 * @brief Register a custom property unconditionally.
 */
#define FORCE_REGISTER_CUSTOM_PROPERTY(PROPERTIES, OPT_NAME, OPT_TYPE, PROP_VISIBILITY, PROP_MUTABILITY, ...) \
    do {                                                                                                      \
        (PROPERTIES).emplace(OPT_NAME.name(),                                                                 \
                             intel_npu::make_property_descriptor(PROP_VISIBILITY, PROP_MUTABILITY, __VA_ARGS__)); \
    } while (0)

/**
 * @brief Register a simple metric property.
 *
 * Metrics are read-only properties. Use this form when the return expression is a single value.
 */
#define REGISTER_SIMPLE_METRIC(PROPERTIES, PROP_NAME, PROP_VISIBILITY, PROP_RETVAL)                         \
    do {                                                                                                      \
        (PROPERTIES).emplace(PROP_NAME.name(),                                                                \
                             intel_npu::make_property_descriptor(PROP_VISIBILITY,                             \
                                                                 ov::PropertyMutability::RO,                  \
                                                                 [&](const Config& config) -> auto { return PROP_RETVAL; })); \
    } while (0)

/**
 * @brief Register a metric property with a custom callback.
 */
#define REGISTER_CUSTOM_METRIC(PROPERTIES, PROP_NAME, PROP_VISIBILITY, ...)                                  \
    do {                                                                                                      \
        (PROPERTIES).emplace(PROP_NAME.name(),                                                                \
                             intel_npu::make_property_descriptor(PROP_VISIBILITY, ov::PropertyMutability::RO, __VA_ARGS__)); \
    } while (0)

/**
 * @brief Register a compiled-model property backed directly by the current config.
 */
#define TRY_REGISTER_COMPILEDMODEL_PROPERTY(CONFIG, PROPERTIES, OPT_NAME, OPT_TYPE)                         \
    do {                                                                                                      \
        std::string ov_prop_name = OPT_NAME.name();                                                           \
        if ((CONFIG).isAvailable(ov_prop_name)) {                                                             \
            bool ov_is_public = (CONFIG).getOpt(ov_prop_name).isPublic();                                     \
            (PROPERTIES).emplace(ov_prop_name,                                                                \
                                 intel_npu::make_property_descriptor(ov_is_public, ov::PropertyMutability::RO, [](const Config& config) { \
                                     return config.get<OPT_TYPE>();                                           \
                                 }));                                                                         \
        }                                                                                                     \
    } while (0)

/**
 * @brief Register a compiled-model property only if it was explicitly set before compilation.
 *
 * Default option values are not materialized into the config, so this form advertises a property only when
 * the user or upper layer actually set it.
 */
#define TRY_REGISTER_COMPILEDMODEL_PROPERTY_IFSET(CONFIG, PROPERTIES, OPT_NAME, OPT_TYPE)                   \
    do {                                                                                                      \
        std::string ov_prop_name = OPT_NAME.name();                                                           \
        if ((CONFIG).has(ov_prop_name) && (CONFIG).isAvailable(ov_prop_name)) {                              \
            (PROPERTIES).emplace(ov_prop_name,                                                                \
                                 intel_npu::make_property_descriptor(true, ov::PropertyMutability::RO, [](const Config& config) { \
                                     return config.get<OPT_TYPE>();                                           \
                                 }));                                                                         \
        }                                                                                                     \
    } while (0)

}  // namespace intel_npu