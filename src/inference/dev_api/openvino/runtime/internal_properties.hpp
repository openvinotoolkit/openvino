// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for internal properties that are passed from one plugin to another
 * @file openvino/runtime/internal_properties.hpp
 */

#pragma once

#include <variant>

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/threading/istreams_executor.hpp"

namespace ov {

namespace internal {
/**
 * @brief Read-only property to get a std::vector<PropertyName> of supported internal properties.
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<std::vector<PropertyName>, PropertyMutability::RO> supported_properties{
    "INTERNAL_SUPPORTED_PROPERTIES"};

/**
 * @brief Read-only property to get a std::vector<PropertyName> of properties
 * which should affect the hash calculation for model cache
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::vector<PropertyName>, PropertyMutability::RO> caching_properties{"CACHING_PROPERTIES"};

/**
 * @brief Read-only property to get a std::vector<PropertyName> of properties
 * which should affect the loading time from cache
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RO> caching_with_mmap{"CACHING_WITH_MMAP"};

/**
 * @brief Allow to create exclusive_async_requests with one executor
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RW> exclusive_async_requests{"EXCLUSIVE_ASYNC_REQUESTS"};

/**
 * @brief the property for setting of required device for which config to be updated
 * values: device id starts from "0" - first device, "1" - second device, etc
 * note: plugin may have different devices naming convention
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::string, PropertyMutability::WO> config_device_id{"CONFIG_DEVICE_ID"};

/**
 * @brief Limit \#threads that are used by IStreamsExecutor to execute `parallel_for` calls
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<int32_t, PropertyMutability::RW> threads_per_stream{"THREADS_PER_STREAM"};

/**
 * @brief It contains compiled_model_runtime_properties information to make plugin runtime can check whether it is
 * compatible with the cached compiled model, the result is returned by get_property() calling.
 *
 * The information details are defined by plugin itself, each plugin may require different runtime contents.
 * For example, CPU plugin will contain OV version, while GPU plugin will contain OV and GPU driver version, etc.
 * Core doesn't understand its content and only read it from plugin and write it into blob header.
 *
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<std::string, PropertyMutability::RO> compiled_model_runtime_properties{
    "COMPILED_MODEL_RUNTIME_PROPERTIES"};

/**
 * @brief Check whether the attached compiled_model_runtime_properties is supported by this device runtime.
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RO> compiled_model_runtime_properties_supported{
    "COMPILED_MODEL_RUNTIME_PROPERTIES_SUPPORTED"};

/**
 * @brief Read-write property to set the percentage of the estimated model size which is used to determine the query
 * model results for further processing
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<float, PropertyMutability::RW> query_model_ratio{"QUERY_MODEL_RATIO"};

/**
 * @brief Allow execution of low precision transformations in plugin's pipelines
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RW> enable_lp_transformations{"LP_TRANSFORMS_MODE"};

/**
 * @brief Read-only property to get plugin specific needed alignment for cache header
 * @ingroup ov_runtime_cpp_prop_api
 */
static constexpr Property<uint32_t, PropertyMutability::RO> cache_header_alignment{"CACHE_HEADER_ALIGNMENT"};

/**
 * @brief Enum to define possible cache quant schema hints.
 */
enum class CacheQuantMode { AUTO = 0, BY_CHANNEL = 1, BY_TOKEN = 2 };

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const CacheQuantMode& mode) {
    switch (mode) {
    case CacheQuantMode::AUTO:
        return os << "AUTO";
    case CacheQuantMode::BY_CHANNEL:
        return os << "BY_CHANNEL";
    case CacheQuantMode::BY_TOKEN:
        return os << "BY_TOKEN";
    default:
        OPENVINO_THROW("Unsupported cache quant mode");
    }
}

inline std::istream& operator>>(std::istream& is, CacheQuantMode& mode) {
    std::string str;
    is >> str;
    if (str == "AUTO") {
        mode = CacheQuantMode::AUTO;
    } else if (str == "BY_CHANNEL") {
        mode = CacheQuantMode::BY_CHANNEL;
    } else if (str == "BY_TOKEN") {
        mode = CacheQuantMode::BY_TOKEN;
    } else {
        OPENVINO_THROW("Unsupported cache quant mode: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief Define quantization mode for key cache. Group size decision policy may be different per plugin.
 * @param AUTO - Default mode decided by plugin.
 * @param BY_CHANNEL - Quantize key cache by channel dimension.
 * @param BY_TOKEN - Quantize key cache by token dimension.

    By channel: Quantize along with tokens in each channel
    ┌──┬──┬─────────────────────────┐
    │  │  │                         │ token[0]
    │  │  │                         │ token[1]
    │  │  │  ...                    │ token[2]
    │  │  │                         │ ..
    │  │  │                         │ token[seq_len - 1]
    └──┴──┴─────────────────────────┘
    c[0] c[1] ...    c[head_size - 1]

    By token: Quantize along with channel dim per each token
    ┌───────────────────────────────┐
    ├───────────────────────────────┤ token[0]
    ├───────────────────────────────┤ token[1]
    │                               │ ...
    │                               │
    │                               │
    │                               │ token[seq_len - 1]
    └───────────────────────────────┘
    c[0] c[1] ...    c[head_size - 1]
 */

static constexpr Property<CacheQuantMode, PropertyMutability::RW> key_cache_quant_mode{"KEY_CACHE_QUANT_MODE"};

/**
 * @brief Define quantization mode for value cache. Group size decision policy may be different per plugin.
 * @param AUTO - Default mode decided by plugin
 * @param BY_CHANNEL - Quantize value cache by channel dimension.
 * @param BY_TOKEN - Quantize key cache by token dimension.
 */

static constexpr Property<CacheQuantMode, PropertyMutability::RW> value_cache_quant_mode{"VALUE_CACHE_QUANT_MODE"};

// * If ENABLE_WEIGHTLESS:false                                                                 -> return false
// * If no ENABLE_WEIGHTLESS and (CACHE_DIR and (CACHE_MODE:OPTIMIZE_SPEED or no CACHE_MODE)    -> return false
// * If no ENABLE_WEIGHTLESS and (CACHE_DIR and CACHE_MODE:OPTIMIZE_SIZE)                       -> return true
// * If ENABLE_WEIGHTLESS:true                                                                  -> return true

//   in current version plugins may not support WL for different than CACHE_MODE:OPTIMIZE_SIZE
//   in any case if there is no cache mode the WL should be supported for export/import but no cache entry
//   if any configuration si not supported or may brake plugin behaviour is should throw. But the unsupported
//   configuration can implemented in next release(s)
inline bool is_weightless_enabled(const ov::AnyMap& config) {
    if (auto it = config.find(ov::enable_weightless.name()); it != config.end()) {
        return it->second.as<bool>();
    } else if (auto cache_mode_it = config.find(ov::cache_mode.name()); cache_mode_it != config.end()) {
        return cache_mode_it->second.as<ov::CacheMode>() == ov::CacheMode::OPTIMIZE_SIZE;
    }
    return false;
}

using WeightlessHint = std::variant<std::filesystem::path, std::shared_ptr<const ov::Model>, ov::Tensor>;

// Extract weightless hint from config as variant which can be used by plugin to apply specific logic to restore weights
inline WeightlessHint get_weightless_hint(const ov::AnyMap& config) {
    WeightlessHint hint;
    if (auto it = config.find(ov::weights_path.name()); it != config.end()) {
        hint = std::filesystem::path(it->second.as<std::string>());
    } else if (auto it = config.find(ov::hint::model.name()); it != config.end()) {
        hint = it->second.as<std::shared_ptr<const ov::Model>>();
    } else if (auto it = config.find(ov::hint::compiled_blob.name()); it != config.end()) {
        hint = it->second.as<Tensor>();
    }
    return hint;
}
}  // namespace internal
}  // namespace ov
