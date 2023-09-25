// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for internal properties that are passed from one plugin to another
 * @file openvino/runtime/internal_properties.hpp
 */

#pragma once

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
 * @brief Allow low precision transform
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RW> lp_transforms_mode{"LP_TRANSFORMS_MODE"};

/** @cond INTERNAL */
inline std::ostream& operator<<(std::ostream& os, const ov::threading::IStreamsExecutor::ThreadBindingType& type) {
    switch (type) {
    case ov::threading::IStreamsExecutor::NONE:
        return os << "NONE";
    case ov::threading::IStreamsExecutor::CORES:
        return os << "CORES";
    case ov::threading::IStreamsExecutor::NUMA:
        return os << "NUMA";
    case ov::threading::IStreamsExecutor::HYBRID_AWARE:
        return os << "HYBRID_AWARE";
    default:
        OPENVINO_THROW("Unsupported thread binding type value");
    }
}

inline std::istream& operator>>(std::istream& is, ov::threading::IStreamsExecutor::ThreadBindingType& type) {
    std::string str;
    is >> str;
    if (str == "NONE") {
        type = ov::threading::IStreamsExecutor::NONE;
    } else if (str == "CORES") {
        type = ov::threading::IStreamsExecutor::CORES;
    } else if (str == "NUMA") {
        type = ov::threading::IStreamsExecutor::NUMA;
    } else if (str == "HYBRID_AWARE") {
        type = ov::threading::IStreamsExecutor::HYBRID_AWARE;
    } else {
        OPENVINO_THROW("Unsupported thread binding type: ", str);
    }
    return is;
}
/** @endcond */

/**
 * @brief The name for setting CPU affinity per thread option.
 *
 * It is passed to Core::get_property()
 *
 * The following options are implemented only for the TBB as a threading option
 * ov::threading::IStreamsExecutor::ThreadBindingType::NUMA (pinning threads to NUMA nodes, best for real-life,
 * contented cases) on the Windows and MacOS* this option behaves as YES
 * ov::threading::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE (let the runtime to do pinning to the cores types,
 * e.g. prefer the "big" cores for latency tasks) on the hybrid CPUs this option is default
 *
 * Also, the settings are ignored, if the OpenVINO compiled with OpenMP and any affinity-related OpenMP's
 * environment variable is set (as affinity is configured explicitly)
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<ov::threading::IStreamsExecutor::ThreadBindingType, PropertyMutability::RW> cpu_bind_thread{
    "CPU_BIND_THREAD"};

/**
 * @brief Number of streams in Performance-core(big core)
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<size_t, PropertyMutability::RW> big_core_streams{"BIG_CORE_STREAMS"};

/**
 * @brief Number of streams in Efficient-core(small core) on hybrid cores machine
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<size_t, PropertyMutability::RW> small_core_streams{"SMALL_CORE_STREAMS"};

/**
 * @brief Number of threads per stream in big cores
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<size_t, PropertyMutability::RW> threads_per_stream_big{"THREADS_PER_STREAM_BIG"};

/**
 * @brief Number of threads per stream in small cores on hybrid cores machine
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<size_t, PropertyMutability::RW> threads_per_stream_small{"THREADS_PER_STREAM_SMALL"};

/**
 * @brief Small core start offset when binding cpu cores
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<size_t, PropertyMutability::RW> small_core_offset{"SMALL_CORE_OFFSET"};

/**
 * @brief Enable hyper thread
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<bool, PropertyMutability::RW> enable_hyper_thread{"ENABLE_HYPER_THREAD"};

/**
 * @brief Limit \#threads that are used by IStreamsExecutor to execute `parallel_for` calls
 * @ingroup ov_dev_api_plugin_api
 */
static constexpr Property<size_t, PropertyMutability::RW> threads_per_stream{"THREADS_PER_STREAM"};

}  // namespace internal
OPENVINO_DEPRECATED(
    "This property is deprecated and will be removed soon. Use ov::internal::caching_properties instead of it.")
constexpr auto caching_properties = internal::caching_properties;
OPENVINO_DEPRECATED(
    "This property is deprecated and will be removed soon. Use ov::internal::exclusive_async_requests instead of it.")
constexpr auto exclusive_async_requests = internal::exclusive_async_requests;
}  // namespace ov
