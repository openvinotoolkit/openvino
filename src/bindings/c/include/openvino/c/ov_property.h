// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is header file for ov_property C API.
 * A header for advanced hardware specific properties for OpenVINO runtime devices.
 * To use in set_property, compile_model, import_model, get_property methods.
 * @file ov_property.h
 */

#pragma once

#include "openvino/c/ov_common.h"

/**
 * @brief Read-only property<string> to get a string list of supported read-only properties.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_supported_properties;

/**
 * @brief Read-only property<string> to get a list of available device IDs.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_available_devices;

/**
 * @brief Read-only property<uint32_t string> to get an unsigned integer value of optimaln
 * number of compiled model infer requests.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_optimal_number_of_infer_requests;

/**
 * @brief Read-only property<string(unsigned int, unsigned int, unsigned int)> to provide a
 * hint for a range for number of async infer requests. If device supports
 * streams, the metric provides range for number of IRs per stream.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_range_for_async_infer_requests;

/**
 * @brief Read-only property<string(unsigned int, unsigned int)> to provide information about a range for
 * streams on platforms where streams are supported
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_range_for_streams;

/**
 * @brief Read-only property<string> to get a string value representing a full device name.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_device_full_name;

/**
 * @brief Read-only property<string> to get a string list of capabilities options per device.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_device_capabilities;

/**
 * @brief Read-only property<string> to get a name of name of a model
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_model_name;

/**
 * @brief Read-only property<uint32_t string> to query information optimal batch size for the given device
 * and the network
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_optimal_batch_size;

/**
 * @brief Read-only property to get maximum batch size which does not cause performance degradation due
 * to memory swap impact.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_max_batch_size;

/**
 * @brief Read-write property<string> to set/get the directory which will be used to store any data cached
 * by plugins.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_cache_dir;

/**
 * @brief Read-write property<string> to select the cache mode between optimize_size and optimize_speed.
 * If optimize_size is selected(default), smaller cache files will be created.
 * If optimize_speed is selected, loading time will decrease but the cache file size will increase.
 * This is only supported from GPU.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_cache_mode;

/**
 * @brief Write-only property<ov_encryption_callbacks*> to set encryption and decryption function for model cache.
 * If ov_property_key_cache_encryption_callbacks is set, model topology will be encrypted when saving to the cache and
 * decrypted when loading from the cache. This property is set in ov_core_compile_model_* only
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_cache_encryption_callbacks;

/**
 * @brief Read-write property<uint32_t string> to set/get the number of executor logical partitions.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_num_streams;

/**
 * @brief Read-write property<int32_t string> to set/get the maximum number of threads that can be used
 * for inference tasks.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_inference_num_threads;

/**
 * @brief Read-write property, it is high-level OpenVINO hint for using CPU pinning to bind CPU threads to processors
 * during inference
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_enable_cpu_pinning;

/**
 * @brief Read-write property, it is high-level OpenVINO hint for using hyper threading processors during CPU inference
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_enable_hyper_threading;

/**
 * @brief Read-write property, it is high-level OpenVINO Performance Hints
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_performance_mode;

/**
 * @brief Read-write property, it is high-level OpenVINO Hints for the type of CPU core used during inference
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_scheduling_core_type;

/**
 * @brief Read-write property<ov_element_type_e> to set the hint for device to use specified precision for inference.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_inference_precision;

/**
 * @brief (Optional) Read-write property<uint32_t string> that backs the Performance Hints by giving
 * additional information on how many inference requests the application will be
 * keeping in flight usually this value comes from the actual use-case  (e.g.
 * number of video-cameras, or other sources of inputs)
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_num_requests;

/**
 * @brief Read-write property<string> for setting desirable log level.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_log_level;

/**
 * @brief Read-write property, high-level OpenVINO model priority hint.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_model_priority;

/**
 * @brief Read-write property<string> for setting performance counters option.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_enable_profiling;

/**
 * @brief Read-write property<std::pair<std::string, Any>>, device Priorities config option,
 * with comma-separated devices listed in the desired priority
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_device_priorities;

/**
 * @brief Read-write property<string> for high-level OpenVINO Execution hint
 * unlike low-level properties that are individual (per-device), the hints are something that every device accepts
 * and turns into device-specific settings
 * Execution mode hint controls preferred optimization targets (performance or accuracy) for given model
 * It can be set to be below value:
 *   "PERFORMANCE",  //!<  Optimize for max performance
 *   "ACCURACY",     //!<  Optimize for max accuracy
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_hint_execution_mode;

/**
 * @brief Read-write property to set whether force terminate tbb when ov core destruction
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_force_tbb_terminate;

/**
 * @brief Read-write property to configure `mmap()` use for model read
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_enable_mmap;

/**
 * @brief Read-write property
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_auto_batch_timeout;

/**
 * @brief Read-write property to configure config file for GPU
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_gpu_config_file;

/**
 * @struct ov_property_t
 * @ingroup ov_property_c_api
 * @brief A key/value property pair for use with the non-variadic @c _props API.
 *
 * The @c value field is a @c const void* that holds any property value. The library
 * interprets the value based on the @c key — string-valued properties pass a @c const char*,
 * GPU/OCL handle properties pass the raw handle pointer, and cache-encryption callbacks pass
 * a pointer to an @c ov_encryption_callbacks struct. This mirrors the existing variadic API where
 * the key determines the expected argument type. The struct size is exactly two pointers (16 bytes
 * on 64-bit) and will never change, making it fully ABI-stable across library versions.
 *
 * In C all pointer types implicitly convert to @c const void*, so no casts are needed:
 * @code
 *   ov_encryption_callbacks cb = {my_encrypt, my_decrypt};
 *   ov_property_t props[] = {
 *       {ov_property_key_hint_performance_mode, "LATENCY"},          // const char* → const void*
 *       {ov_property_key_num_streams,           "4"},
 *       {ov_property_key_cache_encryption_callbacks, &cb},           // struct pointer → const void*
 *   };
 *   ov_core_compile_model_from_file_props(core, "model.xml", "CPU", 3, props, &cm);
 *
 *   // GPU context with a raw OCL handle (any pointer → const void*):
 *   ov_property_t ctx_props[] = {
 *       {ov_property_key_intel_gpu_context_type, "OCL"},
 *       {ov_property_key_intel_gpu_ocl_context,  ocl_context_ptr},
 *   };
 *   ov_core_create_context_props(core, "GPU", 2, ctx_props, &ctx);
 * @endcode
 */
typedef struct {
    const char* key;   /*!< Property key string (see @c ov_property_key_xxx constants). */
    const void* value; /*!< Property value: @c const char* string, GPU/OCL handle, or @c ov_encryption_callbacks*. */
} ov_property_t;
