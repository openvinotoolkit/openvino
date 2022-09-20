// Copyright (C) 2018-2022 Intel Corporation
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
 * @brief property key
 */

//!< Read-only property<string> to get a string list of supported read-only properties.
OPENVINO_C_VAR(const char*) ov_property_key_supported_properties;

//!< Read-only property<string> to get a list of available device IDs
OPENVINO_C_VAR(const char*) ov_property_key_available_devices;

//!< Read-only property<uint32_t string> to get an unsigned integer value of optimaln
//!< number of compiled model infer requests.
OPENVINO_C_VAR(const char*) ov_property_key_optimal_number_of_infer_requests;

//!< Read-only property<string(unsigned int, unsigned int, unsigned int)> to provide a
//!< hint for a range for number of async infer requests. If device supports
//!< streams, the metric provides range for number of IRs per stream.
OPENVINO_C_VAR(const char*) ov_property_key_range_for_async_infer_requests;

//!< Read-only property<string(unsigned int, unsigned int)> to provide information about a range for
//!< streams on platforms where streams are supported
OPENVINO_C_VAR(const char*) ov_property_key_range_for_streams;

//!< Read-only property<string> to get a string value representing a full device name.
OPENVINO_C_VAR(const char*) ov_property_key_device_full_name;

//!< Read-only property<string> to get a string list of capabilities options per device.
OPENVINO_C_VAR(const char*) ov_property_key_device_capabilities;

//!< Read-only property<string> to get a name of name of a model
OPENVINO_C_VAR(const char*) ov_property_key_model_name;

//!< Read-only property<uint32_t string> to query information optimal batch size for the given device
//!< and the network
OPENVINO_C_VAR(const char*) ov_property_key_optimal_batch_size;

//!< Read-only property to get maximum batch size which does not cause performance degradation due
//!< to memory swap impact.
OPENVINO_C_VAR(const char*) ov_property_key_max_batch_size;

//!< Read-write property<string> to set/get the directory which will be used to store any data cached
//!< by plugins.
OPENVINO_C_VAR(const char*) ov_property_key_cache_dir;

//!< Read-write property<uint32_t string> to set/get the number of executor logical partitions.
OPENVINO_C_VAR(const char*) ov_property_key_num_streams;

//!< Read-write property to set/get the name for setting CPU affinity per thread option.
//!< All property values are string format, below are optional value:
//!<    "NONE" - Disable threads affinity pinning
//!<    "CORE" - Pin threads to cores, best for static benchmarks
//!<    "NUMA" - Pin threads to NUMA nodes, best for real-life, contented cases. On the Windows and MacOS* this
//!< option behaves as CORE
//!<    "HYBRID_AWARE" - Let the runtime to do pinning to the cores types, e.g. prefer the "big" cores for latency
//!< tasks. On the hybrid CPUs this option is default.
OPENVINO_C_VAR(const char*) ov_property_key_affinity;

//!<  Read-write property<int32_t string> to set/get the maximum number of threads that can be used
//!<  for inference tasks.
OPENVINO_C_VAR(const char*) ov_property_key_inference_num_threads;

//!< Read-write property, it is high-level OpenVINO Performance Hints
//!< unlike low-level properties that are individual (per-device), the hints are something that
//!< every device accepts and turns into device-specific settings detail.
//!< All property values are string, below are optional value:
//!<   "UNDEFINED_MODE"  - Undefined value, performance setting may vary from device to device
//!<   "LATENCY"         - Optimize for latency
//!<   "THROUGHPUT"      - Optimize for throughput
//!<   "CUMULATIVE_THROUGHPUT" - Optimize for cumulative throughput
OPENVINO_C_VAR(const char*) ov_property_key_hint_performance_mode;

//!< Read-write property<ov_element_type_e> to set the hint for device to use specified
//!< precision for inference
//!< All property values are string, below are optional value:
//!<    "UNDEFINED" -  Undefined element type
//!<    "DYNAMIC"   - Dynamic element type
//!<    "BOOLEAN"   - boolean element type
//!<    "BF16"      - bf16 element type
//!<    "F16"       - f16 element type
//!<    "F32"       - f32 element type
//!<    "F64"       - f64 element type
//!<    "I4"        - i4 element type
//!<    "I8"        - i8 element type
//!<    "I16"       - i16 element type
//!<    "I32"       - i32 element type
//!<    "I64"       - i64 element type
//!<    "U1"        - binary element type
//!<    "U4"        - u4 element type
//!<    "U8"        - u8 element type
//!<    "U16"       - u16 element type
//!<    "U32"       - u32 element type
//!<    "U64"       - u64 element type
OPENVINO_C_VAR(const char*) ov_property_key_hint_inference_precision;

//!< (Optional) Read-write property<uint32_t string> that backs the Performance Hints by giving
//!< additional information on how many inference requests the application will be
//!< keeping in flight usually this value comes from the actual use-case  (e.g.
//!< number of video-cameras, or other sources of inputs)
OPENVINO_C_VAR(const char*) ov_property_key_hint_num_requests;

//!< Read-write property<string> for setting desirable log level.
//!< All property values are string, below are optional value:
//!<    "NO"       - disable any logging
//!<    "ERR"      - error events that might still allow the application to continue running
//!<    "WARNING"  - potentially harmful situations which may further lead to ERROR
//!<    "INFO"     - informational messages that display the progress of the application at coarse-grained level
//!<    "DEBUG"    - fine-grained events that are most useful to debug an application.
//!<    "TRACE"    - finer-grained informational events than the DEBUG
OPENVINO_C_VAR(const char*) ov_property_key_log_level;

//!< Read-write property, high-level OpenVINO model priority hint.
//!< Defines what model should be provided with more performant bounded resource first.
//!< All property value are string format, below is optional value:
//!<    "LOW"     - Low priority
//!<    "MEDIUM"  - Medium priority
//!<    "HIGH"    - High priority
//!<    "DEFAULT" - Default priority is MEDIUM
OPENVINO_C_VAR(const char*) ov_property_key_hint_model_priority;

//!< Read-write property<string> for setting performance counters option.
//!< All property values are string, below are optional value:
//!<     "YES"   - true
//!<     "NO"    - false
OPENVINO_C_VAR(const char*) ov_property_key_enable_profiling;

//!< Read-write property<std::pair<std::string, Any>>, device Priorities config option, with comma-separated devices
//!< listed in the desired priority
//!< Some optional values for MULTI device:
//!    "CPU,GPU"
//!    "GPU,CPU"
//!  Note: CPU plugin is not implement.
OPENVINO_C_VAR(const char*) ov_property_key_device_priorities;

// Property
/**
 * @defgroup Property Property
 * @ingroup openvino_c
 * Set of functions representing of Property.
 * @{
 */

/** @} */  // end of Property
