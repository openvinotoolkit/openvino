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
 * @enum ov_performance_mode_e
 * @brief Enum to define possible performance mode hints
 * @brief This represents OpenVINO 2.0 ov::hint::PerformanceMode entity.
 *
 */
typedef enum {
    UNDEFINED_MODE = -1,        //!<  Undefined value, performance setting may vary from device to device
    LATENCY = 1,                //!<  Optimize for latency
    THROUGHPUT = 2,             //!<  Optimize for throughput
    CUMULATIVE_THROUGHPUT = 3,  //!< Optimize for cumulative throughput
} ov_performance_mode_e;

/**
 * @enum ov_affinity_e
 * @brief Enum to define possible affinity patterns
 */
typedef enum {
    NONE = -1,  //!<  Disable threads affinity pinning
    CORE = 0,   //!<  Pin threads to cores, best for static benchmarks
    NUMA = 1,   //!<  Pin threads to NUMA nodes, best for real-life, contented cases. On the Windows and MacOS* this
                //!<  option behaves as CORE
    HYBRID_AWARE = 2,  //!< Let the runtime to do pinning to the cores types, e.g. prefer the "big" cores for latency
                       //!< tasks. On the hybrid CPUs this option is default
} ov_affinity_e;

/**
 * @enum ov_hint_priority_e
 * @brief Enum to define possible priorities hints
 */
typedef enum {
    LOW = 0,           //!<  Low priority
    MEDIUM = 1,        //!<  Medium priority
    HIGH = 2,          //!<  High priority
    DEFAULT = MEDIUM,  //!<  Default priority is MEDIUM
} ov_hint_priority_e;

/**
 * @enum ov_log_level_e
 * @brief Enum to define possible log levels
 */
typedef enum {
    NO = -1,      //!< disable any logging
    ERR = 0,      //!< error events that might still allow the application to continue running
    WARNING = 1,  //!< potentially harmful situations which may further lead to ERROR
    INFO = 2,     //!< informational messages that display the progress of the application at coarse-grained level
    DEBUG = 3,    //!< fine-grained events that are most useful to debug an application.
    TRACE = 4,    //!< finer-grained informational events than the DEBUG
} ov_log_level_e;

/**
 * @brief property key
 */

//!<  Read-only property<char *> to get a string list of supported read-only properties.
OPENVINO_C_VAR(const char*) ov_property_key_supported_properties;

//!<  Read-only property<char *> to get a list of available device IDs
OPENVINO_C_VAR(const char*) ov_property_key_available_devices;

//!<  Read-only property<uint32_t> to get an unsigned integer value of optimaln
//!<  number of compiled model infer requests.
OPENVINO_C_VAR(const char*) ov_property_key_optimal_number_of_infer_requests;

//!<  Read-only property<unsigned int, unsigned int, unsigned int> to provide a
//!<  hint for a range for number of async infer requests. If device supports
//!<  streams, the metric provides range for number of IRs per stream.
OPENVINO_C_VAR(const char*) ov_property_key_range_for_async_infer_requests;

//!<  Read-only property<unsigned int, unsigned int> to provide information about a range for
//!<  streams on platforms where streams are supported
OPENVINO_C_VAR(const char*) ov_property_key_range_for_streams;

//!<  Read-only property<char *> to get a string value representing a full device name.
OPENVINO_C_VAR(const char*) ov_property_key_device_full_name;

//!<  Read-only property<char *> to get a string list of capabilities options per device.
OPENVINO_C_VAR(const char*) ov_property_key_device_capabilities;

//!<  Read-only property<char *> to get a name of name of a model
OPENVINO_C_VAR(const char*) ov_property_key_model_name;

//!<  Read-only property<uint32_t> to query information optimal batch size for the given device
//!<  and the network
OPENVINO_C_VAR(const char*) ov_property_key_optimal_batch_size;

//!<  Read-only property to get maximum batch size which does not cause performance degradation due
//!<  to memory swap impact.
OPENVINO_C_VAR(const char*) ov_property_key_max_batch_size;

//!<  Read-write property<char *> to set/get the directory which will be used to store any data cached
//!<  by plugins.
OPENVINO_C_VAR(const char*) ov_property_key_cache_dir;

//!<  Read-write property<uint32_t> to set/get the number of executor logical partitions.
OPENVINO_C_VAR(const char*) ov_property_key_num_streams;

//!<  Read-write property to set/get the name for setting CPU affinity per thread option.
OPENVINO_C_VAR(const char*) ov_property_key_affinity;

//!<  Read-write property<int32_t> to set/get the maximum number of threads that can be used
//!<  for inference tasks.
OPENVINO_C_VAR(const char*) ov_property_key_inference_num_threads;

//!< Read-write property<ov_performance_mode_e>, it is high-level OpenVINO Performance Hints
//!< unlike low-level properties that are individual (per-device), the hints are something that
//!< every device accepts and turns into device-specific settings detail see
//!< ov_performance_mode_e to get its hint's key name
OPENVINO_C_VAR(const char*) ov_property_key_hint_performance_mode;

//!< Read-write property<ov_element_type_e> to set the hint for device to use specified
//!< precision for inference
OPENVINO_C_VAR(const char*) ov_property_key_hint_inference_precision;

//!< (Optional) Read-write property<uint32_t> that backs the Performance Hints by giving
//!< additional information on how many inference requests the application will be
//!< keeping in flight usually this value comes from the actual use-case  (e.g.
//!< number of video-cameras, or other sources of inputs)
OPENVINO_C_VAR(const char*) ov_property_key_hint_num_requests;

//!< Read-write property<ov_log_level_e> for setting desirable log level.
OPENVINO_C_VAR(const char*) ov_property_key_log_level;

//!< Read-write property<ov_hint_priority_e>, high-level OpenVINO model priority hint.
//!< Defines what model should be provided with more performant bounded resource first.
OPENVINO_C_VAR(const char*) ov_property_key_hint_model_priority;

//!< Read-write property<bool> for setting performance counters option.
OPENVINO_C_VAR(const char*) ov_property_key_enable_profiling;

//!< Read-write property<std::pair<std::string, Any>>, device Priorities config option, with comma-separated devices
//!< listed in the desired priority
OPENVINO_C_VAR(const char*) ov_property_key_device_priorities;

/**
 * @brief property data type
 */
OPENVINO_C_VAR(const char*) ov_property_value_type_int32;
OPENVINO_C_VAR(const char*) ov_property_value_type_uint32;
OPENVINO_C_VAR(const char*) ov_property_value_type_bool;
OPENVINO_C_VAR(const char*) ov_property_value_type_enum;
OPENVINO_C_VAR(const char*) ov_property_value_type_ptr;
OPENVINO_C_VAR(const char*) ov_property_value_type_string;
OPENVINO_C_VAR(const char*) ov_property_value_type_float;
OPENVINO_C_VAR(const char*) ov_property_value_type_double;
OPENVINO_C_VAR(const char*) ov_property_value_type_map;
OPENVINO_C_VAR(const char*) ov_property_value_type_vector;

// Property
/**
 * @defgroup Property Property
 * @ingroup openvino_c
 * Set of functions representing of Property.
 * @{
 */

/** @} */  // end of Property
