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

typedef struct ov_property {
    const char* key;
    ov_any_t value;
} ov_property_t;

typedef struct ov_properties {
    ov_property_t* list;
    size_t size;
} ov_properties_t;

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

OPENVINO_C_VAR(const char*) ov_property_key_supported_properties;
OPENVINO_C_VAR(const char*) ov_property_key_available_devices;
OPENVINO_C_VAR(const char*) ov_property_key_optimal_number_of_infer_requests;
OPENVINO_C_VAR(const char*) ov_property_key_range_for_async_infer_requests;
OPENVINO_C_VAR(const char*) ov_property_key_range_for_streams;
OPENVINO_C_VAR(const char*) ov_property_key_device_full_name;
OPENVINO_C_VAR(const char*) ov_property_key_device_capabilities;
OPENVINO_C_VAR(const char*) ov_property_key_cache_dir;
OPENVINO_C_VAR(const char*) ov_property_key_num_streams;
OPENVINO_C_VAR(const char*) ov_property_key_affinity;
OPENVINO_C_VAR(const char*) ov_property_key_inference_num_threads;
OPENVINO_C_VAR(const char*) ov_property_key_hint_performance_mode;
OPENVINO_C_VAR(const char*) ov_property_key_model_name;
OPENVINO_C_VAR(const char*) ov_property_key_hint_inference_precision;
OPENVINO_C_VAR(const char*) ov_property_key_optimal_batch_size;
OPENVINO_C_VAR(const char*) ov_property_key_max_batch_size;
OPENVINO_C_VAR(const char*) ov_property_key_hint_num_requests;

// Property
/**
 * @defgroup Property Property
 * @ingroup openvino_c
 * Set of functions representing of Property.
 * @{
 */

/**
 * @brief Initialize a properties list object.
 * @ingroup property
 * @param property The properties list will be initialized.
 * @param size The list size.
 * @return ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_properties_create(ov_properties_t* property, size_t size);

/**
 * @brief Deinitialized properties list.
 * properties->list[i].value.ptr need be managed by user.
 * @ingroup property
 * @param property The properties list object will be deinitialized.
 */
OPENVINO_C_API(void) ov_properties_free(ov_properties_t* property);

/** @} */  // end of Property
