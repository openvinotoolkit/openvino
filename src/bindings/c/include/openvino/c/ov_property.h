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

typedef struct ov_property ov_property_t;

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
 * @struct ov_property_key_e
 * @brief Represent all available property key.
 */
typedef enum {
    SUPPORTED_PROPERTIES = 0U,  //!<  Read-only property<char *> to get a string list of supported read-only properties.
    AVAILABLE_DEVICES,          //!<  Read-only property<char *> to get a list of available device IDs
    OPTIMAL_NUMBER_OF_INFER_REQUESTS,  //!<  Read-only property<uint32_t> to get an unsigned integer value of optimaln
                                       //!<  umber of compiled model infer requests.
    RANGE_FOR_ASYNC_INFER_REQUESTS,    //!<  Read-only property<unsigned int, unsigned int, unsigned int> to provide a
                                       //!<  hint for a range for number of async infer requests. If device supports
                                       //!<  streams, the metric provides range for number of IRs per stream.
    RANGE_FOR_STREAMS,  //!<  Read-only property<unsigned int, unsigned int> to provide information about a range for
                        //!<  streams on platforms where streams are supported
    FULL_DEVICE_NAME,   //!<  Read-only property<char *> to get a string value representing a full device name.
    OPTIMIZATION_CAPABILITIES,  //!<  Read-only property<char *> to get a string list of capabilities options per
                                //!<  device.
    CACHE_DIR,    //!<  Read-write property<char *> to set/get the directory which will be used to store any data cached
                  //!<  by plugins.
    NUM_STREAMS,  //!<  Read-write property<uint32_t> to set/get the number of executor logical partitions
    AFFINITY,  //!<  Read-write property<ov_affinity_e> to set/get the name for setting CPU affinity per thread option.
    INFERENCE_NUM_THREADS,  //!<  Read-write property<int32_t> to set/get the maximum number of threads that can be used
                            //!<  for inference tasks.
    PERFORMANCE_HINT,       //!< Read-write property<ov_performance_mode_e>, it is high-level OpenVINO Performance Hints
                       //!< unlike low-level properties that are individual (per-device), the hints are something that
                       //!< every device accepts and turns into device-specific settings detail see
                       //!< ov_performance_mode_e to get its hint's key name
    NETWORK_NAME,              //!<  Read-only property<char *> to get a name of name of a model
    INFERENCE_PRECISION_HINT,  //!< Read-write property<ov_element_type_e> to set the hint for device to use specified
                               //!< precision for inference
    OPTIMAL_BATCH_SIZE,  //!<  Read-only property<uint32_t> to query information optimal batch size for the given device
                         //!<  and the network
    MAX_BATCH_SIZE,  //!<  Read-only property to get maximum batch size which does not cause performance degradation due
                     //!<  to memory swap impact.
    PERFORMANCE_HINT_NUM_REQUESTS,  //!<  (Optional) property<uint32_t> that backs the Performance Hints by giving
                                    //!<  additional information on how many inference requests the application will be
                                    //!<  keeping in flight usually this value comes from the actual use-case  (e.g.
                                    //!<  number of video-cameras, or other sources of inputs)
} ov_property_key_e;

/**
 * @enum ov_property_value_type_e
 * @brief Enum to define property value type.
 */
typedef enum {
    BOOL = 0U,  //!< boolean data
    CHAR,       //!< char data
    ENUM,       //!< enum data
    INT32,      //!< int32 data
    UINT32,     //!< uint32 data
    FLOAT,      //!< float data
} ov_property_value_type_e;

/**
 * @struct ov_property_value_t
 * @brief Represent a property value
 */
typedef struct {
    void* ptr;
    size_t cnt;
    ov_property_value_type_e type;
} ov_property_value_t;

// Property
/**
 * @defgroup Property Property
 * @ingroup openvino_c
 * Set of functions representing of Property.
 * @{
 */

/**
 * @brief Create a property object.
 * @ingroup property
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_property_create(ov_property_t** property);

/**
 * @brief Free property object.
 * @ingroup property
 * @param property will be released.
 */
OPENVINO_C_API(void) ov_property_free(ov_property_t* property);

/**
 * @brief Create a property value object.
 * @ingroup property
 * @param ov_status_e a status code, return OK if successful
 */
OPENVINO_C_API(ov_status_e) ov_property_value_create(ov_property_value_t** value);

/**
 * @brief Clean property data.
 * @ingroup property
 * @param property data will be clean.
 */
OPENVINO_C_API(void) ov_property_value_clean(ov_property_value_t* value);

/**
 * @brief Put <key, value> into property object.
 * @ingroup property
 * @param property will be add new <key, value>.
 */
OPENVINO_C_API(ov_status_e) ov_property_put(ov_property_t* property, ov_property_key_e key, ov_property_value_t* value);

/** @} */  // end of Property
