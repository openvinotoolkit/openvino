// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_property.h"

#include "common.h"

// Read-only property key
const char* ov_property_key_supported_properties = "SUPPORTED_PROPERTIES";
const char* ov_property_key_available_devices = "AVAILABLE_DEVICES";
const char* ov_property_key_optimal_number_of_infer_requests = "OPTIMAL_NUMBER_OF_INFER_REQUESTS";
const char* ov_property_key_range_for_async_infer_requests = "RANGE_FOR_ASYNC_INFER_REQUESTS";
const char* ov_property_key_range_for_streams = "RANGE_FOR_STREAMS";
const char* ov_property_key_device_full_name = "FULL_DEVICE_NAME";
const char* ov_property_key_device_capabilities = "OPTIMIZATION_CAPABILITIES";
const char* ov_property_key_model_name = "NETWORK_NAME";
const char* ov_property_key_optimal_batch_size = "OPTIMAL_BATCH_SIZE";
const char* ov_property_key_max_batch_size = "MAX_BATCH_SIZE";

// Read-write property key
const char* ov_property_key_cache_dir = "CACHE_DIR";
const char* ov_property_key_num_streams = "NUM_STREAMS";
const char* ov_property_key_affinity = "AFFINITY";
const char* ov_property_key_inference_num_threads = "INFERENCE_NUM_THREADS";
const char* ov_property_key_hint_performance_mode = "PERFORMANCE_HINT";
const char* ov_property_key_hint_inference_precision = "INFERENCE_PRECISION_HINT";
const char* ov_property_key_hint_num_requests = "PERFORMANCE_HINT_NUM_REQUESTS";
const char* ov_property_key_hint_model_priority = "MODEL_PRIORITY";
const char* ov_property_key_log_level = "LOG_LEVEL";
const char* ov_property_key_enable_profiling = "PERF_COUNT";
const char* ov_property_key_device_priorities = "MULTI_DEVICE_PRIORITIES";

// Property data type - singular data
const char* ov_property_value_type_int32 = "INT32";
const char* ov_property_value_type_uint32 = "UINT32";
const char* ov_property_value_type_bool = "BOOL";
const char* ov_property_value_type_enum = "ENUM";
const char* ov_property_value_type_ptr = "PTR";
const char* ov_property_value_type_string = "STRING";
const char* ov_property_value_type_float = "FLOAT";
const char* ov_property_value_type_double = "DOUBLE";

// Property data type - compounded data
const char* ov_property_value_type_map = "MAP";
const char* ov_property_value_type_vector = "VECTOR";

ov::Any get_property_enum_value(std::string key, int value) {
    ov::Any ret = {};
    if (key == ov_property_key_hint_performance_mode) {
        ov::hint::PerformanceMode mode = static_cast<ov::hint::PerformanceMode>(value);
        ret = mode;
    } else if (key == ov_property_key_affinity) {
        ov::Affinity affinity = static_cast<ov::Affinity>(value);
        ret = affinity;
    } else if (key == ov_property_key_hint_inference_precision) {
        ov::element::Type_t type = static_cast<ov::element::Type_t>(value);
        ret = type;
    } else if (key == ov_property_key_log_level) {
        ov::log::Level level = static_cast<ov::log::Level>(value);
        ret = level;
    } else if (key == ov_property_key_hint_model_priority) {
        ov::hint::Priority priority = static_cast<ov::hint::Priority>(value);
        ret = priority;
    }
    return ret;
}

bool check_enum_property(std::string key) {
    ov::Any ret = {};
    if (key == ov_property_key_hint_performance_mode || key == ov_property_key_affinity ||
        key == ov_property_key_hint_inference_precision || key == ov_property_key_log_level ||
        key == ov_property_key_hint_model_priority) {
        return true;
    }
    return false;
}