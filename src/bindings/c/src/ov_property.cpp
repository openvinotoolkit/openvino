// Copyright (C) 2018-2026 Intel Corporation
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
const char* ov_property_key_cache_mode = "CACHE_MODE";
const char* ov_property_key_num_streams = "NUM_STREAMS";
const char* ov_property_key_inference_num_threads = "INFERENCE_NUM_THREADS";
const char* ov_property_key_hint_performance_mode = "PERFORMANCE_HINT";
const char* ov_property_key_hint_enable_cpu_pinning = "ENABLE_CPU_PINNING";
const char* ov_property_key_hint_enable_cpu_reservation = "ENABLE_CPU_RESERVATION";
const char* ov_property_key_hint_scheduling_core_type = "SCHEDULING_CORE_TYPE";
const char* ov_property_key_hint_enable_hyper_threading = "ENABLE_HYPER_THREADING";
const char* ov_property_key_hint_inference_precision = "INFERENCE_PRECISION_HINT";
const char* ov_property_key_hint_num_requests = "PERFORMANCE_HINT_NUM_REQUESTS";
const char* ov_property_key_hint_model_priority = "MODEL_PRIORITY";
const char* ov_property_key_log_level = "LOG_LEVEL";
const char* ov_property_key_enable_profiling = "PERF_COUNT";
const char* ov_property_key_device_priorities = "MULTI_DEVICE_PRIORITIES";
const char* ov_property_key_hint_execution_mode = "EXECUTION_MODE_HINT";
const char* ov_property_key_force_tbb_terminate = "FORCE_TBB_TERMINATE";
const char* ov_property_key_enable_mmap = "ENABLE_MMAP";
const char* ov_property_key_auto_batch_timeout = "AUTO_BATCH_TIMEOUT";
const char* ov_property_key_intel_gpu_config_file = "CONFIG_FILE";

// Write-only property key
const char* ov_property_key_cache_encryption_callbacks = "CACHE_ENCRYPTION_CALLBACKS";

ov_status_e ov_property_create(ov_property** prop) {
    if (!prop) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        std::unique_ptr<ov_property> _prop = std::make_unique<ov_property>();
        _prop->object = {};
        *prop = _prop.release();

    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_property_free(ov_property* prop) {
    if (prop) {
        delete prop;
    }
}

ov_status_e ov_property_put_str(ov_property* prop, const char* key, const char* val) {
    if (!prop) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        prop->object[key] = val;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_property_put_int(ov_property* prop, const char* key, const int val) {
    if (!prop) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        (prop->object)[key] = val;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}

ov_status_e ov_property_put_encryption_callbacks(ov_property* prop,
                                                 const char* key,
                                                 const ov_encryption_callbacks* val) {
    if (!prop) {
        return ov_status_e::INVALID_C_PARAM;
    }
    try {
        auto encrypt_func = val->encrypt_func;
        auto decrypt_func = val->decrypt_func;
        std::function<std::string(const std::string&)> encrypt_value = [encrypt_func](const std::string& in) {
            size_t out_size = 0;
            std::string out_str;
            encrypt_func(in.c_str(), in.length(), nullptr, &out_size);
            if (out_size > 0) {
                std::unique_ptr<char[]> output_ptr = std::make_unique<char[]>(out_size);
                if (output_ptr) {
                    char* output = output_ptr.get();
                    encrypt_func(in.c_str(), in.length(), output, &out_size);
                    out_str.assign(output, out_size);
                }
            }
            return out_str;
        };
        std::function<std::string(const std::string&)> decrypt_value = [decrypt_func](const std::string& in) {
            size_t out_size = 0;
            std::string out_str;
            decrypt_func(in.c_str(), in.length(), nullptr, &out_size);
            if (out_size > 0) {
                std::unique_ptr<char[]> output_ptr = std::make_unique<char[]>(out_size);
                if (output_ptr) {
                    char* output = output_ptr.get();
                    decrypt_func(in.c_str(), in.length(), output, &out_size);
                    out_str.assign(output, out_size);
                }
            }
            return out_str;
        };
        ov::EncryptionCallbacks encryption_callbacks{std::move(encrypt_value), std::move(decrypt_value)};
        prop->object[key] = encryption_callbacks;
    } catch (...) {
        return ov_status_e::UNKNOW_EXCEPTION;
    }
    return ov_status_e::OK;
}
