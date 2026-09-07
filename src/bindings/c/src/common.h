// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <cassert>
#include <fstream>
#include <iterator>
#include <map>
#include <string>

#include "openvino/c/ov_common.h"
#include "openvino/c/ov_property.h"
#include "openvino/c/ov_remote_context.h"
#include "openvino/core/except.hpp"
#include "openvino/openvino.hpp"
#include "openvino/runtime/exception.hpp"

#define CATCH_OV_EXCEPTION(StatusCode, ExceptionType) \
    catch (const ov::ExceptionType& ex) {             \
        dup_last_err_msg(ex.what());                  \
        return ov_status_e::StatusCode;               \
    }

#define CATCH_OV_EXCEPTIONS                                \
    CATCH_OV_EXCEPTION(REQUEST_BUSY, Busy)                 \
    CATCH_OV_EXCEPTION(INFER_CANCELLED, Cancelled)         \
    CATCH_OV_EXCEPTION(NOT_IMPLEMENTED, NotImplemented)    \
    CATCH_OV_EXCEPTION(GENERAL_ERROR, Exception)           \
    catch (...) {                                          \
        dup_last_err_msg("An unknown exception occurred"); \
        return ov_status_e::UNKNOW_EXCEPTION;              \
    }

#define GET_PROPERTY_FROM_ARGS_LIST                                                                            \
    std::string property_key = va_arg(args_ptr, char*);                                                        \
    if (property_key == ov::cache_encryption_callbacks.name()) {                                               \
        ov_encryption_callbacks* _value = va_arg(args_ptr, ov_encryption_callbacks*);                          \
        auto encrypt_func = _value->encrypt_func;                                                              \
        auto decrypt_func = _value->decrypt_func;                                                              \
        std::function<std::string(const std::string&)> encrypt_value = [encrypt_func](const std::string& in) { \
            size_t out_size = 0;                                                                               \
            std::string out_str;                                                                               \
            encrypt_func(in.c_str(), in.length(), nullptr, &out_size);                                         \
            if (out_size > 0) {                                                                                \
                std::unique_ptr<char[]> output_ptr(new char[out_size]);                                        \
                if (output_ptr) {                                                                              \
                    char* output = output_ptr.get();                                                           \
                    encrypt_func(in.c_str(), in.length(), output, &out_size);                                  \
                    out_str.assign(output, out_size);                                                          \
                }                                                                                              \
            }                                                                                                  \
            return out_str;                                                                                    \
        };                                                                                                     \
        std::function<std::string(const std::string&)> decrypt_value = [decrypt_func](const std::string& in) { \
            size_t out_size = 0;                                                                               \
            std::string out_str;                                                                               \
            decrypt_func(in.c_str(), in.length(), nullptr, &out_size);                                         \
            if (out_size > 0) {                                                                                \
                std::unique_ptr<char[]> output_ptr(new char[out_size]);                                        \
                if (output_ptr) {                                                                              \
                    char* output = output_ptr.get();                                                           \
                    decrypt_func(in.c_str(), in.length(), output, &out_size);                                  \
                    out_str.assign(output, out_size);                                                          \
                }                                                                                              \
            }                                                                                                  \
            return out_str;                                                                                    \
        };                                                                                                     \
        ov::EncryptionCallbacks encryption_callbacks{std::move(encrypt_value), std::move(decrypt_value)};      \
        property[property_key] = encryption_callbacks;                                                         \
    } else {                                                                                                   \
        std::string _value = va_arg(args_ptr, char*);                                                          \
        ov::Any value = _value;                                                                                \
        property[property_key] = value;                                                                        \
    }

/**
 * @brief Converts an ov_encryption_callbacks C struct into an ov::EncryptionCallbacks C++ object.
 */
inline ov::EncryptionCallbacks make_ov_encryption_callbacks(const ov_encryption_callbacks* cb) {
    if (!cb || !cb->encrypt_func || !cb->decrypt_func) {
        OPENVINO_THROW("ov_property_key_cache_encryption_callbacks: value must be a valid ov_encryption_callbacks* "
                       "with non-null encrypt_func and decrypt_func");
    }
    auto encrypt_func = cb->encrypt_func;
    auto decrypt_func = cb->decrypt_func;
    std::function<std::string(const std::string&)> encrypt_fn = [encrypt_func](const std::string& in) {
        size_t out_size = 0;
        std::string out_str;
        encrypt_func(in.c_str(), in.length(), nullptr, &out_size);
        if (out_size > 0) {
            std::unique_ptr<char[]> output_ptr(new char[out_size]);
            if (output_ptr) {
                char* output = output_ptr.get();
                encrypt_func(in.c_str(), in.length(), output, &out_size);
                out_str.assign(output, out_size);
            }
        }
        return out_str;
    };
    std::function<std::string(const std::string&)> decrypt_fn = [decrypt_func](const std::string& in) {
        size_t out_size = 0;
        std::string out_str;
        decrypt_func(in.c_str(), in.length(), nullptr, &out_size);
        if (out_size > 0) {
            std::unique_ptr<char[]> output_ptr(new char[out_size]);
            if (output_ptr) {
                char* output = output_ptr.get();
                decrypt_func(in.c_str(), in.length(), output, &out_size);
                out_str.assign(output, out_size);
            }
        }
        return out_str;
    };
    return ov::EncryptionCallbacks{std::move(encrypt_fn), std::move(decrypt_fn)};
}

/**
 * @brief Returns true for GPU property keys whose value is a raw pointer (OCL/VA handle),
 *        not a null-terminated string.  Mirrors the logic in GET_INTEL_GPU_PROPERTY_FROM_ARGS_LIST.
 */
inline bool ov_property_value_is_gpu_ptr(const std::string& key) {
#ifdef _WIN32
    return key == ov_property_key_intel_gpu_ocl_context || key == ov_property_key_intel_gpu_ocl_queue ||
           key == ov_property_key_intel_gpu_va_device || key == ov_property_key_intel_gpu_mem_handle ||
           key == ov_property_key_intel_gpu_dev_object_handle;
#else
    return key == ov_property_key_intel_gpu_ocl_context || key == ov_property_key_intel_gpu_ocl_queue ||
           key == ov_property_key_intel_gpu_va_device || key == ov_property_key_intel_gpu_mem_handle;
#endif
}

/**
 * @brief Builds an ov::AnyMap from a flat array of ov_property_t.
 *
 * The value field is a const void*.  The correct C++ type is determined from the key,
 * mirroring the existing GET_PROPERTY_FROM_ARGS_LIST / GET_INTEL_GPU_PROPERTY_FROM_ARGS_LIST
 * variadic macros:
 *   - ov_property_key_cache_encryption_callbacks → cast to ov_encryption_callbacks*
 *   - known GPU handle keys                      → store as void* (ov::Any)
 *   - everything else                            → cast to const char* and copy as std::string
 */
inline ov::AnyMap ov_build_property_map(const ov_property_t* properties, size_t num_properties) {
    ov::AnyMap result;
    for (size_t i = 0; i < num_properties; ++i) {
        if (!properties[i].key) {
            OPENVINO_THROW("ov_property_t: key must not be null");
        }
        const std::string key = properties[i].key;
        const void* val = properties[i].value;
        if (!val) {
            OPENVINO_THROW("ov_property_t: value for key '", key, "' must not be null");
        }
        if (key == ov::cache_encryption_callbacks.name()) {
            result[key] = make_ov_encryption_callbacks(static_cast<const ov_encryption_callbacks*>(val));
        } else if (ov_property_value_is_gpu_ptr(key)) {
            result[key] = ov::Any(const_cast<void*>(val));
        } else {
            result[key] = std::string(static_cast<const char*>(val));
        }
    }
    return result;
}

/**
 * @struct ov_core
 * @brief This struct represents OpenVINO Core entity.
 */
struct ov_core {
    std::shared_ptr<ov::Core> object;
};

/**
 * @struct ov_model
 * @brief This is an interface of ov::Model
 */
struct ov_model {
    std::shared_ptr<ov::Model> object;
};

/**
 * @struct ov_output_const_port
 * @brief This is an interface of ov::Output<const ov::Node>
 */
struct ov_output_const_port {
    std::shared_ptr<ov::Output<const ov::Node>> object;
};

/**
 * @struct ov_output_port
 * @brief This is an interface of ov::Output<ov::Node>
 */
struct ov_output_port {
    std::shared_ptr<ov::Output<ov::Node>> object;
};

/**
 * @struct ov_compiled_model
 * @brief This is an interface of ov::CompiledModel
 */
struct ov_compiled_model {
    std::shared_ptr<ov::CompiledModel> object;
};

/**
 * @struct ov_infer_request
 * @brief This is an interface of ov::InferRequest
 */
struct ov_infer_request {
    std::shared_ptr<ov::InferRequest> object;
};

/**
 * @struct ov_layout
 * @brief This is an interface of ov::Layout
 */
struct ov_layout {
    ov::Layout object;
};

/**
 * @struct ov_tensor
 * @brief This is an interface of ov_tensor
 */
struct ov_tensor {
    std::shared_ptr<ov::Tensor> object;
};

/**
 * @struct ov_preprocess_prepostprocessor
 * @brief This is an interface of ov::preprocess::PrePostProcessor
 */
struct ov_preprocess_prepostprocessor {
    std::shared_ptr<ov::preprocess::PrePostProcessor> object;
};

/**
 * @struct ov_preprocess_input_info
 * @brief This is an interface of ov::preprocess::InputInfo
 */
struct ov_preprocess_input_info {
    ov::preprocess::InputInfo* object;
};

/**
 * @struct ov_preprocess_input_tensor_info
 * @brief This is an interface of ov::preprocess::InputTensorInfo
 */
struct ov_preprocess_input_tensor_info {
    ov::preprocess::InputTensorInfo* object;
};

/**
 * @struct ov_preprocess_output_info
 * @brief This is an interface of ov::preprocess::OutputInfo
 */
struct ov_preprocess_output_info {
    ov::preprocess::OutputInfo* object;
};

/**
 * @struct ov_preprocess_output_tensor_info
 * @brief This is an interface of ov::preprocess::OutputTensorInfo
 */
struct ov_preprocess_output_tensor_info {
    ov::preprocess::OutputTensorInfo* object;
};

/**
 * @struct ov_preprocess_input_model_info
 * @brief This is an interface of ov::preprocess::InputModelInfo
 */
struct ov_preprocess_input_model_info {
    ov::preprocess::InputModelInfo* object;
};

/**
 * @struct ov_preprocess_preprocess_steps
 * @brief This is an interface of ov::preprocess::PreProcessSteps
 */
struct ov_preprocess_preprocess_steps {
    ov::preprocess::PreProcessSteps* object;
};

/**
 * @struct ov_remote_context
 * @brief This is an interface of ov::RemoteContext
 */
struct ov_remote_context {
    std::shared_ptr<ov::RemoteContext> object;
};

char* str_to_char_array(const std::string& str);
ov_element_type_e find_ov_element_type_e(ov::element::Type type);
ov::element::Type get_element_type(ov_element_type_e type);
void dup_last_err_msg(const char* msg);
