// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/c/ov_remote_context.h"

#include <stdarg.h>

#include "common.h"
#include "openvino/c/ov_core.h"

const char* ov_property_key_intel_gpu_context_type = "CONTEXT_TYPE";
const char* ov_property_key_intel_gpu_ocl_context = "OCL_CONTEXT";
const char* ov_property_key_intel_gpu_ocl_context_device_id = "OCL_CONTEXT_DEVICE_ID";
const char* ov_property_key_intel_gpu_tile_id = "TILE_ID";
const char* ov_property_key_intel_gpu_ocl_queue = "OCL_QUEUE";
const char* ov_property_key_intel_gpu_va_device = "VA_DEVICE";
const char* ov_property_key_intel_gpu_shared_mem_type = "SHARED_MEM_TYPE";
const char* ov_property_key_intel_gpu_mem_handle = "MEM_HANDLE";
const char* ov_property_key_intel_gpu_dev_object_handle = "DEV_OBJECT_HANDLE";
const char* ov_property_key_intel_gpu_va_plane = "VA_PLANE";

inline bool check_intel_gpu_property_value_is_ptr(std::string& key) {
#ifdef _WIN32
    return (key == ov_property_key_intel_gpu_ocl_context) || (key == ov_property_key_intel_gpu_ocl_queue) ||
           (key == ov_property_key_intel_gpu_va_device) || (key == ov_property_key_intel_gpu_mem_handle) ||
           (key == ov_property_key_intel_gpu_dev_object_handle);
#else
    return (key == ov_property_key_intel_gpu_ocl_context) || (key == ov_property_key_intel_gpu_ocl_queue) ||
           (key == ov_property_key_intel_gpu_va_device) || (key == ov_property_key_intel_gpu_mem_handle);
#endif
}

//!< Properties of intel gpu cannot be compeletly handled by (char*) type, because it contains non-char pointer which
//!< points to memory block, so we have to use (void *) type to parse it from va_arg list.
//!< (char *) type data will be copied before pass to ov::AnyMap, to prevent it from being freed out of ov api calling.
//!< (void *) type data is memory block or gpu object handle, it cannot be copied into a new place.
#define GET_INTEL_GPU_PROPERTY_FROM_ARGS_LIST(property_size)       \
    for (size_t i = 0; i < property_size; i++) {                   \
        std::string property_key = va_arg(args_ptr, char*);        \
        if (check_intel_gpu_property_value_is_ptr(property_key)) { \
            ov::Any value = va_arg(args_ptr, void*);               \
            property[property_key] = std::move(value);             \
        } else {                                                   \
            std::string _value = va_arg(args_ptr, char*);          \
            ov::Any value = _value;                                \
            property[property_key] = std::move(value);             \
        }                                                          \
    }

ov_status_e ov_core_create_context(const ov_core_t* core,
                                   const char* device_name,
                                   const size_t property_args_size,
                                   ov_remote_context** context,
                                   ...) {
    if (!core || !device_name || !context || property_args_size % 2 == 1) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap property = {};
        size_t property_size = property_args_size / 2;
        va_list args_ptr;
        va_start(args_ptr, context);
        GET_INTEL_GPU_PROPERTY_FROM_ARGS_LIST(property_size);
        va_end(args_ptr);

        std::string dev_name = device_name;
        ov::RemoteContext object = core->object->create_context(dev_name, property);

        std::unique_ptr<ov_remote_context> _context(new ov_remote_context);
        _context->object = std::make_shared<ov::RemoteContext>(std::move(object));
        *context = _context.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_remote_context_create_tensor(const ov_remote_context_t* context,
                                            const ov_element_type_e type,
                                            const ov_shape_t shape,
                                            const size_t property_args_size,
                                            ov_tensor_t** remote_tensor,
                                            ...) {
    if (!context || !shape.dims || !remote_tensor || property_args_size % 2 == 1) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::AnyMap property = {};
        size_t property_size = property_args_size / 2;
        va_list args_ptr;
        va_start(args_ptr, remote_tensor);
        GET_INTEL_GPU_PROPERTY_FROM_ARGS_LIST(property_size);
        va_end(args_ptr);

        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        auto tmp_type = get_element_type(type);
        ov::RemoteTensor object = context->object->create_tensor(tmp_type, tmp_shape, property);

        std::unique_ptr<ov_tensor> _remote_tensor(new ov_tensor);
        _remote_tensor->object = std::make_shared<ov::RemoteTensor>(std::move(object));
        *remote_tensor = _remote_tensor.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_remote_context_get_device_name(const ov_remote_context_t* context, char** device_name) {
    if (!context || !device_name) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto value = context->object->get_device_name();
        *device_name = str_to_char_array(value);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

//!< Output ov::AnyMap to be C string style data:
//!<    "key_name_1 key_value_str_1 key_name_2 key_value_str_2 ..."
//!< Note: there is a space separator beween each key and value.
inline void convert_params_to_string(ov::AnyMap& paramsMap, char*& res_str, size_t& size) {
    ov::Any param = paramsMap;
    std::string res = param.as<std::string>();
    size = paramsMap.size();
    res_str = str_to_char_array(res);
}

ov_status_e ov_remote_context_get_params(const ov_remote_context_t* context, size_t* size, char** params) {
    if (!context || !size || !params) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto paramsMap = context->object->get_params();
        convert_params_to_string(paramsMap, *params, *size);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_remote_context_create_host_tensor(const ov_remote_context_t* context,
                                                 const ov_element_type_e type,
                                                 const ov_shape_t shape,
                                                 ov_tensor_t** tensor) {
    if (!context || !shape.dims || !tensor) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        ov::Shape tmp_shape;
        std::copy_n(shape.dims, shape.rank, std::back_inserter(tmp_shape));
        auto tmp_type = get_element_type(type);
        ov::Tensor object = context->object->create_host_tensor(tmp_type, tmp_shape);

        std::unique_ptr<ov_tensor> _tensor(new ov_tensor_t);
        _tensor->object = std::make_shared<ov::Tensor>(std::move(object));
        *tensor = _tensor.release();
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

void ov_remote_context_free(ov_remote_context_t* context) {
    if (context)
        delete context;
}

ov_status_e ov_remote_tensor_get_params(ov_tensor_t* tensor, size_t* size, char** params) {
    if (!tensor || !size || !params) {
        return ov_status_e::INVALID_C_PARAM;
    }

    if (!tensor->object->is<ov::RemoteTensor>()) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto& remote_tensor = tensor->object->as<ov::RemoteTensor>();
        auto paramsMap = remote_tensor.get_params();
        convert_params_to_string(paramsMap, *params, *size);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}

ov_status_e ov_remote_tensor_get_device_name(ov_tensor_t* remote_tensor, char** device_name) {
    if (!remote_tensor || !device_name) {
        return ov_status_e::INVALID_C_PARAM;
    }

    if (!remote_tensor->object->is<ov::RemoteTensor>()) {
        return ov_status_e::INVALID_C_PARAM;
    }

    try {
        auto& _remote_tensor = remote_tensor->object->as<ov::RemoteTensor>();
        auto value = _remote_tensor.get_device_name();
        *device_name = str_to_char_array(value);
    }
    CATCH_OV_EXCEPTIONS
    return ov_status_e::OK;
}