// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdbool.h>

// Fix Windows BOOLEAN conflict before including OpenVINO C API headers

#include "openvino/c/openvino.h"

/**
 * @brief Function pointer types for OpenVINO C API functions
 */

// Version functions
typedef ov_status_e (*ov_get_openvino_version_t)(ov_version_t* version);
typedef void (*ov_version_free_t)(ov_version_t* version);

// Core functions  
typedef ov_status_e (*ov_core_create_t)(ov_core_t** core);
typedef void (*ov_core_free_t)(ov_core_t* core);
typedef ov_status_e (*ov_core_read_model_t)(const ov_core_t* core, const char* model_path, const char* bin_path, ov_model_t** model);
typedef ov_status_e (*ov_core_compile_model_t)(const ov_core_t* core, const ov_model_t* model, const char* device_name, const size_t property_args_size, ov_compiled_model_t** compiled_model);

// Model functions
typedef ov_status_e (*ov_model_get_friendly_name_t)(const ov_model_t* model, char** friendly_name);
typedef ov_status_e (*ov_model_const_output_t)(const ov_model_t* model, ov_output_const_port_t** output_port);
typedef ov_status_e (*ov_model_const_input_t)(const ov_model_t* model, ov_output_const_port_t** input_port);
typedef void (*ov_model_free_t)(ov_model_t* model);

// Tensor functions
typedef ov_status_e (*ov_tensor_create_from_host_ptr_t)(const ov_element_type_e type, const ov_shape_t shape, void* host_ptr, ov_tensor_t** tensor);
typedef ov_status_e (*ov_tensor_get_shape_t)(const ov_tensor_t* tensor, ov_shape_t* shape);
typedef ov_status_e (*ov_tensor_data_t)(const ov_tensor_t* tensor, void** data);
typedef void (*ov_tensor_free_t)(ov_tensor_t* tensor);

// Shape functions
typedef ov_status_e (*ov_shape_create_t)(const int64_t rank, const int64_t* dims, ov_shape_t* shape);
typedef void (*ov_shape_free_t)(ov_shape_t* shape);

// Layout functions
typedef ov_status_e (*ov_layout_create_t)(const char* layout_desc, ov_layout_t** layout);
typedef void (*ov_layout_free_t)(ov_layout_t* layout);

// Preprocessing functions
typedef ov_status_e (*ov_preprocess_prepostprocessor_create_t)(const ov_model_t* model, ov_preprocess_prepostprocessor_t** preprocess);
typedef void (*ov_preprocess_prepostprocessor_free_t)(ov_preprocess_prepostprocessor_t* preprocess);
typedef ov_status_e (*ov_preprocess_prepostprocessor_get_input_info_by_index_t)(const ov_preprocess_prepostprocessor_t* preprocess, const size_t index, ov_preprocess_input_info_t** input_info);
typedef void (*ov_preprocess_input_info_free_t)(ov_preprocess_input_info_t* input_info);
typedef ov_status_e (*ov_preprocess_input_info_get_tensor_info_t)(const ov_preprocess_input_info_t* input_info, ov_preprocess_input_tensor_info_t** input_tensor_info);
typedef void (*ov_preprocess_input_tensor_info_free_t)(ov_preprocess_input_tensor_info_t* input_tensor_info);
typedef ov_status_e (*ov_preprocess_input_tensor_info_set_from_t)(ov_preprocess_input_tensor_info_t* input_tensor_info, const ov_tensor_t* tensor);
typedef ov_status_e (*ov_preprocess_input_tensor_info_set_layout_t)(ov_preprocess_input_tensor_info_t* input_tensor_info, const ov_layout_t* layout);
typedef ov_status_e (*ov_preprocess_input_info_get_preprocess_steps_t)(const ov_preprocess_input_info_t* input_info, ov_preprocess_preprocess_steps_t** input_process);
typedef void (*ov_preprocess_preprocess_steps_free_t)(ov_preprocess_preprocess_steps_t* input_process);
typedef ov_status_e (*ov_preprocess_preprocess_steps_resize_t)(ov_preprocess_preprocess_steps_t* preprocess_input_process_steps, const ov_preprocess_resize_algorithm_e resize_algorithm);
typedef ov_status_e (*ov_preprocess_input_info_get_model_info_t)(const ov_preprocess_input_info_t* input_info, ov_preprocess_input_model_info_t** input_model_info);
typedef void (*ov_preprocess_input_model_info_free_t)(ov_preprocess_input_model_info_t* input_model_info);
typedef ov_status_e (*ov_preprocess_input_model_info_set_layout_t)(ov_preprocess_input_model_info_t* input_model_info, const ov_layout_t* layout);
typedef ov_status_e (*ov_preprocess_prepostprocessor_get_output_info_by_index_t)(const ov_preprocess_prepostprocessor_t* preprocess, const size_t index, ov_preprocess_output_info_t** output_info);
typedef void (*ov_preprocess_output_info_free_t)(ov_preprocess_output_info_t* output_info);
typedef ov_status_e (*ov_preprocess_output_info_get_tensor_info_t)(const ov_preprocess_output_info_t* output_info, ov_preprocess_output_tensor_info_t** output_tensor_info);
typedef void (*ov_preprocess_output_tensor_info_free_t)(ov_preprocess_output_tensor_info_t* output_tensor_info);
typedef ov_status_e (*ov_preprocess_output_set_element_type_t)(ov_preprocess_output_tensor_info_t* output_tensor_info, const ov_element_type_e element_type);
typedef ov_status_e (*ov_preprocess_prepostprocessor_build_t)(const ov_preprocess_prepostprocessor_t* preprocess, ov_model_t** model);

// Compiled model functions
typedef ov_status_e (*ov_compiled_model_create_infer_request_t)(const ov_compiled_model_t* compiled_model, ov_infer_request_t** infer_request);
typedef void (*ov_compiled_model_free_t)(ov_compiled_model_t* compiled_model);

// Infer request functions
typedef ov_status_e (*ov_infer_request_set_input_tensor_by_index_t)(ov_infer_request_t* infer_request, const size_t index, const ov_tensor_t* tensor);
typedef ov_status_e (*ov_infer_request_infer_t)(ov_infer_request_t* infer_request);
typedef ov_status_e (*ov_infer_request_get_output_tensor_by_index_t)(const ov_infer_request_t* infer_request, const size_t index, ov_tensor_t** tensor);
typedef void (*ov_infer_request_free_t)(ov_infer_request_t* infer_request);

// Output port functions
typedef void (*ov_output_const_port_free_t)(ov_output_const_port_t* port);

// Utility functions
typedef void (*ov_free_t)(const char* content);
typedef void (*ov_shutdown_t)(void);

/**
 * @brief Structure holding all function pointers
 */
typedef struct {
    void* handle;  // DLL handle
    
    // Function pointers
    ov_get_openvino_version_t ov_get_openvino_version;
    ov_version_free_t ov_version_free;
    ov_core_create_t ov_core_create;
    ov_core_free_t ov_core_free;
    ov_core_read_model_t ov_core_read_model;
    ov_core_compile_model_t ov_core_compile_model;
    ov_model_get_friendly_name_t ov_model_get_friendly_name;
    ov_model_const_output_t ov_model_const_output;
    ov_model_const_input_t ov_model_const_input;
    ov_model_free_t ov_model_free;
    ov_tensor_create_from_host_ptr_t ov_tensor_create_from_host_ptr;
    ov_tensor_get_shape_t ov_tensor_get_shape;
    ov_tensor_data_t ov_tensor_data;
    ov_tensor_free_t ov_tensor_free;
    ov_shape_create_t ov_shape_create;
    ov_shape_free_t ov_shape_free;
    ov_layout_create_t ov_layout_create;
    ov_layout_free_t ov_layout_free;
    ov_preprocess_prepostprocessor_create_t ov_preprocess_prepostprocessor_create;
    ov_preprocess_prepostprocessor_free_t ov_preprocess_prepostprocessor_free;
    ov_preprocess_prepostprocessor_get_input_info_by_index_t ov_preprocess_prepostprocessor_get_input_info_by_index;
    ov_preprocess_input_info_free_t ov_preprocess_input_info_free;
    ov_preprocess_input_info_get_tensor_info_t ov_preprocess_input_info_get_tensor_info;
    ov_preprocess_input_tensor_info_free_t ov_preprocess_input_tensor_info_free;
    ov_preprocess_input_tensor_info_set_from_t ov_preprocess_input_tensor_info_set_from;
    ov_preprocess_input_tensor_info_set_layout_t ov_preprocess_input_tensor_info_set_layout;
    ov_preprocess_input_info_get_preprocess_steps_t ov_preprocess_input_info_get_preprocess_steps;
    ov_preprocess_preprocess_steps_free_t ov_preprocess_preprocess_steps_free;
    ov_preprocess_preprocess_steps_resize_t ov_preprocess_preprocess_steps_resize;
    ov_preprocess_input_info_get_model_info_t ov_preprocess_input_info_get_model_info;
    ov_preprocess_input_model_info_free_t ov_preprocess_input_model_info_free;
    ov_preprocess_input_model_info_set_layout_t ov_preprocess_input_model_info_set_layout;
    ov_preprocess_prepostprocessor_get_output_info_by_index_t ov_preprocess_prepostprocessor_get_output_info_by_index;
    ov_preprocess_output_info_free_t ov_preprocess_output_info_free;
    ov_preprocess_output_info_get_tensor_info_t ov_preprocess_output_info_get_tensor_info;
    ov_preprocess_output_tensor_info_free_t ov_preprocess_output_tensor_info_free;
    ov_preprocess_output_set_element_type_t ov_preprocess_output_set_element_type;
    ov_preprocess_prepostprocessor_build_t ov_preprocess_prepostprocessor_build;
    ov_compiled_model_create_infer_request_t ov_compiled_model_create_infer_request;
    ov_compiled_model_free_t ov_compiled_model_free;
    ov_infer_request_set_input_tensor_by_index_t ov_infer_request_set_input_tensor_by_index;
    ov_infer_request_infer_t ov_infer_request_infer;
    ov_infer_request_get_output_tensor_by_index_t ov_infer_request_get_output_tensor_by_index;
    ov_infer_request_free_t ov_infer_request_free;
    ov_output_const_port_free_t ov_output_const_port_free;
    ov_free_t ov_free;
    ov_shutdown_t ov_shutdown;
} ov_api_t;

/**
 * @brief Load OpenVINO C API library dynamically
 * @param dll_path Path to the OpenVINO C DLL (e.g., "openvino_c.dll")
 * @param api Pointer to ov_api_t structure to be filled
 * @return true on success, false on failure
 */
bool ov_api_load(const char* dll_path, ov_api_t* api);

/**
 * @brief Unload OpenVINO C API library
 * @param api Pointer to ov_api_t structure
 */
void ov_api_unload(ov_api_t* api);
