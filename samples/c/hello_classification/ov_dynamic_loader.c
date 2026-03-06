// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "ov_dynamic_loader.h"

#define LOAD_FUNCTION(api, name) \
    do { \
        api->name = (name##_t)get_function_address(api->handle, #name); \
        if (!api->name) { \
            fprintf(stderr, "[ERROR] Failed to load function: %s\n", #name); \
            ov_api_unload(api); \
            return false; \
        } \
    } while(0)

static void* load_library(const char* path) {
#ifdef _WIN32
    return (void*)LoadLibraryA(path);
#else
    return dlopen(path, RTLD_NOW | RTLD_LOCAL);
#endif
}

static void unload_library(void* handle) {
    if (!handle) return;
#ifdef _WIN32
    FreeLibrary((HMODULE)handle);
#else
    dlclose(handle);
#endif
}

static void* get_function_address(void* handle, const char* name) {
    if (!handle) return NULL;
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)handle, name);
#else
    return dlsym(handle, name);
#endif
}

bool ov_api_load(const char* dll_path, ov_api_t* api) {
    if (!dll_path || !api) {
        fprintf(stderr, "[ERROR] Invalid parameters\n");
        return false;
    }
    
    // Initialize structure
    memset(api, 0, sizeof(ov_api_t));
    
    // Load library
    api->handle = load_library(dll_path);
    if (!api->handle) {
#ifdef _WIN32
        fprintf(stderr, "[ERROR] Failed to load library: %s (Error code: %lu)\n", dll_path, GetLastError());
#else
        fprintf(stderr, "[ERROR] Failed to load library: %s (%s)\n", dll_path, dlerror());
#endif
        return false;
    }
    
    printf("[INFO] Successfully loaded library: %s\n", dll_path);
    
    // Load all function pointers
    LOAD_FUNCTION(api, ov_get_openvino_version);
    LOAD_FUNCTION(api, ov_version_free);
    LOAD_FUNCTION(api, ov_core_create);
    LOAD_FUNCTION(api, ov_core_free);
    LOAD_FUNCTION(api, ov_core_read_model);
    LOAD_FUNCTION(api, ov_core_compile_model);
    LOAD_FUNCTION(api, ov_model_get_friendly_name);
    LOAD_FUNCTION(api, ov_model_const_output);
    LOAD_FUNCTION(api, ov_model_const_input);
    LOAD_FUNCTION(api, ov_model_free);
    LOAD_FUNCTION(api, ov_tensor_create_from_host_ptr);
    LOAD_FUNCTION(api, ov_tensor_get_shape);
    LOAD_FUNCTION(api, ov_tensor_data);
    LOAD_FUNCTION(api, ov_tensor_free);
    LOAD_FUNCTION(api, ov_shape_create);
    LOAD_FUNCTION(api, ov_shape_free);
    LOAD_FUNCTION(api, ov_layout_create);
    LOAD_FUNCTION(api, ov_layout_free);
    LOAD_FUNCTION(api, ov_preprocess_prepostprocessor_create);
    LOAD_FUNCTION(api, ov_preprocess_prepostprocessor_free);
    LOAD_FUNCTION(api, ov_preprocess_prepostprocessor_get_input_info_by_index);
    LOAD_FUNCTION(api, ov_preprocess_input_info_free);
    LOAD_FUNCTION(api, ov_preprocess_input_info_get_tensor_info);
    LOAD_FUNCTION(api, ov_preprocess_input_tensor_info_free);
    LOAD_FUNCTION(api, ov_preprocess_input_tensor_info_set_from);
    LOAD_FUNCTION(api, ov_preprocess_input_tensor_info_set_layout);
    LOAD_FUNCTION(api, ov_preprocess_input_info_get_preprocess_steps);
    LOAD_FUNCTION(api, ov_preprocess_preprocess_steps_free);
    LOAD_FUNCTION(api, ov_preprocess_preprocess_steps_resize);
    LOAD_FUNCTION(api, ov_preprocess_input_info_get_model_info);
    LOAD_FUNCTION(api, ov_preprocess_input_model_info_free);
    LOAD_FUNCTION(api, ov_preprocess_input_model_info_set_layout);
    LOAD_FUNCTION(api, ov_preprocess_prepostprocessor_get_output_info_by_index);
    LOAD_FUNCTION(api, ov_preprocess_output_info_free);
    LOAD_FUNCTION(api, ov_preprocess_output_info_get_tensor_info);
    LOAD_FUNCTION(api, ov_preprocess_output_tensor_info_free);
    LOAD_FUNCTION(api, ov_preprocess_output_set_element_type);
    LOAD_FUNCTION(api, ov_preprocess_prepostprocessor_build);
    LOAD_FUNCTION(api, ov_compiled_model_create_infer_request);
    LOAD_FUNCTION(api, ov_compiled_model_free);
    LOAD_FUNCTION(api, ov_infer_request_set_input_tensor_by_index);
    LOAD_FUNCTION(api, ov_infer_request_infer);
    LOAD_FUNCTION(api, ov_infer_request_get_output_tensor_by_index);
    LOAD_FUNCTION(api, ov_infer_request_free);
    LOAD_FUNCTION(api, ov_output_const_port_free);
    LOAD_FUNCTION(api, ov_free);
    LOAD_FUNCTION(api, ov_shutdown);
    
    printf("[INFO] All OpenVINO C API functions loaded successfully\n");
    return true;
}

void ov_api_unload(ov_api_t* api) {
    if (!api) return;
    
    if (api->handle) {
        unload_library(api->handle);
        printf("[INFO] OpenVINO C library unloaded\n");
    }
    
    memset(api, 0, sizeof(ov_api_t));
}
