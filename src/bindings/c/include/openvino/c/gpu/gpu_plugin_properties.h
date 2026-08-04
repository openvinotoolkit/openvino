// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a specified header file for gpu plugin's properties
 *
 * @file gpu_plugin_properties.h
 */

#pragma once
#include "openvino/c/ov_common.h"

/**
 * @brief gpu plugin properties key for remote context/tensor
 */

//!< Read-write property: shared device context type.
//!< Value is string, it can be one of below strings:
//!<    "VULKAN" - Vulkan context
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_context_type;

//!< Read-write property<int string>: In case of multi-tile system, this key identifies tile within given context.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_tile_id;

//!< Read-write property: type of internal shared memory in a shared memory blob
//!< parameter map.
//!< Value is string, it can be one of below strings:
//!<    "USM_USER_BUFFER"   - Shared USM pointer allocated by user
//!<    "USM_HOST_BUFFER"   - Shared USM pointer type with host allocation type allocated by plugin
//!<    "USM_DEVICE_BUFFER" - Shared USM pointer type with device allocation type allocated by plugin
//!<    "BUFFER_FROM_HANDLE" - Shared buffer from external handle
//!<    "CPU_VA"            - Shared host pointer
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_shared_mem_type;

//!< Read-write property<void *>: device memory handle in a shared memory blob parameter map.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_mem_handle;

//!< Read-write property to pass config file.
OPENVINO_C_VAR(const char*) ov_property_key_intel_gpu_config_file;
