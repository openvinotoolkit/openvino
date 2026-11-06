// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a specified header file for auto plugin's properties
 *
 * @file properties.h
 */

#pragma once
#include "openvino/c/ov_common.h"

/**
 * @brief Read-write property<string> for setting that enables performance improvement by binding
 * buffer to hw infer request
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_auto_device_bind_buffer;

/**
 * @brief Read-write property<string> to enable/disable CPU as accelerator (or helper device) at the beginning
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_auto_enable_startup_fallback;

/**
 * @brief Read-write property<string> to enable/disable runtime fallback to other devices when infer fails
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_auto_enable_runtime_fallback;

/**
 * @brief Boolean property to enable background cache warm-up for other eligible AUTO candidate devices.
 * In the C API, pass the string value "YES" or "NO". Requires ov::cache_dir or ov::cache_path; the property is
 * ignored in cumulative throughput mode or when ov::cache_blob_id is set. When active, startup and runtime fallbacks
 * are disabled for this compile.
 * @ingroup ov_property_c_api
 */
OPENVINO_C_VAR(const char*)
ov_property_key_intel_auto_compile_for_all;