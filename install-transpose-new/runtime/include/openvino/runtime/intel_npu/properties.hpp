// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header for advanced hardware related properties for NPU plugin
 *        To use in set_property, compile_model, import_model, get_property methods
 *
 * @file openvino/runtime/intel_npu/properties.hpp
 */
#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {

/**
 * @defgroup ov_runtime_npu_prop_cpp_api Intel NPU specific properties
 * @ingroup ov_runtime_cpp_api
 * Set of Intel NPU specific properties.
 */

/**
 * @brief Namespace with Intel NPU specific properties
 */
namespace intel_npu {

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of already allocated NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_alloc_mem_size{"NPU_DEVICE_ALLOC_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of available NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_total_mem_size{"NPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint32_t
 * Read-only property to get NPU driver version (for both discrete/integrated NPU devices)
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<uint32_t, ov::PropertyMutability::RO> driver_version{"NPU_DRIVER_VERSION"};

/**
 * @brief [Only for NPU compiler]
 * Type: std::string
 * Set various parameters supported by the NPU compiler.
 * @ingroup ov_runtime_npu_prop_cpp_api
 */
static constexpr ov::Property<std::string> compilation_mode_params{"NPU_COMPILATION_MODE_PARAMS"};

}  // namespace intel_npu
}  // namespace ov
