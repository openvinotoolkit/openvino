// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/properties.hpp"

namespace ov {
namespace intel_npu {

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of already allocated NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_alloc_mem_size{"NPU_DEVICE_ALLOC_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint64_t
 * Read-only property to get size of available NPU DDR memory (both for discrete/integrated NPU devices)
 *
 * Note: Queries driver both for discrete/integrated NPU devices
 */
static constexpr ov::Property<uint64_t, ov::PropertyMutability::RO> device_total_mem_size{"NPU_DEVICE_TOTAL_MEM_SIZE"};

/**
 * @brief [Only for NPU plugin]
 * Type: uint32_t
 * Read-only property to get NPU driver version (for both discrete/integrated NPU devices)
 */
static constexpr ov::Property<uint32_t, ov::PropertyMutability::RO> driver_version{"NPU_DRIVER_VERSION"};

}  // namespace intel_npu
}  // namespace ov
