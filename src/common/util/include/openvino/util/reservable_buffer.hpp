// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <limits>
#include <string>
#include <string_view>

#include "openvino/util/mmap_object.hpp"

namespace ov::util {
/**
 * \brief Reserves a block of memory of the specified size without actually allocating physical memory.
 * The reserved memory block is not accessible until acquire_buffer() is called on it. The reserved memory block should
 * be released with release_buffer() when it is no longer needed.
 * \param size [in] Size of the memory block to reserve in bytes. Must be greater than 0.
 * \param error [out] Optional Error message in case of failure, empty otherwise.
 * \return Pointer to the reserved memory block, or nullptr if the reservation fail.
 */
void* reserve_buffer(size_t size, std::string* error = nullptr) noexcept;

/**
 * \brief Acquires the reserved memory block, making it accessible for read/write operations.
 * \param reserved_buffer [in] Pointer to the reserved memory block to acquire.
 * \param size [in] Size of the memory block to acquire in bytes. Must be greater than 0.
 * \param error [out] Optional Error message in case of failure, empty otherwise.
 * \note The reserved memory block should be successfully acquired before it can be used.
 */
void acquire_buffer(void* reserved_buffer, size_t size, std::string* error = nullptr) noexcept;

/**
 * \brief Evicts the acquired memory block, making it inaccessible and allowing the system to free physical memory.
 * \param reserved_buffer [in] Pointer to the reserved memory block to evict.
 * \param size [in] Size of the memory block to evict in bytes. Must be greater than 0.
 * \param error [out] Optional Error message in case of failure, empty otherwise.
 * \note After eviction, the reserved memory block can be acquired again with acquire_buffer().
 */
void evict_buffer(void* reserved_buffer, size_t size, std::string* error = nullptr) noexcept;

/**
 * \brief Releases the reserved memory block, freeing any associated resources. After this call, the reserved memory
 * block is no longer valid and should not be used.
 * \param reserved_buffer [in] Pointer to the reserved memory block to release.
 * \param size [in] Size of the memory block to release in bytes. Must be greater than 0.
 * \param error [out] Optional Error message in case of failure, empty otherwise.
 */
void release_buffer(void* reserved_buffer, size_t size, std::string* error = nullptr) noexcept;

}  // namespace ov::util
