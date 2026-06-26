// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ov::util {

/**
 * @brief Faults every page of [ptr, ptr + size) resident using up to @p num_threads workers.
 *
 * Splits the range into chunks of at least one mebibyte and touches one byte per page on each
 * worker thread, blocking until all pages are resident. @p ptr and @p size are assumed
 * page-aligned (guaranteed by vm_prefetch's precondition).
 *
 * @param ptr         Page-aligned base address of the range.
 * @param size        Page-aligned byte count to fault in. Must be greater than 0.
 * @param num_threads Number of worker threads to use. Must be greater than 0.
 */
void populate_pages(void* ptr, size_t size, size_t num_threads) noexcept;

}  // namespace ov::util
