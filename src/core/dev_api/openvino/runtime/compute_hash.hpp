// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>

namespace ov {
namespace runtime {

/**
 * @brief Computes the hash value for the input data
 * @param src  A pointer to the input data
 * @param size The length of the input data in bytes
 */
size_t compute_hash(const void* src, size_t size);

}  // namespace runtime
}  // namespace ov
