// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>
#include "ie_api.h"

/**
 * @brief Copies bytes between buffers with security enhancements
 * Copies count bytes from src to dest. If the source and destination
 * overlap, the behavior is undefined.
 * @param dest
 * pointer to the object to copy to
 * @param destsz
 * max number of bytes to modify in the destination (typically the size
 * of the destination object)
 * @param src
 pointer to the object to copy from
 * @param count
 number of bytes to copy
 @return zero on success and non-zero value on error.
 */

INFERENCE_ENGINE_API_CPP(int) simple_copy(void* dest, size_t destsz, void const* src, size_t count);
