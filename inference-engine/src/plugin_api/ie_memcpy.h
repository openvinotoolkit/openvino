// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief Defines a ie_memcpy to safely (SDL-friendly) copy arrays 
 * @file ie_memcpy.h
 */

#pragma once

#include <stdlib.h>

#include "ie_api.h"

/**
 * @brief      Copies bytes between buffers with security enhancements
 *             Copies count bytes from src to dest. If the source and destination
 *             overlap, the behavior is undefined.
 * @ingroup    ie_dev_api_memory 
 * 
 * @param dest A Pointer to the object to copy to
 * @param destsz A max number of bytes to modify in the destination (typically the size
 *               of the destination object)
 * @param src A pointer to the object to copy from
 * @param count A number of bytes to copy
 * 
 * @return zero on success and non-zero value on error.
 */

INFERENCE_ENGINE_API_CPP(int) ie_memcpy(void* dest, size_t destsz, void const* src, size_t count);
