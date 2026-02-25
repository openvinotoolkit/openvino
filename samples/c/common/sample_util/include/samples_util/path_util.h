// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef _WIN32
#    include <windows.h>
#    define PATH_MAX MAX_PATH
typedef BOOLEAN WIN_BOOLEAN;
#else
#    include <limits.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief  Sanitizes and validate a file path.
 * @param filename The file path to be sanitized.
 * @param sanitized_path The buffer to store the sanitized path.
 * @param size The size of the buffer.
 * @return true if the path is valid, false otherwise.
 */
bool sanitize_path(const char* filename, char* sanitized_path, size_t size);

#ifdef __cplusplus
}
#endif
