// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "samples_util/path_util.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool sanitize_path(const char* filename, char* sanitized_path, size_t max_len) {
    if (filename == NULL) {
        printf("[ERROR] File path is NULL.\n");
        return false;
    }

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__)
    if (GetFullPathNameA(filename, (DWORD)max_len, sanitized_path, NULL) == 0) {
        printf("[ERROR] Invalid file path: %lu\n", GetLastError());
        return false;
    }
#else
    if (realpath(filename, sanitized_path) == NULL) {
        printf("[ERROR] Invalid file path: %s\n", strerror(errno));
        return false;
    }
#endif

    // Check if the path length exceeds the maximum allowed length
    if (strlen(sanitized_path) >= max_len) {
        printf("[ERROR] File path is too long.\n");
        return false;
    }

    // Check for directory traversal patterns
    if (strstr(sanitized_path, "..") != NULL) {
        printf("[ERROR] Path contains directory traversal patterns: %s\n", sanitized_path);
        return false;
    }

    // Check can access the file
    FILE* file = fopen(sanitized_path, "r");
    if (file != NULL) {
        fclose(file);
        return true;
    } else {
        printf("[ERROR] File does not exist or cannot be accessed: %s\n", sanitized_path);
        return false;
    }
}
