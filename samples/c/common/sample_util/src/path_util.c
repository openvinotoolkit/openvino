// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "samples_util/path_util.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__)
#    include <libloaderapi.h>
#else
#    include <unistd.h>
#    include <libgen.h>
#endif

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

bool get_executable_dir(char* dir_path, size_t size) {
    if (dir_path == NULL || size == 0) {
        return false;
    }

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__MINGW64__)
    char exe_path[MAX_PATH];
    DWORD len = GetModuleFileNameA(NULL, exe_path, MAX_PATH);
    if (len == 0 || len == MAX_PATH) {
        return false;
    }
    // Find last backslash
    char* last_slash = strrchr(exe_path, '\\');
    if (last_slash) {
        size_t dir_len = last_slash - exe_path;
        if (dir_len >= size) {
            return false;
        }
        strncpy(dir_path, exe_path, dir_len);
        dir_path[dir_len] = '\0';
        return true;
    }
    return false;
#else
    char exe_path[PATH_MAX];
    ssize_t len = readlink("/proc/self/exe", exe_path, sizeof(exe_path) - 1);
    if (len == -1) {
        return false;
    }
    exe_path[len] = '\0';
    char* dir = dirname(exe_path);
    if (strlen(dir) >= size) {
        return false;
    }
    strncpy(dir_path, dir, size - 1);
    dir_path[size - 1] = '\0';
    return true;
#endif
}
