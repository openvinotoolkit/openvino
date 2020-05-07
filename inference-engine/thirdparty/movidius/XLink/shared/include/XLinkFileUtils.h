// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINK_FILE_UTILS_H
#define _XLINK_FILE_UTILS_H

#include <stddef.h>
#include <stdio.h>

#ifdef _WIN32
#include <wchar.h>
#else
#include <unistd.h>
#include <fcntl.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

FILE* utf8_fopen(const char* filename, const char* mode);
int utf8_open(const char* filename, int flag, int mode);
int utf8_access(const char* filename, int mode);

/// Returns 1 on success, 0 on failure.
int utf8_getenv_s(size_t bufferSize, char* buffer, const char* varName);

/// Returns 1 on success, 0 on failure.
int utf8_shared_lib_path(size_t bufferSize, char* buffer);

#ifdef __cplusplus
}
#endif

#endif //_XLINK_FILE_UTILS_H
