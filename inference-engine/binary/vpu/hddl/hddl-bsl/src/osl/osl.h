// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define UNUSED(x) (void)(x)

#ifdef WIN32
#include <process.h>
#include <stdint.h>
#include <windows.h>

#define MAX_PATH_LENGTH _MAX_PATH
typedef CRITICAL_SECTION bsl_mutex_t;

#else
#include <linux/limits.h>
#include <pthread.h>

#define MAX_PATH_LENGTH PATH_MAX
typedef pthread_mutex_t bsl_mutex_t;

#endif

#ifndef WIN32
#define errno_t int
#endif

errno_t bsl_strncpy(char* _Destination, size_t _SizeInBytes, char const* _Source, size_t _MaxCount);

errno_t bsl_fopen(FILE** _Stream, char const* _FileName, char const* _Mode);

bool check_path_is_dir(const char *path);
