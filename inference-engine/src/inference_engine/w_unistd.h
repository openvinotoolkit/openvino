// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if defined(_WIN32)

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include <winsock2.h>
#include <windows.h>
#include <stdlib.h>
#include <process.h>
#include <direct.h>
#include <io.h>
#include <chrono>

#define strncasecmp _strnicmp
#define getcwd _getcwd
#define fileno _fileno

#define SecuredGetEnv GetEnvironmentVariableA

#if defined usleep
#undef usleep
#endif

#define usleep(m) std::this_thread::sleep_for(std::chrono::microseconds(m))

#else

#include <unistd.h>
#include <cstdlib>
#include <string.h>

static inline int SecuredGetEnv(const char *envName, char *buf, int bufLen) {
    char *pe = getenv(envName);
    if (!pe) return 0;
    strncpy(buf, pe, bufLen - 1);
    buf[bufLen - 1] = 0;
    return strlen(buf);
}

#endif


