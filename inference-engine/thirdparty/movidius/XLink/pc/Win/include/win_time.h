// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef __cplusplus
extern "C" {
#endif

#pragma once
#include "windows.h"
#define CLOCK_REALTIME      0
#define CLOCK_MONOTONIC     0
#define sleep(x)            Sleep((DWORD)x)
#define usleep(x)           Sleep((DWORD)(x/1000))


int clock_gettime(int, struct timespec *);
#ifdef __cplusplus
}
#endif
