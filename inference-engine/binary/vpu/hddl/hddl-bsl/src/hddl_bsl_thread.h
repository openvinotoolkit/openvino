// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __HDDL_BSL_THREAD_H__
#define __HDDL_BSL_THREAD_H__

#include <sys/stat.h>
#if defined(WIN32)

#include <process.h>
#include <stdint.h>
#include <windows.h>

#define sleep(seconds) Sleep((seconds))
#define usleep(us) Sleep((us) / 1000)

typedef CRITICAL_SECTION bsl_mutex_t;
#else

#include <pthread.h>
#include <unistd.h>
typedef pthread_mutex_t bsl_mutex_t;

#endif
// thread
int bsl_mutex_init(bsl_mutex_t* mutex);
void bsl_mutex_destroy(bsl_mutex_t* mutex);
void bsl_mutex_lock(bsl_mutex_t* mutex);
int bsl_mutex_trylock(bsl_mutex_t* mutex);
void bsl_mutex_unlock(bsl_mutex_t* mutex);

#endif
