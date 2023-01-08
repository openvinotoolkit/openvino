// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef WIN_SYNCHAPI
#define WIN_SYNCHAPI

#include "win_pthread.h"
#include "synchapi.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PTHREAD_COND_INITIALIZER {0}

typedef struct _pthread_condattr_t pthread_condattr_t;

typedef struct
{
    CONDITION_VARIABLE _cv;
}
pthread_cond_t;

int pthread_cond_init(pthread_cond_t* __cond, const pthread_condattr_t* __cond_attr);
int pthread_cond_destroy(pthread_cond_t* __cond);

int pthread_cond_wait(pthread_cond_t *__cond,
    pthread_mutex_t *__mutex);
int pthread_cond_timedwait(pthread_cond_t* __cond,
    pthread_mutex_t* __mutex,
    const struct timespec* __abstime);

int pthread_cond_signal(pthread_cond_t* __cond);
int pthread_cond_broadcast(pthread_cond_t* __cond);

#ifdef __cplusplus
}
#endif

#endif /* WIN_MUTEX */
