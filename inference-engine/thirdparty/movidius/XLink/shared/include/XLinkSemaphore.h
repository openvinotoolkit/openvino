// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///

#ifndef _XLINKSEMAPHORE_H
#define _XLINKSEMAPHORE_H

# if (defined(_WIN32) || defined(_WIN64))
#  include "win_pthread.h"
#  include "win_semaphore.h"
#  include "win_synchapi.h"
# else
#  include <pthread.h>
#  ifdef __APPLE__
#   include "pthread_semaphore.h"
#  else
#   include <semaphore.h>
# endif
# endif

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
    sem_t psem;
    volatile int refs;
} XLink_sem_t;

int XLink_sem_init(XLink_sem_t* sem, int pshared, unsigned int value);
int XLink_sem_destroy(XLink_sem_t* sem);
int XLink_sem_post(XLink_sem_t* sem);
int XLink_sem_wait(XLink_sem_t* sem);
int XLink_sem_timedwait(XLink_sem_t* sem, const struct timespec *abstime);

int XLink_sem_set_refs(XLink_sem_t* sem, int refs);
int XLink_sem_get_refs(XLink_sem_t* sem, int *sval);

int XLink_sem_inc(XLink_sem_t* sem);
int XLink_sem_dec(XLink_sem_t* sem);

#ifdef __cplusplus
}
#endif

#endif  // _XLINKSEMAPHORE_H
