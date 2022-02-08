// Copyright (C) 2018-2022 Intel Corporation
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
#  include <unistd.h>
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

//
// This structure describes the semaphore used in XLink and
// extends the standard semaphore with a reference count.
// The counter is thread-safe and changes only in cases if
// all tools of thread synchronization are really unlocked.
// refs == -1 in case if semaphore was destroyed;
// refs == 0 in case if semaphore was initialized but has no waiters;
// refs == N in case if there are N waiters which called sem_wait().
//

typedef struct {
    sem_t psem;
    int refs;
} XLink_sem_t;

//
// XLink wrappers for POSIX semaphore functions (refer sem_overview for details)
// In description of standard sem_destroy the following can be noted:
// "Destroying a semaphore that other processes or threads are currently
// blocked on (in sem_wait(3)) produces undefined behavior."
// XLink wrappers use thread-safe reference count and destroy the semaphore only in case
// if there are no waiters
//

int XLink_sem_init(XLink_sem_t* sem, int pshared, unsigned int value);
int XLink_sem_destroy(XLink_sem_t* sem);
int XLink_sem_post(XLink_sem_t* sem);
int XLink_sem_wait(XLink_sem_t* sem);
int XLink_sem_timedwait(XLink_sem_t* sem, const struct timespec* abstime);
int XLink_sem_trywait(XLink_sem_t* sem);

//
// Helper functions for XLink semaphore wrappers.
// Use them only in case if you know what you are doing.
//

int XLink_sem_set_refs(XLink_sem_t* sem, int refs);
int XLink_sem_get_refs(XLink_sem_t* sem, int *sval);

#ifdef __cplusplus
}
#endif

#endif  // _XLINKSEMAPHORE_H
