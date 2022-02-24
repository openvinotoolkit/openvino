// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _SEMAPHORE_H_
#define _SEMAPHORE_H_


#include <errno.h>
#include <fcntl.h>
#include <windows.h>
#include <stdio.h>


#if !defined(malloc)
#include <malloc.h>
#endif
#if !defined(INT_MAX)
#include <limits.h>
#endif


#ifndef SEM_VALUE_MAX
#define SEM_VALUE_MAX           INT_MAX
#endif


#if (_MSC_VER == 1800)
struct timespec
{
    /* long long in windows is the same as long in unix for 64bit */
    long long tv_sec;
    long long tv_nsec;
};
#elif (_MSC_VER >= 1800)
#include "time.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct sem_t_
{
    HANDLE handle;
};
typedef struct sem_t_ * sem_t;


int sem_init(sem_t *sem, int pshared, unsigned int value);
int sem_wait(sem_t *sem);
int sem_timedwait(sem_t *sem, const struct timespec *ts);
int sem_trywait(sem_t *sem);
int sem_post(sem_t *sem);
int sem_destroy(sem_t *sem);


#ifdef __cplusplus
}
#endif


#endif /* _SEMAPHORE_H_ */
