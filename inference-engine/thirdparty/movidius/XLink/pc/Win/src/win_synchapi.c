// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "win_synchapi.h"

int pthread_cond_init(pthread_cond_t* __cond, const pthread_condattr_t* __cond_attr)
{
    if (__cond == NULL) {
        return ERROR_INVALID_HANDLE;
    }

    (void)__cond_attr;
    InitializeConditionVariable(&__cond->_cv);
    return 0;
}

int pthread_cond_destroy(pthread_cond_t* __cond)
{
    (void)__cond;
    return 0;
}

int pthread_cond_wait(pthread_cond_t *__cond,
    pthread_mutex_t *__mutex)
{
    if (__cond == NULL || __mutex == NULL)
        return ERROR_INVALID_HANDLE;
    return pthread_cond_timedwait(__cond, __mutex, NULL);
}

int pthread_cond_timedwait(pthread_cond_t* __cond,
    pthread_mutex_t* __mutex,
    const struct timespec* __abstime)
{
    if (__cond == NULL) {
        return ERROR_INVALID_HANDLE;
    }

    long long msec = INFINITE;
    if (__abstime != NULL) {
        msec = __abstime->tv_sec * 1000 + __abstime->tv_nsec / 1000000;
    }

    // SleepConditionVariableCS returns bool=true on success.
    if (SleepConditionVariableCS(&__cond->_cv, __mutex, (DWORD)msec))
        return 0;

    const int rc = (int)GetLastError();
    return rc == ERROR_TIMEOUT ? ETIMEDOUT : rc;
}

int pthread_cond_signal(pthread_cond_t *__cond)
{
    if (__cond == NULL) {
        return ERROR_INVALID_HANDLE;
    }

    WakeConditionVariable(&__cond->_cv);
    return 0;
}

int pthread_cond_broadcast(pthread_cond_t *__cond)
{
    if (__cond == NULL) {
        return ERROR_INVALID_HANDLE;
    }

    WakeAllConditionVariable(&__cond->_cv);
    return 0;
}
