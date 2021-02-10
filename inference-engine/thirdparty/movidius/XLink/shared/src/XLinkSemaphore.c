// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "XLinkSemaphore.h"
#include "XLinkErrorUtils.h"
#include "XLinkLog.h"

static pthread_mutex_t ref_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t ref_cond = PTHREAD_COND_INITIALIZER;

int XLink_sem_inc(XLink_sem_t* sem)
{
    XLINK_RET_IF_FAIL(pthread_mutex_lock(&ref_mutex));
    if (sem->refs < 0) {
        // Semaphore has been already destroyed
        XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));
        return -1;
    }

    sem->refs++;
    XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));

    return 0;
}

int XLink_sem_dec(XLink_sem_t* sem)
{
    XLINK_RET_IF_FAIL(pthread_mutex_lock(&ref_mutex));
    if (sem->refs < 1) {
        // Can't decrement reference count if there are no waiters
        // or semaphore has been already destroyed
        XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));
        return -1;
    }

    sem->refs--;
    int ret = pthread_cond_broadcast(&ref_cond);
    XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));

    return ret;
}


int XLink_sem_init(XLink_sem_t* sem, int pshared, unsigned int value)
{
    XLINK_RET_ERR_IF(sem == NULL, -1);

    XLINK_RET_IF_FAIL(sem_init(&sem->psem, pshared, value));
    XLINK_RET_IF_FAIL(pthread_mutex_lock(&ref_mutex));
    sem->refs = 0;
    XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));

    return 0;
}

int XLink_sem_destroy(XLink_sem_t* sem)
{
    XLINK_RET_ERR_IF(sem == NULL, -1);

    XLINK_RET_IF_FAIL(pthread_mutex_lock(&ref_mutex));
    if (sem->refs < 0) {
        // Semaphore has been already destroyed
        XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));
        return -1;
    }

    while(sem->refs > 0) {
        if (pthread_cond_wait(&ref_cond, &ref_mutex)) {
            break;
        };
    }
    sem->refs = -1;
    int ret = sem_destroy(&sem->psem);
    XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));

    return ret;
}

int XLink_sem_post(XLink_sem_t* sem)
{
    XLINK_RET_ERR_IF(sem == NULL, -1);
    if (sem->refs < 0) {
        return -1;
    }

    return sem_post(&sem->psem);
}

int XLink_sem_wait(XLink_sem_t* sem)
{
    XLINK_RET_ERR_IF(sem == NULL, -1);

    XLINK_RET_IF_FAIL(XLink_sem_inc(sem));
    int ret = sem_wait(&sem->psem);
    XLINK_RET_IF_FAIL(XLink_sem_dec(sem));

    return ret;
}

int XLink_sem_timedwait(XLink_sem_t* sem, const struct timespec* abstime)
{
    XLINK_RET_ERR_IF(sem == NULL, -1);
    XLINK_RET_ERR_IF(abstime == NULL, -1);

    XLINK_RET_IF_FAIL(XLink_sem_inc(sem));
    int ret = sem_timedwait(&sem->psem, abstime);
    XLINK_RET_IF_FAIL(XLink_sem_dec(sem));

    return ret;
}

int XLink_sem_set_refs(XLink_sem_t* sem, int refs)
{
    XLINK_RET_ERR_IF(sem == NULL, -1);
    XLINK_RET_ERR_IF(refs < -1, -1);

    XLINK_RET_IF_FAIL(pthread_mutex_lock(&ref_mutex));
    sem->refs = refs;
    int ret = pthread_cond_broadcast(&ref_cond);
    XLINK_RET_IF_FAIL(pthread_mutex_unlock(&ref_mutex));

    return ret;
}

int XLink_sem_get_refs(XLink_sem_t* sem, int *sval)
{
    XLINK_RET_ERR_IF(sem == NULL, -1);

    *sval = sem->refs;
    return 0;
}
