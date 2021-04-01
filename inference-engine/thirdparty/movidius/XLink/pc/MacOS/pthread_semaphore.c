// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pthread_semaphore.h"

#include <stdlib.h>
#include <pthread.h>
#include <errno.h>
#include <limits.h>

#ifndef SEM_VALUE_MAX
# define SEM_VALUE_MAX INT_MAX
#endif

struct pthread_sem_private_t {
    pthread_mutex_t   access;
    pthread_cond_t    conditional;
    volatile int counter; // >= 0 no waiters, == -1 some waiters
};

int pthread_sem_init(pthread_sem_t *psem, int pshared, unsigned int value) {
    int result = 0;
    if (NULL == psem) {
        errno = EINVAL;
        return -1;
    }
    if (value > SEM_VALUE_MAX){
        errno = EINVAL;
        return -1;
    }
    if (pshared != 0) {
        errno = ENOSYS;
        return -1;
    }
    struct pthread_sem_private_t *psem_private = malloc(sizeof(struct pthread_sem_private_t));
    if (NULL == psem_private) {
        return -1;
    }

    result = pthread_mutex_init(&psem_private->access, NULL);
    if (result) {
        free(psem_private);
        errno = result;
        return -1;
    }

    result = pthread_cond_init(&psem_private->conditional, NULL);
    if (result) {
        pthread_mutex_destroy(&psem_private->access);
        free(psem_private);
        errno = result;
        return -1;
    }

    psem_private->counter = value;

    *psem = (pthread_sem_t)psem_private;
    errno = 0;
    return 0;
}

int pthread_sem_destroy(pthread_sem_t *psem) {
    int result = 0;

    if (NULL == psem) {
        errno = EINVAL;
        return -1;
    }
    if (0 == *psem) {
        errno = EINVAL;
        return -1;
    }

    struct pthread_sem_private_t *psem_private = (struct pthread_sem_private_t *)*psem;

    result = pthread_mutex_lock(&psem_private->access);
    if (result) {
        errno = result;
        return -1;
    }

    if (psem_private->counter == -1) {
        pthread_mutex_unlock(&psem_private->access);
        errno = EBUSY;
        return -1;
    }

    // conditional variable might not be deleted due to wait queue - lets notify users
    result = pthread_cond_destroy(&psem_private->conditional);
    if (result) {
        pthread_mutex_unlock(&psem_private->access);
        errno = result;
        return -1;
    }

    result = pthread_mutex_unlock(&psem_private->access);
    if (result) {
        errno = result;
        return -1;
    }

    // UB - untested if mutex object corrupted
    result = pthread_mutex_destroy(&psem_private->access);
    if (result) {
        errno = result;
        return -1;
    }

    free(psem_private);
    *psem = 0;

    errno = 0;
    return 0;
}
static int pthread_sem_post_signal_or_broadcast(pthread_sem_t *psem, int broadcast) {
    int result;
    if (NULL == psem) {
        errno = EINVAL;
        return -1;
    }
    if (0 == *psem) {
        errno = EINVAL;
        return -1;
    }

    struct pthread_sem_private_t *psem_private = (struct pthread_sem_private_t *)*psem;
    result = pthread_mutex_lock(&psem_private->access);
    if (result) {
        errno = result;
        return -1;
    }

    // right now value == 0 not usually means that there is a waiter queue
    if (broadcast) {
        result = pthread_cond_broadcast(&psem_private->conditional);
    } else {
        result = pthread_cond_signal(&psem_private->conditional);
    }
    if (result) {
        pthread_mutex_unlock(&psem_private->access);
        errno = result;
        return -1;
    }

    // up counter
    if (psem_private->counter == SEM_VALUE_MAX) {
        pthread_mutex_unlock(&psem_private->access);
        errno = EOVERFLOW;
        return -1;
    }
    if (psem_private->counter == -1) {
        psem_private->counter = 1;
    } else {
        psem_private->counter ++;
    }

    result = pthread_mutex_unlock(&psem_private->access);
    if (result) {
        errno = result;
        return -1;
    }

    errno = 0;
    return 0;
}

int pthread_sem_post_broadcast(pthread_sem_t *psem) {
    return pthread_sem_post_signal_or_broadcast(psem, 1);
}

int pthread_sem_post(pthread_sem_t *psem) {
    return pthread_sem_post_signal_or_broadcast(psem, 0);
}

static int pthread_sem_timed_or_blocked_wait(pthread_sem_t *psem, const struct timespec *abstime) {
    int result = 0;
    if (NULL == psem) {
        errno = EINVAL;
        return -1;
    }
    if (0 == *psem) {
        errno = EINVAL;
        return -1;
    }
    struct pthread_sem_private_t *psem_private = (struct pthread_sem_private_t *)*psem;
    result = pthread_mutex_lock(&psem_private->access);
    if (result) {
        errno = result;
        return -1;
    }

    for (;psem_private->counter < 1;) {
        // indicate that we will be waiting this counter
        psem_private->counter = -1;
        if (abstime == NULL) {
            result = pthread_cond_wait(&psem_private->conditional, &psem_private->access);
        } else {
            result = pthread_cond_timedwait(&psem_private->conditional, &psem_private->access, abstime);
        }
        if (result != 0) {
            break;
        }
    }

    // printf("cond_wait=%d\n", result);
    if (result) {
        // sema not obtained - resetting counter back
        if (psem_private->counter == -1) {
            psem_private->counter = 0;
        }
        pthread_mutex_unlock(&psem_private->access);
        errno = result;
        return -1;
    }

    // acquire semaphore
    psem_private->counter --;

    result = pthread_mutex_unlock(&psem_private->access);
    if (result) {
        errno = result;
        return -1;
    }

    errno = 0;
    return 0;
}

int pthread_sem_wait(pthread_sem_t *psem) {
    return pthread_sem_timed_or_blocked_wait(psem, NULL);
}

int pthread_sem_timedwait(pthread_sem_t *psem, const struct timespec *abstime) {
    if (NULL == abstime) {
        errno = EINVAL;
        return -1;
    }
    if (abstime->tv_sec < 0 || abstime->tv_nsec < 0) {
        errno = EINVAL;
        return -1;
    }
    return pthread_sem_timed_or_blocked_wait(psem, abstime);
}


# ifdef __APPLE__

int sem_init(sem_t *psem, int pshared, unsigned int value) {
    return pthread_sem_init(psem, pshared, value);
}
int sem_destroy(sem_t *psem) {
    return pthread_sem_destroy(psem);
}
int sem_post(sem_t *psem) {
    return pthread_sem_post(psem);
}
int sem_wait(sem_t *psem) {
    return pthread_sem_wait(psem);
}
int sem_timedwait(sem_t *psem, const struct timespec *abstime) {
    return pthread_sem_timedwait(psem, abstime);
}

#endif
