// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PTHREAD_SEMAPHORE_H
#define PTHREAD_SEMAPHORE_H

# include <time.h>
# include <stdint.h>
typedef intptr_t pthread_sem_t;

# ifdef __cplusplus
extern "C" {
# endif
int pthread_sem_init(pthread_sem_t *psem, int pshared, unsigned int value);
int pthread_sem_destroy(pthread_sem_t *psem);
int pthread_sem_post(pthread_sem_t *psem);
int pthread_sem_post_broadcast(pthread_sem_t *psem);
int pthread_sem_wait(pthread_sem_t *psem);
int pthread_sem_timedwait(pthread_sem_t *psem, const struct timespec *abstime);
# ifdef __cplusplus
}
# endif

# ifdef __APPLE__

typedef pthread_sem_t sem_t;

#define SEM_VALUE_MAX 32767

#  ifdef __cplusplus
extern "C" {
#  endif

int sem_init(sem_t *psem, int pshared, unsigned int value);
int sem_destroy(sem_t *psem);
int sem_post(sem_t *psem);
int sem_wait(sem_t *psem);
int sem_timedwait(sem_t *psem, const struct timespec *abstime);

#  ifdef __cplusplus
}
#  endif

# elif defined(_WIN32)
#  error "pthread based semaphores not implemented for WIN32"
# else
#  include <semaphore.h>
# endif  // linux case
#endif  // PTHREAD_SEMAPHORE_H
