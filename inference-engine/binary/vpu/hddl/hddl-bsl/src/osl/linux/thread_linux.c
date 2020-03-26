// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <errno.h>
#include <pthread.h>

#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <sys/resource.h> /* getrlimit() */
#include <sys/time.h>
#include <unistd.h> /* getpagesize() */
#include "hddl_bsl_thread.h"

int bsl_mutex_init(bsl_mutex_t* mutex) {
#if defined(NDEBUG) || !defined(PTHREAD_MUTEX_ERRORCHECK)
  return pthread_mutex_init(mutex, NULL);
#else
  pthread_mutexattr_t attr;
  int err;

  if (pthread_mutexattr_init(&attr))
    abort();

  if (pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK))
    abort();

  err = pthread_mutex_init(mutex, &attr);

  err = pthread_mutexattr_destroy(&attr);

  return err;
#endif
}

void bsl_mutex_destroy(bsl_mutex_t* mutex) {
  pthread_mutex_destroy(mutex);
}

void bsl_mutex_lock(bsl_mutex_t* mutex) {
  pthread_mutex_lock(mutex);
}

int bsl_mutex_trylock(bsl_mutex_t* mutex) {
  int err;

  err = pthread_mutex_trylock(mutex);
  if (err) {
    if (err != EBUSY && err != EAGAIN)
      printf("try lock != [EBUSY,EAGAIN]!\n");
    return err;
  }

  return 0;
}

void bsl_mutex_unlock(bsl_mutex_t* mutex) {
  pthread_mutex_unlock(mutex);
}
