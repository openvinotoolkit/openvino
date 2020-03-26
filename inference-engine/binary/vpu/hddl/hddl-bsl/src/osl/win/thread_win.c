// Copyright (C) 2018 Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include "hddl_bsl_thread.h"

int bsl_mutex_init(bsl_mutex_t* mutex) {
  InitializeCriticalSection(mutex);
  return 0;
}

void bsl_mutex_destroy(bsl_mutex_t* mutex) {
  DeleteCriticalSection(mutex);
}

void bsl_mutex_lock(bsl_mutex_t* mutex) {
  EnterCriticalSection(mutex);
}

int bsl_mutex_trylock(bsl_mutex_t* mutex) {
  if (TryEnterCriticalSection(mutex))
    return 0;
  else
    return 1;
}

void bsl_mutex_unlock(bsl_mutex_t* mutex) {
  LeaveCriticalSection(mutex);
}
