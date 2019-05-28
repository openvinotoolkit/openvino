// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

//lightweight semaphone wrapper

#include "win_semaphore.h"
#include "gettime.h"


static int ls_set_errno(int result){
	if (result != 0) {
		errno = result;
		return -1;
		}

	return 0;
}


//Create an semaphore.
int sem_init(sem_t *sem, int pshared, unsigned int value){
	sem_t s = NULL;

	if (sem == NULL || value > (unsigned int) SEM_VALUE_MAX){
		return ls_set_errno(EINVAL);
	}

	if (NULL == (s = (sem_t *)calloc(1, sizeof(*s)))){
		return ls_set_errno(ENOMEM);
	}

	if (pshared != 0){
	    free(s);
		//share between processes
		return ls_set_errno(EPERM);
	}

	if ((s->handle = CreateSemaphoreA(NULL, value, SEM_VALUE_MAX, NULL)) == NULL){
		free(s);
		return ls_set_errno(ENOSPC);
	}

	*sem = s;
	return 0;
}


//Wait for a semaphore
int sem_wait(sem_t *sem){
    if (sem == NULL || *sem == NULL) {
          return ls_set_errno(EINVAL);
    }
	sem_t s = *sem;

	if (WaitForSingleObject(s->handle, INFINITE) != WAIT_OBJECT_0){
		return ls_set_errno(EINVAL);
	}

	return 0;
}


//Wait for a semaphore
int sem_timedwait(sem_t *sem, const struct timespec *ts) {
    if (sem == NULL || *sem == NULL) {
        return ls_set_errno(EINVAL);
    }

    sem_t s = *sem;

	struct timespec cts;
	if (clock_gettime(CLOCK_REALTIME, &cts) == -1) {
		return ls_set_errno(EINVAL);
	}

	unsigned long long t = (ts->tv_sec - cts.tv_sec) * 1000;
	t += (ts->tv_nsec - cts.tv_nsec) / 1000000;

	if (WaitForSingleObject(s->handle, t) != WAIT_OBJECT_0) {
		return ls_set_errno(EINVAL);
	}

	return 0;
}


//Release a semaphone
int sem_post(sem_t *sem){
    if (sem == NULL || *sem == NULL){
        return ls_set_errno(EINVAL);
    }

	sem_t s = *sem;
	if (ReleaseSemaphore(s->handle, 1, NULL) == 0){
		return ls_set_errno(EINVAL);
	}

	return 0;
}



//Destroy a semaphore
int sem_destroy(sem_t *sem){
    if (sem == NULL || *sem == NULL){
        return ls_set_errno(EINVAL);
    }

	sem_t s = *sem;
	if (CloseHandle(s->handle) == 0){
		return ls_set_errno(EINVAL);
	}

	free(s);
	*sem = NULL;
	return 0;
}
