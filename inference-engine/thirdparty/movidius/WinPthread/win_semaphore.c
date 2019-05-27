/*
* Copyright 2017-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

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
