// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif

#include "XLinkBlockingQueue.h"
#include "XLinkErrorUtils.h"
#include "XLinkStringUtils.h"
#include "XLinkLog.h"

#if (defined(_WIN32) || defined(_WIN64))
# include "win_time.h"
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif

#include <errno.h>
#include <stdlib.h>
#include <string.h>

static void msToTimespec(struct timespec *ts, unsigned long ms)
{
    ASSERT_XLINK(!clock_gettime(CLOCK_REALTIME, ts));

    ts->tv_nsec += (ms % 1000) * 1000000;
    ts->tv_sec += ms / 1000;
    ts->tv_sec += ts->tv_nsec / 1000000000;
    ts->tv_nsec %= 1000000000;
}

// ------------------------------------
// Private methods declaration. Begin.
// ------------------------------------

static XLinkError_t _blockingQueue_Push(BlockingQueue* queue, void* packet);
static XLinkError_t _blockingQueue_Pop(BlockingQueue* queue, void** packet);

static XLinkError_t _blockingQueue_IncrementPendingCounter(BlockingQueue* queue);
static XLinkError_t _blockingQueue_DecrementPendingCounter(BlockingQueue* queue);

// ------------------------------------
// Private methods declaration. End.
// ------------------------------------

// ------------------------------------
// API methods implementation. Begin.
// ------------------------------------

XLinkError_t BlockingQueue_Create(BlockingQueue* blockingQueue, const char* name) {
    if (blockingQueue == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate BlockingQueue");
        return X_LINK_ERROR;
    }

    memset(blockingQueue, 0, sizeof(BlockingQueue));

    mv_strcpy(blockingQueue->name, MAX_QUEUE_NAME_LENGTH, name);
    for (size_t i = 0; i < blockingQueue->count; ++i) {
        blockingQueue->packets[i] = NULL;
    }
    blockingQueue->front = 0;
    blockingQueue->back = 0;
    blockingQueue->count = 0;

    blockingQueue->pendingToPop = 0;

    if (sem_init(&blockingQueue->addPacketSem, 0, 0) ||
        sem_init(&blockingQueue->removePacketSem, 0, QUEUE_SIZE) ||
        pthread_mutex_init(&blockingQueue->lock, NULL)) {
        mvLog(MVLOG_ERROR, "Cannot initialize synchronization tools, destroying the queue");
        BlockingQueue_Destroy(blockingQueue);
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

void BlockingQueue_Destroy(BlockingQueue* blockingQueue) {
    if (blockingQueue == NULL) {
        mvLog(MVLOG_ERROR, "BlockingQueue_Destroy: queue has been already destroyed");
        return;
    }

    mvLog(MVLOG_DEBUG, "Blocking packet queue with name %s is being destroyed",
          blockingQueue->name);

    if (pthread_mutex_destroy(&blockingQueue->lock)) {
        mvLog(MVLOG_ERROR, "BlockingQueue_Destroy: Cannot destroy lock mutex");
    }
    if (sem_destroy(&blockingQueue->removePacketSem)) {
        mvLog(MVLOG_ERROR, "BlockingQueue_Destroy: Cannot destroy removePacketSem");
    }
    if (sem_destroy(&blockingQueue->addPacketSem)) {
        mvLog(MVLOG_ERROR, "BlockingQueue_Destroy: Cannot destroy addPacketSem");
    }
}

XLinkError_t BlockingQueue_Push(BlockingQueue* queue, void* packet) {
    XLINK_RET_IF(queue == NULL);
    XLINK_RET_IF(packet == NULL);

    mvLog(MVLOG_DEBUG, "BlockingQueue_Push: %s is waiting to perform operation", queue->name);

    XLINK_RET_IF(sem_wait(&queue->removePacketSem));

    return _blockingQueue_Push(queue, packet);
}

XLinkError_t BlockingQueue_TimedPush(BlockingQueue* queue, void* packet, unsigned long ms) {
    XLINK_RET_IF(queue == NULL);
    XLINK_RET_IF(packet == NULL);

    mvLog(MVLOG_DEBUG, "BlockingQueue_TimedPush: %s is waiting to perform operation. count=%u",
          queue->name, queue->count);

    struct timespec ts;
    msToTimespec(&ts, ms);

    int rc = sem_timedwait(&queue->removePacketSem, &ts);
    if (rc && errno == ETIMEDOUT) {
        return X_LINK_TIMEOUT;
    } else if (rc) {
        return X_LINK_ERROR;
    }

    return _blockingQueue_Push(queue, packet);
}

XLinkError_t BlockingQueue_Pop(BlockingQueue* queue, void** packet) {
    XLINK_RET_IF(queue == NULL);
    XLINK_RET_IF(packet == NULL);

    mvLog(MVLOG_DEBUG, "BlockingQueue_Pop: %s is waiting to perform operation. count=%u",
          queue->name, queue->count);

    XLINK_RET_IF(_blockingQueue_IncrementPendingCounter(queue));
    XLINK_RET_IF(sem_wait(&queue->addPacketSem));
    XLINK_RET_IF(_blockingQueue_DecrementPendingCounter(queue));

    return _blockingQueue_Pop(queue, packet);
}

XLinkError_t BlockingQueue_TimedPop(BlockingQueue* queue, void** packet, unsigned long ms) {
    XLINK_RET_IF(queue == NULL);
    XLINK_RET_IF(packet == NULL);

    mvLog(MVLOG_DEBUG, "BlockingQueue_TimedPop: %s is waiting to perform operation. count=%u",
          queue->name, queue->count);

    struct timespec ts;
    msToTimespec(&ts, ms);
    XLINK_RET_IF(_blockingQueue_IncrementPendingCounter(queue));
    int semRc = sem_timedwait(&queue->addPacketSem, &ts);
    XLinkError_t rc;
    if (semRc && (errno == ETIMEDOUT || errno == EINTR)) {
        rc = X_LINK_TIMEOUT;
    } else if (semRc) {
        rc = X_LINK_ERROR;
    } else {
        rc = _blockingQueue_Pop(queue, packet);
    }
    XLINK_RET_IF(_blockingQueue_DecrementPendingCounter(queue));

    return rc;
}

// ------------------------------------
// API methods implementation. End.
// ------------------------------------


// ------------------------------------
// Private methods implementation. Begin.
// ------------------------------------

XLinkError_t _blockingQueue_Push(BlockingQueue* queue, void* packet) {
    XLINK_RET_IF(queue == NULL);
    XLINK_RET_IF(pthread_mutex_lock(&queue->lock));

    queue->packets[queue->back++] = packet;
    if (queue->back >= QUEUE_SIZE) {
        queue->back = 0;
    }
    queue->count++;

    XLINK_RET_IF(sem_post(&queue->addPacketSem));

    mvLog(MVLOG_DEBUG, "_blockingQueue_Push: %s Successfully completed. count=%u",
          queue->name, queue->count);

    XLINK_RET_IF(pthread_mutex_unlock(&queue->lock));

    return X_LINK_SUCCESS;
}

XLinkError_t _blockingQueue_Pop(BlockingQueue* queue, void** packet) {
    XLINK_RET_IF(queue == NULL);
    XLINK_RET_IF(pthread_mutex_lock(&queue->lock));

    *packet = queue->packets[queue->front++];
    if (queue->front >= QUEUE_SIZE) {
        queue->front = 0;
    }
    queue->count--;

    XLINK_RET_IF(sem_post(&queue->removePacketSem));

    mvLog(MVLOG_DEBUG, "_blockingQueue_Pop %s Successfully completed. count=%u",
          queue->name, queue->count);

    XLINK_RET_IF(pthread_mutex_unlock(&queue->lock));

    return X_LINK_SUCCESS;
}

XLinkError_t _blockingQueue_IncrementPendingCounter(BlockingQueue* queue) {
    XLINK_RET_IF(pthread_mutex_lock(&queue->lock));
    queue->pendingToPop++;
    XLINK_RET_IF(pthread_mutex_unlock(&queue->lock));
    return X_LINK_SUCCESS;
}

static XLinkError_t _blockingQueue_DecrementPendingCounter(BlockingQueue* queue) {
    XLINK_RET_IF(pthread_mutex_lock(&queue->lock));
    queue->pendingToPop--;
    XLINK_RET_IF(pthread_mutex_unlock(&queue->lock));
    return X_LINK_SUCCESS;
}

// ------------------------------------
// Private methods implementation. End.
// ------------------------------------
