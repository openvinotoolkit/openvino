#include <time.h>
#include <errno.h>
#include <assert.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#include <string.h>

#include "XLinkBlockingQueue.h"
#include "XLinkTool.h"
#include "XLinkStringUtils.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "XLinkLog.h"

static void msToTimespec(struct timespec *ts, unsigned long ms)
{
    ts->tv_sec = ms / 1000;
    ts->tv_nsec = (ms % 1000) * 1000000;
}

#define QUEUE_SIZE (MAX_PACKET_PER_STREAM * 2)

struct BlockingQueue_t {
    char name[MAX_QUEUE_NAME_LENGHT];

    size_t head;
    size_t tail;
    size_t count;
    void* packets[QUEUE_SIZE];
    pthread_mutex_t lock;
    sem_t addPacketSem;
    sem_t removePacketSem;
};

// ------------------------------------
// Private methods declaration. Begin.
// ------------------------------------

static void _blockingQueue_Push(BlockingQueue* queue, void* packet);
static void* _blockingQueue_Pop(BlockingQueue* queue);

// ------------------------------------
// Private methods declaration. End.
// ------------------------------------

// ------------------------------------
// API methods implementation. Begin.
// ------------------------------------

BlockingQueue* BlockingQueue_Create(const char* name) {
    BlockingQueue* ret_blockingQueue = NULL;
    BlockingQueue* blockingQueue = malloc(sizeof(BlockingQueue));

    if (blockingQueue == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate BlockingQueue\n");
        return NULL;
    }

    memset(blockingQueue, 0, sizeof(BlockingQueue));

    mv_strcpy(blockingQueue->name, MAX_QUEUE_NAME_LENGHT, name);
    XLINK_OUT_WITH_LOG_IF(sem_init(&blockingQueue->addPacketSem, 0, 0),
                      mvLog(MVLOG_ERROR, "Cannot initialize addPacketSem\n"));

    XLINK_OUT_WITH_LOG_IF(sem_init(&blockingQueue->removePacketSem, 0, QUEUE_SIZE),
                      mvLog(MVLOG_ERROR, "Cannot initialize removePacketSem\n"));

    XLINK_OUT_WITH_LOG_IF(pthread_mutex_init(&blockingQueue->lock, NULL),
                      mvLog(MVLOG_ERROR, "Cannot initialize lock mutex\n"));

    ret_blockingQueue = blockingQueue;

    XLINK_OUT:
    if(ret_blockingQueue == NULL
       && blockingQueue != NULL) {
        BlockingQueue_Destroy(blockingQueue);
    }
    return ret_blockingQueue;
}

void BlockingQueue_Destroy(BlockingQueue* blockingQueue) {
    if(blockingQueue == NULL) {
        return;
    }

    ASSERT_RC_XLINK(sem_destroy(&blockingQueue->addPacketSem));
    ASSERT_RC_XLINK(sem_destroy(&blockingQueue->removePacketSem));
    ASSERT_RC_XLINK(pthread_mutex_destroy(&blockingQueue->lock));

    free(blockingQueue);

    return;
}

XLinkError_t BlockingQueue_Push(BlockingQueue* queue, void* packet) {
    ASSERT_XLINK(queue != NULL);
    ASSERT_XLINK(packet != NULL);

    mvLog(MVLOG_DEBUG, "%s Waiting to perform operation\n", queue->name);

    sem_wait(&queue->removePacketSem);
    _blockingQueue_Push(queue, packet);

    return X_LINK_SUCCESS;
}

int BlockingQueue_TryPush(BlockingQueue* queue, void* packet) {
    ASSERT_XLINK(queue != NULL);
    ASSERT_XLINK(packet != NULL);

    mvLog(MVLOG_DEBUG, "%s Waiting to perform operation. count=%u\n",
        queue->name, queue->count);

    if(sem_trywait(&queue->removePacketSem) == 0) {
        _blockingQueue_Push(queue, packet);
        return 1;
    }

    return 0;
}

int BlockingQueue_TimedPush(BlockingQueue* queue, void* packet, unsigned long ms) {
    ASSERT_XLINK(queue != NULL);
    ASSERT_XLINK(packet != NULL);

    mvLog(MVLOG_DEBUG, "%s Waiting to perform operation. count=%u\n",
          queue->name, queue->count);

    struct timespec ts;
    msToTimespec(&ts, ms);
    if(sem_timedwait(&queue->removePacketSem, &ts) == 0) {
        _blockingQueue_Push(queue, packet);
        return 1;
    }

    return 0;
}

XLinkError_t BlockingQueue_Pop(BlockingQueue* queue, void** packet) {
    ASSERT_XLINK(queue != NULL);
    ASSERT_XLINK(packet != NULL);

    mvLog(MVLOG_DEBUG, "%s Waiting to perform operation. count=%u\n",
          queue->name, queue->count);

    sem_wait(&queue->addPacketSem);
    *packet = _blockingQueue_Pop(queue);

    return X_LINK_SUCCESS;
}

int BlockingQueue_TryPop(BlockingQueue* queue, void** packet) {
    ASSERT_XLINK(queue != NULL);
    ASSERT_XLINK(packet != NULL);

    mvLog(MVLOG_DEBUG, "%s Waiting to perform operation. count=%u\n",
          queue->name, queue->count);

    if(sem_trywait(&queue->addPacketSem) == 0) {
        *packet = _blockingQueue_Pop(queue);
        return 1;
    }

    return 0;
}

int BlockingQueue_TimedPop(BlockingQueue* queue, void** packet, unsigned long ms) {
    ASSERT_XLINK(queue != NULL);
    ASSERT_XLINK(packet != NULL);

    struct timespec ts;
    msToTimespec(&ts, ms);
    if(sem_timedwait(&queue->addPacketSem, &ts) == 0) {
        *packet = _blockingQueue_Pop(queue);
        return 1;
    }

    return 0;
}

// ------------------------------------
// API methods implementation. End.
// ------------------------------------


// ------------------------------------
// Private methods implementation. Begin.
// ------------------------------------

void _blockingQueue_Push(BlockingQueue* queue, void* packet) {
    pthread_mutex_lock(&queue->lock);
    queue->packets[queue->head++] = packet;
    if (queue->head >= QUEUE_SIZE) {
        queue->head = 0;
    }
    queue->count++;

    mvLog(MVLOG_DEBUG, "%s Successfully completed. count=%u\n",
        queue->name, queue->count);

    pthread_mutex_unlock(&queue->lock);

    sem_post(&queue->addPacketSem);
}

void* _blockingQueue_Pop(BlockingQueue* queue) {
    pthread_mutex_lock(&queue->lock);
    void* packet = queue->packets[queue->tail++];
    if (queue->tail >= QUEUE_SIZE) {
        queue->tail = 0;
    }
    queue->count--;
    mvLog(MVLOG_DEBUG, "%s Successfully completed. count=%u\n",
          queue->name, queue->count);

    pthread_mutex_unlock(&queue->lock);

    sem_post(&queue->removePacketSem);

    return packet;
}

// ------------------------------------
// Private methods implementation. End.
// ------------------------------------
