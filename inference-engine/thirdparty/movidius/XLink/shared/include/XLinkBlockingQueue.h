// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINK_BLOCKINGQUEUE_H
#define OPENVINO_XLINK_BLOCKINGQUEUE_H

#include "XLinkPublicDefines.h"
#include "XLinkSemaphore.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define MAX_QUEUE_NAME_LENGTH (64)
#define QUEUE_SIZE (XLINK_MAX_PACKET_PER_STREAM * 2)

typedef struct BlockingQueue_t {
    char name[MAX_QUEUE_NAME_LENGTH];
    void* packets[QUEUE_SIZE];

    size_t front;
    size_t back;
    size_t count;

    pthread_mutex_t lock;
    sem_t addPacketSem;
    sem_t removePacketSem;
    int pendingToPop;
} BlockingQueue;

XLinkError_t BlockingQueue_Create(
        BlockingQueue* blockingQueue,
        const char* name);
void BlockingQueue_Destroy(
        BlockingQueue* dispatcher);

XLinkError_t BlockingQueue_Push(
        BlockingQueue* queue,
        void* packet);
XLinkError_t BlockingQueue_TimedPush(
        BlockingQueue* queue,
        void* packet,
        unsigned long ms);

XLinkError_t BlockingQueue_Pop(
        BlockingQueue* queue,
        void** packet);
XLinkError_t BlockingQueue_TimedPop(
        BlockingQueue* queue,
        void** packet,
        unsigned long ms);

#endif  // OPENVINO_XLINK_BLOCKINGQUEUE_H
