// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_PACKETQUEUE_H
#define OPENVINO_PACKETQUEUE_H

#define MAX_QUEUE_NAME_LENGHT (64)

#include "XLinkPublicDefines.h"

typedef struct BlockingQueue_t BlockingQueue;

BlockingQueue* BlockingQueue_Create(const char* name);
void BlockingQueue_Destroy(BlockingQueue* dispatcher);

XLinkError_t BlockingQueue_Push(BlockingQueue* queue, void* packet);
int BlockingQueue_TryPush(BlockingQueue* queue, void* packet);
int BlockingQueue_TimedPush(BlockingQueue* queue, void* packet, unsigned long ms);

XLinkError_t BlockingQueue_Pop(BlockingQueue* queue, void** packet);
int BlockingQueue_TryPop(BlockingQueue* queue, void** packet);
int BlockingQueue_TimedPop(BlockingQueue* queue, void** packet, unsigned long ms);

#endif //OPENVINO_PACKETQUEUE_H
