// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "XLinkPacket.h"
#include "XLinkTool.h"
#include "XLinkPlatform.h"
#include "XLinkMacros.h"
#include "XLinkStringUtils.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME XLinkPacket
#endif
#include "XLinkLog.h"

// ------------------------------------
// Packet API implementation. Begin.
// ------------------------------------

PacketNew* Packet_Create(PacketPool* packetPool, packetIdx_t idx, streamId_t streamId) {
    XLINK_RET_WITH_ERR_IF(packetPool == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(streamId >= MAX_STREAM_NAME_LENGTH, NULL);

    PacketNew* ret_packet = NULL;
    PacketNew* packet = malloc(sizeof(PacketNew));
    if (packet == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate StreamPacketPool\n");
        return ret_packet;
    }

    memset(packet, 0, sizeof(PacketNew));

    packet->header.streamId = streamId;
    packet->privateFields.idx = idx;
    packet->privateFields.status = PACKET_FREE;
    packet->privateFields.isUserData = 0;
    packet->privateFields.packetPool = packetPool;
    XLINK_OUT_WITH_LOG_IF(sem_init(&packet->privateFields.completedSem, 0, 0),
                      mvLog(MVLOG_ERROR, "Cannot initialize addPacketSem\n"));

    ret_packet = packet;

    XLINK_OUT:
    if(ret_packet == NULL
        && packet != NULL) {
        Packet_Destroy(packet);
    }
    return ret_packet;
}

void Packet_Destroy(PacketNew* packet) {
    if(packet == NULL) {
        return;
    }

    ASSERT_RC_XLINK(Packet_ReleaseData(packet));
    ASSERT_RC_XLINK(sem_destroy(&packet->privateFields.completedSem));
    free(packet);

    return;
}

XLinkError_t Packet_Release(PacketNew* packet) {
    XLINK_RET_IF(packet == NULL);

    PacketPool* packetPool = packet->privateFields.packetPool;
    ASSERT_XLINK(packetPool);

    Packet_ReleaseData(packet);
    PacketPool_ReleasePacket(packetPool, packet);

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_FreePending(PacketNew* packet, packetStatus_t status) {
    XLINK_RET_IF(packet == NULL);

    packet->privateFields.status = status;
    mvLog(MVLOG_DEBUG, "Post complete semaphore\n");
    if (sem_post(&packet->privateFields.completedSem)) {
        mvLog(MVLOG_ERROR, "can't post completedSem semaphore\n");
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_SetData(PacketNew* packet, void* data, int size) {
    XLINK_RET_IF(packet == NULL);

    mvLog(MVLOG_DEBUG, "id=%d, idx=%d, data=%p, size=%d\n",
          packet->header.id, packet->privateFields.idx, data, size);

    packet->header.size = size;
    packet->data = data;
    packet->privateFields.isUserData = 1;

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_AllocateData(PacketNew* packet) {
    XLINK_RET_IF(packet == NULL);
    XLINK_RET_IF(packet->header.size < 0);
    ASSERT_XLINK(packet->data == NULL);

    mvLog(MVLOG_DEBUG, "id=%d, idx=%d, size=%d\n",
          packet->header.id, packet->privateFields.idx, packet->header.size);

    if(packet->header.size != 0) {
        void* data = XLinkPlatformAllocateData(
            ALIGN_UP(packet->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);

        XLINK_RET_IF(data == NULL);
        packet->data = data;
    } else {
        packet->data = NULL;
    }

    packet->privateFields.isUserData = 0;

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_ReleaseData(PacketNew* packet) {
    XLINK_RET_IF(packet == NULL);

    //mvLog(MVLOG_DEBUG, "id=%d, idx=%d, data=%p, size=%d, isUserData=%d\n",
    //      packet->header.id, packet->privateFields.idx,
    //      packet->data, packet->header.size, packet->privateFields.isUserData);

    if(packet->privateFields.isUserData
        || packet->data == NULL) {
        packet->data = NULL;
        return X_LINK_SUCCESS;
    }

    XLinkPlatformDeallocateData(packet->data,
        ALIGN_UP(packet->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
    packet->data = NULL;

    return X_LINK_SUCCESS;
}

packetCommType_t Packet_GetCommType(PacketNew* packet) {
    ASSERT_XLINK(packet);

    if(packet->header.type < XLINK_REQUEST_LAST) {
        return PACKET_REQUEST;
    } else {
        PACKET_RESPONSE;
    }
}

// ------------------------------------
// Packet API implementation. End.
// ------------------------------------

// ------------------------------------
// PacketPool API implementation. Begin.
// ------------------------------------

#define FREE_PACKET_FLAG 0
#define BUSY_PACKET_FLAG 1

typedef struct PacketPool_t {
    char name[MAX_STREAM_NAME_LENGTH];

    int freePacketsIdx[MAX_PACKET_PER_STREAM];
    PacketNew* packets[MAX_PACKET_PER_STREAM];

    pthread_mutex_t packetAccessLock;
    packetId_t nexUniqueId;

    //Metrics
    size_t busyPacketsCount;
} PacketPool;

static int getFirstFreePacket(PacketPool* packetPool);

PacketPool* PacketPool_Create(streamId_t streamId, const char* name) {
    PacketPool *ret_packetPool = NULL;
    PacketPool *packetPool = malloc(sizeof(PacketPool));

    if (packetPool == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate PacketPool\n");
        return ret_packetPool;
    }

    memset(packetPool, 0, sizeof(PacketPool));

    mv_strcpy(packetPool->name, MAX_STREAM_NAME_LENGTH, name);
    packetPool->nexUniqueId = rand() % (streamId + 1); // TODO. Make "right" unique id generator
    for (int i = 0; i < MAX_PACKET_PER_STREAM; ++i) {
        PacketNew* packet = Packet_Create(packetPool, i, streamId);
        XLINK_OUT_IF(packet == NULL);

        packetPool->packets[i] = packet;
        packetPool->freePacketsIdx[i] = FREE_PACKET_FLAG;
    }

    XLINK_OUT_WITH_LOG_IF(pthread_mutex_init(&packetPool->packetAccessLock, NULL),
                          mvLog(MVLOG_ERROR, "Cannot initialize lock mutex\n"));

    ret_packetPool = packetPool;

    XLINK_OUT:
    if(ret_packetPool == NULL
       && packetPool != NULL) {
        PacketPool_Destroy(packetPool);
    }
    return ret_packetPool;
}

void PacketPool_Destroy(PacketPool* packetPool) {
    if(packetPool == NULL) {
        return;
    }

    for (int i = 0; i < MAX_PACKET_PER_STREAM; ++i) {
        Packet_Destroy(packetPool->packets[i]);
    }

    ASSERT_RC_XLINK(pthread_mutex_destroy(&packetPool->packetAccessLock));

    free(packetPool);
    return;
}

PacketNew* PacketPool_GetPacket(PacketPool* packetPool) {
    XLINK_RET_WITH_ERR_IF(packetPool == NULL, NULL);
    PacketNew* ret_packet = NULL;

    ASSERT_RC_XLINK(pthread_mutex_lock(&packetPool->packetAccessLock));
    int freeIdx = getFirstFreePacket(packetPool);
    XLINK_OUT_IF(freeIdx == MAX_PACKET_PER_STREAM);

    packetPool->busyPacketsCount++;
    packetPool->freePacketsIdx[freeIdx] = BUSY_PACKET_FLAG;
    ret_packet = packetPool->packets[freeIdx];
    ret_packet->privateFields.status = PACKET_PROCESSING;
    ret_packet->header.id = packetPool->nexUniqueId++;

    mvLog(MVLOG_DEBUG, "%s Locked packet. Busy packets count: %u. Packet: id=%d, idx=%d\n",
          packetPool->name, packetPool->busyPacketsCount, ret_packet->header.id, ret_packet->privateFields.idx);
    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&packetPool->packetAccessLock));
    ASSERT_XLINK(ret_packet);

    return ret_packet;
}

XLinkError_t PacketPool_ReleasePacket(PacketPool* packetPool, PacketNew* packet) {
    XLINK_RET_IF(packetPool == NULL);
    XLINK_RET_IF(packet == NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&packetPool->packetAccessLock));

    packet->privateFields.status = PACKET_FREE;
    packetPool->freePacketsIdx[packet->privateFields.idx] = FREE_PACKET_FLAG;
    packetPool->busyPacketsCount--;

    mvLog(MVLOG_DEBUG, "%s Released packet. Busy packets count: %u. Packet: id=%d, idx=%d\n",
          packetPool->name, packetPool->busyPacketsCount, packet->header.id, packet->privateFields.idx);

    ASSERT_RC_XLINK(pthread_mutex_unlock(&packetPool->packetAccessLock));
    return X_LINK_SUCCESS;
}

PacketNew* PacketPool_FindPacket(PacketPool* packetPool, packetId_t id) {
    XLINK_RET_WITH_ERR_IF(packetPool == NULL, NULL);

    for (int i = 0; i < MAX_PACKET_PER_STREAM; ++i) {
        if(packetPool->packets[i]->header.id == id
           && packetPool->packets[i]->privateFields.status == PACKET_PENDING) {
            return packetPool->packets[i];
        }
    }

    mvLog(MVLOG_DEBUG, "%s  Cannot find pending packet. id=%d", packetPool->name, id);
    return NULL;
}

XLinkError_t PacketPool_FreePendingPackets(PacketPool* packetPool, packetStatus_t status) {
    XLINK_RET_IF(packetPool == NULL);

    for (int i = 0; i < MAX_PACKET_PER_STREAM; ++i) {
        if(packetPool->packets[i]->privateFields.status == PACKET_PENDING) {
            XLINK_RET_IF(Packet_FreePending(packetPool->packets[i], status));
        }
    }

    return X_LINK_SUCCESS;
}

XLinkError_t PacketPool_SetStreamName(PacketPool* packetPool, const char* streamName) {
    XLINK_RET_IF(packetPool == NULL);
    XLINK_RET_IF(streamName == NULL);

    for (int i = 0; i < MAX_PACKET_PER_STREAM; ++i) {
        mv_strcpy(packetPool->packets[i]->header.streamName, MAX_STREAM_NAME_LENGTH, streamName);
    }

    return X_LINK_SUCCESS;
}

int getFirstFreePacket(PacketPool* packetPool) {
    ASSERT_XLINK(packetPool != NULL);

    for (int i = 0; i < MAX_PACKET_PER_STREAM; ++i) {
        if(packetPool->freePacketsIdx[i] == FREE_PACKET_FLAG) {
            return i;
        }
    }

    return MAX_PACKET_PER_STREAM;
}

// ------------------------------------
// PacketPool API implementation. End.
// ------------------------------------
