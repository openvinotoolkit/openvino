// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME XLinkPacket
#endif
#include "XLinkLog.h"

#include "XLinkErrorUtils.h"
#include "XLinkPacket.h"
#include "XLinkPlatform.h"
#include "XLinkMacros.h"
#include "XLinkStringUtils.h"

#include <stdlib.h>
#include <string.h>
#include <XLinkPrivateFields.h>

// ------------------------------------
// Packet API implementation. Begin.
// ------------------------------------

XLinkError_t Packet_Create(Packet* packet, PacketPool* packetPool, packetIdx_t idx, streamId_t streamId) {
    XLINK_RET_IF(streamId >= XLINK_MAX_STREAMS);
    XLINK_RET_IF(packet == NULL);

    memset(packet, 0, sizeof(Packet));

    packet->header.streamId = streamId;
    packet->privateFields.idx = idx;
    packet->privateFields.status = PACKET_FREE;
    packet->privateFields.isUserData = 0;
    packet->privateFields.blockingType = PACKET_NON_BLOCKING;
    packet->privateFields.packetPool = packetPool;

    if (sem_init(&packet->privateFields.completedSem, 0, 0)) {
        mvLog(MVLOG_ERROR, "Cannot initialize completedSem");
        Packet_Destroy(packet);
        return X_LINK_ERROR;
    }

    if (pthread_mutex_init(&packet->privateFields.packetLock, NULL)) {
        mvLog(MVLOG_ERROR, "Cannot initialize packetLock");
        Packet_Destroy(packet);
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

void Packet_Destroy(Packet* packet) {
    ASSERT_XLINK(packet);

    ASSERT_XLINK(packet->privateFields.status == PACKET_FREE);
    if (sem_destroy(&packet->privateFields.completedSem)) {
        mvLog(MVLOG_ERROR, "Cannot destroy completedSem");
    }
    if (pthread_mutex_destroy(&packet->privateFields.packetLock)) {
        mvLog(MVLOG_ERROR, "Cannot destroy packetLock");
    }
}

XLinkError_t Packet_Release(Packet* packet) {
    XLINK_RET_IF(packet == NULL);

    Packet_ReleaseData(packet);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));
    PacketPool* packetPool = packet->privateFields.packetPool;
    ASSERT_XLINK(packetPool);
    ASSERT_XLINK(!pthread_mutex_lock(&packetPool->packetAccessLock));

    packet->privateFields.status = PACKET_FREE;
    mv_strcpy(packet->header.streamName, MAX_STREAM_NAME_LENGTH, packetPool->name);
    packetPool->busyPacketsCount--;

    mvLog(MVLOG_DEBUG, "%s Released packet. Busy packets count: %u. Packet: id=%d, idx=%d",
          packetPool->name, packetPool->busyPacketsCount, packet->header.id, packet->privateFields.idx);

    ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));
    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_WaitPacketComplete(Packet* packet) {
    XLINK_RET_IF(packet == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));
    if (packet->privateFields.status == PACKET_PROCESSING) {
        packet->privateFields.status = PACKET_PENDING;
    }
    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));

    if (sem_wait(&packet->privateFields.completedSem)) {
        mvLog(MVLOG_ERROR, "can't wait completedSem semaphore");
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_FreePending(Packet* packet, packetStatus_t status) {
    XLINK_RET_IF(packet == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));
    mvLog(MVLOG_DEBUG, "Post complete semaphore id %u, streamId %u", packet->header.id, packet->header.streamId);
    packet->privateFields.status = status;
    packet->privateFields.blockingType = PACKET_NON_BLOCKING;

    XLinkError_t rc = sem_post(&packet->privateFields.completedSem) == 0 ? X_LINK_SUCCESS : X_LINK_ERROR;
    if (rc) {
        mvLog(MVLOG_ERROR, "can't post completedSem semaphore");
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));

    return rc;
}

XLinkError_t Packet_SetData(Packet* packet, void* data, int size) {
    XLINK_RET_IF(packet == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));
    mvLog(MVLOG_DEBUG, "id=%d, idx=%d, data=%p, size=%d",
          packet->header.id, packet->privateFields.idx, data, size);

    packet->header.size = size;
    packet->data = data;
    packet->privateFields.isUserData = 1;

    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_AllocateData(Packet* packet) {
    XLINK_RET_IF(packet == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));

    ASSERT_XLINK(packet->data == NULL);

    mvLog(MVLOG_DEBUG, "id=%d, idx=%d, size=%d",
          packet->header.id, packet->privateFields.idx, packet->header.size);

    if (packet->header.size != 0) {
        void* data = XLinkPlatformAllocateData(
            ALIGN_UP(packet->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);

        XLINK_RET_IF(data == NULL);
        packet->data = data;
    } else {
        packet->data = NULL;
    }

    packet->privateFields.isUserData = 0;

    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));

    return X_LINK_SUCCESS;
}

XLinkError_t Packet_ReleaseData(Packet* packet) {
    XLINK_RET_IF(packet == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));

    if (packet->privateFields.isUserData || packet->data == NULL) {
        packet->data = NULL;
        ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));
        return X_LINK_SUCCESS;
    }

    XLinkPlatformDeallocateData(packet->data,
        ALIGN_UP(packet->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
    packet->data = NULL;

    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));

    return X_LINK_SUCCESS;
}

packetCommType_t Packet_GetCommType(Packet* packet) {
    ASSERT_XLINK(packet);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));

    int type = PACKET_RESPONSE;
    if (packet->header.type < XLINK_REQUEST_LAST) {
        type = PACKET_REQUEST;
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));

    return type;
}

packetBlockingType_t Packet_GetPacketBlockingType(Packet* packet) {
    ASSERT_XLINK(packet);

    return packet->privateFields.blockingType;
}

void Packet_SetPacketBlockingType(Packet* packet, packetBlockingType_t blockingStatus) {
    ASSERT_XLINK(packet);

    ASSERT_XLINK(!pthread_mutex_lock(&packet->privateFields.packetLock));

    packet->privateFields.blockingType = blockingStatus;

    ASSERT_XLINK(!pthread_mutex_unlock(&packet->privateFields.packetLock));
}

// ------------------------------------
// Packet API implementation. End.
// ------------------------------------

// ------------------------------------
// PacketPool API implementation. Begin.
// ------------------------------------

static int getFirstFreePacketIdx(PacketPool* packetPool);

XLinkError_t PacketPool_Create(PacketPool* packetPool, streamId_t streamId, const char* name) {
    XLINK_RET_IF(packetPool == NULL);

    memset(packetPool, 0, sizeof(PacketPool));

    mv_strcpy(packetPool->name, MAX_STREAM_NAME_LENGTH, name);
    packetPool->nexUniqueId = rand() % (streamId + 1); // TODO. Make "right" unique id generator
    for (int i = 0; i < XLINK_MAX_PACKET_PER_STREAM; ++i) {
        if (Packet_Create(&packetPool->packets[i], packetPool, i, streamId) != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Cannot create packet, destroying the packet pool");
            PacketPool_Destroy(packetPool);
            return X_LINK_ERROR;
        }
    }

    if (pthread_mutex_init(&packetPool->packetAccessLock, NULL)) {
        mvLog(MVLOG_ERROR, "Cannot initialize lock mutex, destroying the packet pool");
        PacketPool_Destroy(packetPool);
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

void PacketPool_Destroy(PacketPool* packetPool) {
    ASSERT_XLINK(packetPool);

    ASSERT_XLINK(packetPool->busyPacketsCount == 0);

    for (int i = 0; i < XLINK_MAX_PACKET_PER_STREAM; ++i) {
        Packet_Destroy(&packetPool->packets[i]);
    }

    if (pthread_mutex_destroy(&packetPool->packetAccessLock)) {
        mvLog(MVLOG_ERROR, "Cannot destroy packetAccessLock");
    }
}

Packet* PacketPool_GetPacket(PacketPool* packetPool) {
    XLINK_RET_ERR_IF(packetPool == NULL, NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packetPool->packetAccessLock));

    int freeIdx = getFirstFreePacketIdx(packetPool);
    if (freeIdx >= XLINK_MAX_PACKET_PER_STREAM) {
        mvLog(MVLOG_ERROR, "Maximum packet count in packet pool is reached");
        ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));
        return NULL;
    }

    Packet* ret_packet = &packetPool->packets[freeIdx];
    ASSERT_XLINK(ret_packet);
    ret_packet->privateFields.status = PACKET_PROCESSING;
    ret_packet->header.id = packetPool->nexUniqueId++;

    packetPool->busyPacketsCount++;

    mvLog(MVLOG_DEBUG, "%s Locked packet. Busy packets count: %u. Packet: id=%d, idx=%d",
          packetPool->name, packetPool->busyPacketsCount, ret_packet->header.id, ret_packet->privateFields.idx);

    ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));

    return ret_packet;
}

Packet* PacketPool_FindPendingPacket(PacketPool* packetPool, packetId_t id) {
    XLINK_RET_ERR_IF(packetPool == NULL, NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packetPool->packetAccessLock));

    for (int i = 0; i < XLINK_MAX_PACKET_PER_STREAM; ++i) {
        if (packetPool->packets[i].header.id == id
           && packetPool->packets[i].privateFields.status == PACKET_PENDING) {
            ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));
            return &packetPool->packets[i];
        }
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));

    mvLog(MVLOG_ERROR, "%s Cannot find pending packet. id=%d", packetPool->name, id);
    return NULL;
}

XLinkError_t PacketPool_FreePendingPackets(PacketPool* packetPool, packetStatus_t status) {
    XLINK_RET_IF(packetPool == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packetPool->packetAccessLock));

    for (int i = 0; i < XLINK_MAX_PACKET_PER_STREAM; ++i) {
        if (packetPool->packets[i].privateFields.status == PACKET_PENDING) {
            if (Packet_FreePending(&packetPool->packets[i], status) != X_LINK_SUCCESS) {
                ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));
                return X_LINK_ERROR;
            }
        }
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));

    return X_LINK_SUCCESS;
}

XLinkError_t PacketPool_SetStreamName(PacketPool* packetPool, const char* streamName) {
    XLINK_RET_IF(packetPool == NULL);
    XLINK_RET_IF(streamName == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&packetPool->packetAccessLock));

    for (int i = 0; i < XLINK_MAX_PACKET_PER_STREAM; ++i) {
        mv_strcpy(packetPool->packets[i].header.streamName, MAX_STREAM_NAME_LENGTH, streamName);
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&packetPool->packetAccessLock));

    return X_LINK_SUCCESS;
}

int getFirstFreePacketIdx(PacketPool* packetPool) {
    ASSERT_XLINK(packetPool != NULL);

    for (int i = 0; i < XLINK_MAX_PACKET_PER_STREAM; ++i) {
        if (packetPool->packets[i].privateFields.status == PACKET_FREE) {
            return i;
        }
    }

    return XLINK_MAX_PACKET_PER_STREAM;
}

// ------------------------------------
// PacketPool API implementation. End.
// ------------------------------------
