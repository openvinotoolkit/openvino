// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINK_PACKET_H
#define OPENVINO_XLINK_PACKET_H

#include "XLinkSemaphore.h"
#include "XLinkPublicDefines.h"
#include "XLinkPrivateDefines.h"

typedef int32_t packetId_t;
typedef int32_t packetIdx_t;

typedef struct PacketPool_t PacketPool;

typedef enum {
    PACKET_FREE,
    PACKET_PROCESSING,
    PACKET_PENDING,
    PACKET_COMPLETED,
    PACKET_DROPPED
} packetStatus_t;

typedef enum {
    PACKET_REQUEST,
    PACKET_RESPONSE,
} packetCommType_t;

typedef enum {
    PACKET_NON_BLOCKING,
    PACKET_BLOCKING,
} packetBlockingType_t;

typedef struct PacketHeader_t {
    packetId_t       id;
    xLinkEventType_t type;
    char             streamName[MAX_STREAM_NAME_LENGTH];
    streamId_t       streamId;
    uint32_t         size;
    int32_t          serviceInfo;
} PacketHeader;

typedef struct PacketPrivate_t{
    packetIdx_t          idx;
    packetStatus_t       status;
    sem_t                completedSem;
    pthread_mutex_t      packetLock;
    packetBlockingType_t blockingType;
    int                  isUserData;
    PacketPool*          packetPool;
} PacketPrivate;

// ------------------------------------
// Packet API. Begin.
// ------------------------------------

typedef struct Packet_t {
    PacketPrivate privateFields;
    XLINK_ALIGN_TO_BOUNDARY(64) PacketHeader header;
    streamPacketDesc_t userData; // For support old API
    void* data;
} Packet;

XLinkError_t Packet_Create(
        Packet* packet,
        PacketPool* packetPool,
        packetIdx_t idx,
        streamId_t streamId);
void Packet_Destroy(
        Packet* packet);

XLinkError_t Packet_Release(
        Packet* packet);

XLinkError_t Packet_WaitPacketComplete(
        Packet* packet);
XLinkError_t Packet_FreePending(
        Packet* packet,
        packetStatus_t status);

XLinkError_t Packet_SetData(
        Packet* packet,
        void* data,
        int size);
XLinkError_t Packet_AllocateData(
        Packet* packet);
XLinkError_t Packet_ReleaseData(
        Packet* packet);

packetCommType_t Packet_GetCommType(
        Packet* packet);

packetBlockingType_t Packet_GetPacketBlockingType(
        Packet* packet);
void Packet_SetPacketBlockingType(
        Packet* packet,
        packetBlockingType_t blockingStatus);

// ------------------------------------
// Packet API. End.
// ------------------------------------

// ------------------------------------
// Packet pool API. Begin.
// ------------------------------------

struct PacketPool_t {
    char name[MAX_STREAM_NAME_LENGTH];

    XLINK_ALIGN_TO_BOUNDARY(64) Packet packets[XLINK_MAX_PACKET_PER_STREAM];

    pthread_mutex_t packetAccessLock;
    pthread_cond_t packetAccessCond;
    packetId_t nexUniqueId;

    // Metrics
    size_t busyPacketsCount;
};

XLinkError_t PacketPool_Create(
        PacketPool* packetPool,
        streamId_t streamId,
        const char* name);
void PacketPool_Destroy(
        PacketPool* packetPool);

Packet* PacketPool_GetPacket(
        PacketPool* packetPool);

Packet* PacketPool_FindPendingPacket(
        PacketPool* packetPool,
        packetId_t id);
XLinkError_t PacketPool_FreePendingPackets(
        PacketPool* packetPool,
        packetStatus_t status);
XLinkError_t PacketPool_SetStreamName(
        PacketPool* packetPool,
        const char* streamName);

// ------------------------------------
// Packet pool API. End.
// ------------------------------------

#endif  // OPENVINO_XLINK_PACKET_H
