// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINKPACKET_H
#define OPENVINO_XLINKPACKET_H

#if (defined(_WIN32) || defined(_WIN64))
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif

#include "XLinkPublicDefines.h"
#include "XLinkPrivateDefines.h"

typedef struct PacketPool_t PacketPool;

typedef int32_t packetId_t;
typedef int32_t packetIdx_t;

typedef enum
{
    PACKET_FREE,
    PACKET_PROCESSING,
    PACKET_PENDING,
    PACKET_COMPLETED,
    PACKET_DROPED
} packetStatus_t;

typedef enum
{
    PACKET_REQUEST,
    PACKET_RESPONSE,
} packetCommType_t;

typedef struct PacketHeader_t{
    packetId_t          id;
    xLinkEventType_t    type;
    char                streamName[MAX_STREAM_NAME_LENGTH];
    streamId_t          streamId;
    uint32_t            size;

    // for backward compatibility
    union{
        uint32_t raw;
        struct{
            uint32_t ack : 1;
            uint32_t nack : 1;
            uint32_t block : 1;
            uint32_t localServe : 1;
            uint32_t terminate : 1;
            uint32_t bufferFull : 1;
            uint32_t sizeTooBig : 1;
            uint32_t noSuchStream : 1;
        }bitField;
    }flags;
}PacketHeader;

typedef struct PacketPrivate_t{
    packetIdx_t     idx;
    packetStatus_t  status;
    sem_t           completedSem;
    int             isUserData;
    PacketPool*     packetPool;
} PacketPrivate;

typedef struct PacketNew_t {
    PacketPrivate privateFields;
    XLINK_ALIGN_TO_BOUNDARY(64) PacketHeader header;
    streamPacketDesc_t userData; // For support old API
    void* data;
}PacketNew;

// ------------------------------------
// Packet API. Begin.
// ------------------------------------

PacketNew* Packet_Create(PacketPool* packetPool, packetIdx_t idx, streamId_t streamId);
void Packet_Destroy(PacketNew* packet);

XLinkError_t Packet_Release(PacketNew* packet);
XLinkError_t Packet_FreePending(PacketNew* packet, packetStatus_t status);

XLinkError_t Packet_SetData(PacketNew* packet, void* data, int size);
XLinkError_t Packet_AllocateData(PacketNew* packet);
XLinkError_t Packet_ReleaseData(PacketNew* packet);

packetCommType_t Packet_GetCommType(PacketNew* packet);

// ------------------------------------
// Packet API. End.
// ------------------------------------

// ------------------------------------
// Packet pool API. Begin.
// ------------------------------------

PacketPool* PacketPool_Create(streamId_t streamId, const char* name);
void PacketPool_Destroy(PacketPool* packetPool);

PacketNew* PacketPool_GetPacket(PacketPool* packetPool);
XLinkError_t PacketPool_ReleasePacket(PacketPool* packetPool, PacketNew* packet);

PacketNew* PacketPool_FindPacket(PacketPool* packetPool, packetId_t id);
XLinkError_t PacketPool_FreePendingPackets(PacketPool* packetPool, packetStatus_t status);
XLinkError_t PacketPool_SetStreamName(PacketPool* packetPool, const char* streamName);

// ------------------------------------
// Packet pool API. End.
// ------------------------------------

#endif //OPENVINO_XLINKDISPATCHERNEW_H
