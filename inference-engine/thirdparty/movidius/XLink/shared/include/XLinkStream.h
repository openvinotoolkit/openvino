// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINK_STREAM_H
#define OPENVINO_XLINK_STREAM_H

#include "XLinkPublicDefines.h"
#include "XLinkPacket.h"

// ------------------------------------
// Stream API. Begin.
// ------------------------------------

typedef enum {
    IN_CHANNEL,
    OUT_CHANNEL,
} ChannelType_t;

typedef enum {
    STREAM_CLOSED,
    STREAM_OPENED,
} streamStatus_t;

typedef struct Stream_t {
    streamId_t streamId;
    streamStatus_t streamStatus;
    pthread_mutex_t streamLock;
    char name[MAX_STREAM_NAME_LENGTH];

    PacketPool inPacketsPool;
    PacketPool outPacketsPool;
} Stream;

XLinkError_t Stream_Create(
        Stream* stream,
        streamId_t streamId);
void Stream_Destroy(
        Stream* stream);

XLinkError_t Stream_Open(
        Stream* stream,
        const char* name);
XLinkError_t Stream_Close(
        Stream* stream);

streamId_t Stream_GetId(
        Stream* stream);

Packet* Stream_GetPacket(
        Stream* stream,
        ChannelType_t channelType);
Packet* Stream_FindPendingPacket(
        Stream* stream,
        Packet* packet);
XLinkError_t Stream_FreePendingPackets(
        Stream* stream,
        packetStatus_t status);

// ------------------------------------
// Stream API. End.
// ------------------------------------

// ------------------------------------
// StreamDispatcher API. Begin.
// ------------------------------------

typedef struct StreamDispatcher_t {
    Stream streams[XLINK_MAX_STREAMS];
    pthread_mutex_t streamDispatcherLock;
} StreamDispatcher;

XLinkError_t StreamDispatcher_Create(
        StreamDispatcher* streamDispatcher);

void StreamDispatcher_Destroy(
        StreamDispatcher* streamDispatcher);

Stream* StreamDispatcher_OpenStream(
        StreamDispatcher* streamDispatcher,
        const char* streamName);

Stream* StreamDispatcher_OpenStreamById(
        StreamDispatcher* streamDispatcher,
        const char* streamName,
        streamId_t streamId);

XLinkError_t StreamDispatcher_CloseStream(
        StreamDispatcher* streamDispatcher,
        streamId_t streamId);

Stream* StreamDispatcher_GetStream(
        StreamDispatcher* streamDispatcher,
        streamId_t streamId);

Packet* StreamDispatcher_GetPacket(
        StreamDispatcher* streamDispatcher,
        streamId_t streamId,
        ChannelType_t channelType);

Packet* StreamDispatcher_FindPendingPacket(
        StreamDispatcher* streamDispatcher,
        streamId_t streamId,
        Packet* packet);

XLinkError_t StreamDispatcher_FreePendingPackets(
        StreamDispatcher* streamDispatcher,
        streamId_t streamId,
        packetStatus_t status);

XLinkError_t StreamDispatcher_Lock(
        StreamDispatcher* streamDispatcher);
XLinkError_t StreamDispatcher_Unlock(
        StreamDispatcher* streamDispatcher);

XLinkError_t StreamDispatcher_GetOpenedStreamIds(
        StreamDispatcher* streamDispatcher,
        int openedStreamIds[XLINK_MAX_STREAMS],
        int* count);

// ------------------------------------
// StreamDispatcher API. End.
// ------------------------------------

//Metrics

#endif  // OPENVINO_XLINK_STREAM_H
