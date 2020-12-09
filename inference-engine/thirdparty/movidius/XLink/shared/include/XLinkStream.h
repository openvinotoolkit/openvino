// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINKSTREAM_H
#define OPENVINO_XLINKSTREAM_H

#include "XLinkPublicDefines.h"
#include "XLinkPacket.h"

// ------------------------------------
// Stream API. Begin.
// ------------------------------------

typedef enum{
    IN_CHANNEL = 0,
    OUT_CHANNEL,
} ChannelType_t;

typedef struct Stream_t Stream;

Stream* Stream_Create(streamId_t streamId);
void Stream_Destroy(Stream* stream);

XLinkError_t Stream_SetName(Stream* stream, const char* name);
streamId_t Stream_GetId(Stream* stream);

PacketNew* Stream_GetPacket(Stream* stream, ChannelType_t channelType);
PacketNew* Stream_FindPendingPacket(Stream* stream, packetId_t packetId);
XLinkError_t Stream_FreePendingPackets(Stream* stream, packetStatus_t status);

// ------------------------------------
// Stream API. End.
// ------------------------------------

// ------------------------------------
// StreamDispatcher API. Begin.
// ------------------------------------

typedef struct StreamDispatcher_t StreamDispatcher;

StreamDispatcher* StreamDispatcher_Create();
void StreamDispatcher_Destroy(StreamDispatcher* streamDispatcher);

Stream* StreamDispatcher_OpenStream(StreamDispatcher* streamDispatcher, const char* streamName);
Stream* StreamDispatcher_OpenStreamById(StreamDispatcher* streamDispatcher, const char* streamName, streamId_t streamId);
XLinkError_t StreamDispatcher_CloseStream(StreamDispatcher* streamDispatcher, streamId_t streamId);

Stream* StreamDispatcher_GetStream(StreamDispatcher* streamDispatcher, streamId_t streamId);
Stream* StreamDispatcher_GetStreamByName(StreamDispatcher* streamDispatcher, const char* streamName);

PacketNew* StreamDispatcher_GetPacket(StreamDispatcher* streamDispatcher,
    streamId_t streamId, ChannelType_t channelType);
PacketNew* StreamDispatcher_FindPendingPacket(StreamDispatcher* streamDispatcher,
    streamId_t streamId, packetId_t packetId);
XLinkError_t StreamDispatcher_FreePendingPackets(StreamDispatcher* streamDispatcher,
    streamId_t streamId, packetStatus_t status);

XLinkError_t StreamDispatcher_Lock(StreamDispatcher* streamDispatcher);
XLinkError_t StreamDispatcher_Unlock(StreamDispatcher* streamDispatcher);

XLinkError_t StreamDispatcher_GetOpenedStreamIds(StreamDispatcher* streamDispatcher,
    int openedStreamIds[MAX_STREAMS_NEW], int* count);

// ------------------------------------
// StreamDispatcher API. End.
// ------------------------------------

//Metrics

#endif //OPENVINO_XLINKSTREAM_H
