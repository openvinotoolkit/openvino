// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>
#include <errno.h>
#include <XLinkPrivateFields.h>

#include "XLinkErrorUtils.h"
#include "XLinkStream.h"
#include "XLinkStringUtils.h"
#include "XLinkMacros.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME XLinkStream
#endif
#include "XLinkLog.h"

// ------------------------------------
// Stream API implementation. Begin.
// ------------------------------------

XLinkError_t Stream_Create(Stream* stream, streamId_t streamId) {
    if (stream == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate PacketPool");
        return X_LINK_ERROR;
    }

    memset(stream, 0, sizeof(Stream));

    snprintf(stream->name, MAX_STREAM_NAME_LENGTH, "Stream[%u]", streamId);
    stream->streamId = streamId;
    stream->streamStatus = STREAM_CLOSED;

    if (pthread_mutex_init(&stream->streamLock, NULL)) {
        mvLog(MVLOG_ERROR, "Cannot initialize streamLock, destroying the stream");
        Stream_Destroy(stream);
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

void Stream_Destroy(Stream* stream) {
    ASSERT_XLINK(stream);

    if (pthread_mutex_destroy(&stream->streamLock)) {
        mvLog(MVLOG_ERROR, "Cannot destroy streamLock");
    }
}

XLinkError_t Stream_Open(Stream* stream, const char* name) {
    XLINK_RET_IF(stream == NULL);
    XLINK_RET_IF(name == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&stream->streamLock));

    if (stream->streamStatus != STREAM_CLOSED) {
        ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));
        return X_LINK_ALREADY_OPEN;
    }

    mvLog(MVLOG_DEBUG, "Stream opening, name %s, id %u", name, stream->streamId);

    stream->streamStatus = STREAM_OPENED;
    mv_strcpy(stream->name, MAX_STREAM_NAME_LENGTH, name);

    if (PacketPool_Create(&stream->inPacketsPool, stream->streamId, name) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot create input packet pool");
        ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));
        Stream_Close(stream);
        return X_LINK_ERROR;
    }
    if (PacketPool_SetStreamName(&stream->inPacketsPool, name) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot set name for input packet pool");
        ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));
        Stream_Close(stream);
        return X_LINK_ERROR;
    }

    if (PacketPool_Create(&stream->outPacketsPool, stream->streamId, name) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot create output packet pool");
        ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));
        Stream_Close(stream);
        return X_LINK_ERROR;
    }
    if (PacketPool_SetStreamName(&stream->outPacketsPool, name) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot set name for output packet pool");
        ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));
        Stream_Close(stream);
        return X_LINK_ERROR;
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));

    return X_LINK_SUCCESS;
}

XLinkError_t Stream_Close(Stream* stream) {
    XLINK_RET_IF(stream == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&stream->streamLock));

    if (stream->streamStatus == STREAM_CLOSED) {
        ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));
        return X_LINK_SUCCESS;
    }

    mvLog(MVLOG_DEBUG, "Stream closing, name %s, id %u", stream->name, stream->streamId);

    PacketPool_FreePendingPackets(&stream->inPacketsPool, PACKET_DROPPED);
    PacketPool_FreePendingPackets(&stream->outPacketsPool, PACKET_DROPPED);

    PacketPool_Destroy(&stream->inPacketsPool);
    PacketPool_Destroy(&stream->outPacketsPool);

    snprintf(stream->name, MAX_STREAM_NAME_LENGTH, "Stream[%u]", stream->streamId);
    stream->streamStatus = STREAM_CLOSED;

    ASSERT_XLINK(!pthread_mutex_unlock(&stream->streamLock));

    return X_LINK_SUCCESS;
}

streamId_t Stream_GetId(Stream* stream) {
    ASSERT_XLINK(stream);

    return stream->streamId;
}

Packet* Stream_GetPacket(Stream* stream, ChannelType_t channelType) {
    XLINK_RET_ERR_IF(stream == NULL, NULL);
    XLINK_RET_ERR_IF(stream->streamStatus != STREAM_OPENED, NULL);

    PacketPool* packetPool = NULL;
    switch (channelType) {
        case IN_CHANNEL: {
            packetPool = &stream->inPacketsPool;
            break;
        }
        case OUT_CHANNEL: {
            packetPool = &stream->outPacketsPool;
            break;
        }
        default: {
            mvLog(MVLOG_ERROR, "Bad channel type");
            return NULL;
        }
    }

    ASSERT_XLINK(packetPool);
    Packet* packet = PacketPool_GetPacket(packetPool);
    if (packet == NULL) {
        mvLog(MVLOG_ERROR, "Cannot get packet from packet pool");
        return NULL;
    }

    return packet;
}

Packet* Stream_FindPendingPacket(Stream* stream, Packet* packet) {
    XLINK_RET_ERR_IF(stream == NULL, NULL);
    XLINK_RET_ERR_IF(stream->streamStatus != STREAM_OPENED, NULL);

    return PacketPool_FindPendingPacket(&stream->outPacketsPool, packet->header.id);
}

XLinkError_t Stream_FreePendingPackets(Stream* stream, packetStatus_t status) {
    XLINK_RET_IF(stream == NULL);
    XLINK_RET_IF(stream->streamStatus != STREAM_OPENED);

    return PacketPool_FreePendingPackets(&stream->outPacketsPool, status);
}

// ------------------------------------
// Stream API implementation. End.
// ------------------------------------


// ------------------------------------
// StreamDispatcher helpers. Begin.
// ------------------------------------

int StreamDispatcher_GetFirstFreeStream(StreamDispatcher* streamDispatcher) {
    ASSERT_XLINK(streamDispatcher);

    for (int i = 0; i < XLINK_MAX_STREAMS; ++i) {
        if (streamDispatcher->streams[i].streamStatus == STREAM_CLOSED) {
            return i;
        }
    }

    return XLINK_MAX_STREAMS;
}

// ------------------------------------
// StreamDispatcher helpers. End.
// ------------------------------------


// ------------------------------------
// StreamDispatcher API implementation. Begin.
// ------------------------------------



XLinkError_t StreamDispatcher_Create(StreamDispatcher* streamDispatcher) {
    if (streamDispatcher == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate StreamDispatcher");
        return X_LINK_ERROR;
    }

    memset(streamDispatcher, 0, sizeof(StreamDispatcher));

    srand(time(NULL));
    for (int i = 0; i < XLINK_MAX_STREAMS; ++i) {
        if (Stream_Create(&streamDispatcher->streams[i], i) != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Cannot create stream, destroying the stream dispatcher");
            StreamDispatcher_Destroy(streamDispatcher);
            return X_LINK_ERROR;
        }
    }

    if (pthread_mutex_init(&streamDispatcher->streamDispatcherLock, NULL)) {
        mvLog(MVLOG_ERROR, "Cannot initialize lock mutex, destroying the stream dispatcher");
        StreamDispatcher_Destroy(streamDispatcher);
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

void StreamDispatcher_Destroy(StreamDispatcher* streamDispatcher) {
    if (streamDispatcher == NULL) {
        return;
    }

    for (int i = 0; i < XLINK_MAX_STREAMS; ++i) {
        Stream_Destroy(&streamDispatcher->streams[i]);
    }

    if (pthread_mutex_destroy(&streamDispatcher->streamDispatcherLock)) {
        mvLog(MVLOG_ERROR, "Cannot destroy streamDispatcherLock");
    }
}

Stream* StreamDispatcher_OpenStream(StreamDispatcher* streamDispatcher, const char* streamName) {
    XLINK_RET_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_ERR_IF(streamName == NULL, NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));

    int streamIdx = 0;
    for (; streamIdx < XLINK_MAX_STREAMS; ++streamIdx) {
        Stream* tmp_stream = &streamDispatcher->streams[streamIdx];
        if (strcmp(tmp_stream->name, streamName) == 0) {
            break;
        }
    }

    Stream* ret_stream = NULL;
    if (streamIdx < XLINK_MAX_STREAMS) {
        ret_stream = &streamDispatcher->streams[streamIdx];
    } else {
        int freeStreamIdx = StreamDispatcher_GetFirstFreeStream(streamDispatcher);
        if (freeStreamIdx >= XLINK_MAX_STREAMS) {
            mvLog(MVLOG_ERROR, "The maximum stream count has been reached");
            ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
            return NULL;
        }

        ret_stream = &streamDispatcher->streams[freeStreamIdx];
        if (Stream_Open(ret_stream, streamName) != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Cannot open stream with name %s and id %u", streamName, freeStreamIdx);
            ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
            return NULL;
        }
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
    return ret_stream;
}

Stream* StreamDispatcher_OpenStreamById(StreamDispatcher* streamDispatcher,
                                        const char* streamName, streamId_t streamId) {
    XLINK_RET_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_ERR_IF(streamName == NULL, NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));
    for (streamId_t tmpStreamId = 0; tmpStreamId < XLINK_MAX_STREAMS; ++tmpStreamId) {
        Stream* tmpStream = &streamDispatcher->streams[tmpStreamId];
        if (tmpStream->streamStatus == STREAM_OPENED) {
            if (tmpStreamId == streamId) {
                if (strcmp(streamDispatcher->streams[streamId].name, streamName) != 0) {
                    tmpStream = NULL;
                }
                ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
                return tmpStream;
            }
            if (strcmp(tmpStream->name, streamName) == 0) {
                if (tmpStreamId != streamId) {
                    tmpStream = NULL;
                }
                ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
                return tmpStream;
            }
        }
    }

    Stream* ret_stream = &streamDispatcher->streams[streamId];
    if (Stream_Open(ret_stream, streamName) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot open stream with name %s and id %u", streamName, streamId);
        ret_stream = NULL;
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
    return ret_stream;
}

XLinkError_t StreamDispatcher_CloseStream(StreamDispatcher* streamDispatcher, streamId_t streamId) {
    XLINK_RET_IF(streamDispatcher == NULL);
    XLINK_RET_IF(streamId >= XLINK_MAX_STREAMS);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));

    Stream* stream = &streamDispatcher->streams[streamId];
    XLinkError_t rc = Stream_Close(stream);
    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot close stream with id %u", streamId);
    }

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));

    return rc;
}

Stream* StreamDispatcher_GetStream(StreamDispatcher* streamDispatcher, streamId_t streamId) {
    XLINK_RET_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_ERR_IF(streamId >= XLINK_MAX_STREAMS, NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streams[streamId].streamLock));

    Stream* stream = &streamDispatcher->streams[streamId];
    ASSERT_XLINK(stream != NULL);

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streams[streamId].streamLock));

    return stream;
}

Packet* StreamDispatcher_GetPacket(StreamDispatcher* streamDispatcher,
                                   streamId_t streamId, ChannelType_t channelType) {
    XLINK_RET_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_ERR_IF(streamId >= XLINK_MAX_STREAMS, NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streams[streamId].streamLock));

    Stream* stream = &streamDispatcher->streams[streamId];
    ASSERT_XLINK(stream != NULL);

    Packet* retPacket = Stream_GetPacket(stream, channelType);

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streams[streamId].streamLock));

    return retPacket;
}

Packet* StreamDispatcher_FindPendingPacket(StreamDispatcher* streamDispatcher,
                                           streamId_t streamId, Packet* packet) {
    XLINK_RET_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_ERR_IF(streamId >= XLINK_MAX_STREAMS, NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streams[streamId].streamLock));

    Stream* stream = &streamDispatcher->streams[streamId];
    ASSERT_XLINK(stream != NULL);

    Packet* retPacket = Stream_FindPendingPacket(stream, packet);

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streams[streamId].streamLock));
    return retPacket;
}

XLinkError_t StreamDispatcher_FreePendingPackets(StreamDispatcher* streamDispatcher,
                                                 streamId_t streamId, packetStatus_t status) {
    XLINK_RET_IF(streamDispatcher == NULL);
    XLINK_RET_IF(streamId >= XLINK_MAX_STREAMS);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streams[streamId].streamLock));

    Stream* stream = &streamDispatcher->streams[streamId];
    ASSERT_XLINK(stream != NULL);
    XLinkError_t rc = Stream_FreePendingPackets(stream, status);

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streams[streamId].streamLock));

    return rc;
}

XLinkError_t StreamDispatcher_Lock(StreamDispatcher* streamDispatcher) {
    XLINK_RET_IF(streamDispatcher == NULL);

    ASSERT_XLINK(!pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));

    return X_LINK_SUCCESS;
}

XLinkError_t StreamDispatcher_Unlock(StreamDispatcher* streamDispatcher) {
    XLINK_RET_IF(streamDispatcher == NULL);

    ASSERT_XLINK(!pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));

    return X_LINK_SUCCESS;
}

XLinkError_t StreamDispatcher_GetOpenedStreamIds(StreamDispatcher* streamDispatcher,
                                                 int openedStreamIds[XLINK_MAX_STREAMS], int* count) {
    XLINK_RET_IF(streamDispatcher == NULL);

    int out_idx = 0;
    for (int i = 0; i < XLINK_MAX_STREAMS; ++i) {
        if (streamDispatcher->streams[i].streamStatus == STREAM_OPENED) {
            openedStreamIds[out_idx] = i;
            out_idx++;
        }
    }

    *count = out_idx;

    return X_LINK_SUCCESS;
}

// ------------------------------------
// StreamDispatcher API implementation. End.
// ------------------------------------
