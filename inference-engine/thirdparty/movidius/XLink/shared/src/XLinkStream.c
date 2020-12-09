// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <stdlib.h>

#include "XLinkStream.h"
#include "XLinkTool.h"
#include "XLinkMacros.h"
#include "XLinkStringUtils.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME XLinkStream
#endif
#include "XLinkLog.h"

// ------------------------------------
// Stream API implementation. Begin.
// ------------------------------------

struct Stream_t {
    streamId_t streamId;
    char name[MAX_STREAM_NAME_LENGTH];

    PacketPool* inPacketsPool;
    PacketPool* outPacketsPool;
};

Stream* Stream_Create(streamId_t streamId) {
    Stream *ret_stream = NULL;
    Stream *stream = malloc(sizeof(Stream));

    if (stream == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate PacketPool\n");
        return ret_stream;
    }

    memset(stream, 0, sizeof(Stream));

    snprintf(stream->name, sizeof(stream->name), "Stream[%u]", streamId);
    stream->streamId = streamId;

    char poolName[MAX_STREAM_NAME_LENGTH];
    snprintf(poolName, sizeof(poolName), "Stream[%u][InPacketsPool]", streamId);
    PacketPool* pool = PacketPool_Create(streamId, poolName);
    XLINK_OUT_IF(pool == NULL);

    stream->inPacketsPool = pool;

    snprintf(poolName, sizeof(poolName), "Stream[%u][OutPacketsPool]", streamId);
    pool = PacketPool_Create(streamId, poolName);
    XLINK_OUT_IF(pool == NULL);

    stream->outPacketsPool = pool;

    ret_stream = stream;

    XLINK_OUT:
    if(ret_stream == NULL
       && stream != NULL) {
        Stream_Destroy(stream);
    }
    return ret_stream;
}

void Stream_Destroy(Stream* stream) {
    if(stream == NULL) {
        return;
    }

    PacketPool_Destroy(stream->inPacketsPool);
    PacketPool_Destroy(stream->outPacketsPool);

    return;
}

XLinkError_t Stream_SetName(Stream* stream, const char* name) {
    XLINK_RET_IF(stream == NULL);
    XLINK_RET_IF(name == NULL);

    //mvLog(MVLOG_DEBUG, "For stream=%s new name: %s", stream->name, name);
    mv_strcpy(stream->name, MAX_STREAM_NAME_LENGTH, name);

    XLINK_RET_IF(PacketPool_SetStreamName(stream->inPacketsPool, name));
    XLINK_RET_IF(PacketPool_SetStreamName(stream->outPacketsPool, name));

    return X_LINK_SUCCESS;
}

streamId_t Stream_GetId(Stream* stream) {
    ASSERT_XLINK(stream);

    return stream->streamId;
}

PacketNew* Stream_GetPacket(Stream* stream, ChannelType_t channelType) {
    PacketNew* ret_packet = NULL;
    XLINK_OUT_IF(stream == NULL);

    PacketPool* packetPool = NULL;
    switch (channelType) {
        case IN_CHANNEL: {
            packetPool = stream->inPacketsPool;
            break;
        }
        case OUT_CHANNEL: {
            packetPool = stream->outPacketsPool;
            break;
        }
        default: {
            XLINK_OUT_IF("Bad channel type");
        }
    }

    ASSERT_XLINK(packetPool);
    PacketNew* packet = PacketPool_GetPacket(packetPool);
    XLINK_OUT_IF(packet == NULL);

    ret_packet = packet;
    XLINK_OUT:
    return ret_packet;
}

PacketNew* Stream_FindPendingPacket(Stream* stream, packetId_t packetId) {
    XLINK_RET_WITH_ERR_IF(stream == NULL, NULL);

    return PacketPool_FindPacket(stream->outPacketsPool, packetId);
}

XLinkError_t Stream_FreePendingPackets(Stream* stream, packetStatus_t status) {
    XLINK_RET_IF(stream == NULL);

    return PacketPool_FreePendingPackets(stream->outPacketsPool, status);
}

// ------------------------------------
// Stream API implementation. End.
// ------------------------------------



// ------------------------------------
// StreamDispatcher API implementation. Begin.
// ------------------------------------

#define FREE_STREAM_FLAG 0
#define BUSY_STREAM_FLAG 1

struct StreamDispatcher_t {
    Stream* streams[MAX_STREAMS_NEW];
    int freeStreamsIdx[MAX_PACKET_PER_STREAM];

    pthread_mutex_t streamDispatcherLock;
    pthread_mutex_t streamLock[MAX_STREAMS_NEW];
};

static int getFirstFreeStream(StreamDispatcher* streamDispatcher);

StreamDispatcher* StreamDispatcher_Create() {
    StreamDispatcher *ret_streamDispatcher = NULL;
    StreamDispatcher *streamDispatcher = malloc(sizeof(StreamDispatcher));

    if (streamDispatcher == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate PoolContainer_t\n");
        return ret_streamDispatcher;
    }

    memset(streamDispatcher, 0, sizeof(StreamDispatcher));

    srand(time(NULL));
    for (int i = 0; i < MAX_STREAMS_NEW; ++i) {
        Stream* stream = Stream_Create(i);
        XLINK_OUT_IF(stream == NULL);

        streamDispatcher->streams[i] = stream;
        streamDispatcher->freeStreamsIdx[i] = FREE_STREAM_FLAG;

        XLINK_OUT_WITH_LOG_IF(pthread_mutex_init(&streamDispatcher->streamLock[i], NULL),
                              mvLog(MVLOG_ERROR, "Cannot initialize lock mutex\n"));
    }

    XLINK_OUT_WITH_LOG_IF(pthread_mutex_init(&streamDispatcher->streamDispatcherLock, NULL),
                          mvLog(MVLOG_ERROR, "Cannot initialize lock mutex\n"));

    ret_streamDispatcher = streamDispatcher;

    XLINK_OUT:
    if(ret_streamDispatcher == NULL
       && streamDispatcher != NULL) {
        StreamDispatcher_Destroy(streamDispatcher);
    }
    return ret_streamDispatcher;
}

void StreamDispatcher_Destroy(StreamDispatcher* streamDispatcher) {
    if(streamDispatcher == NULL) {
        return;
    }

    for (int i = 0; i < MAX_STREAMS_NEW; ++i) {
        Stream_Destroy(streamDispatcher->streams[i]);
        ASSERT_RC_XLINK(pthread_mutex_destroy(&streamDispatcher->streamLock[i]));
    }

    ASSERT_RC_XLINK(pthread_mutex_destroy(&streamDispatcher->streamDispatcherLock));

    free(streamDispatcher);
    return;
}

Stream* StreamDispatcher_OpenStream(StreamDispatcher* streamDispatcher, const char* streamName) {
    Stream* ret_stream = NULL;
    XLINK_RET_WITH_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(streamName == NULL, NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));

    int streamIdx = 0;
    for (streamIdx = 0; streamIdx < MAX_STREAMS_NEW; ++streamIdx) {
        Stream* tmp_stream = streamDispatcher->streams[streamIdx];
        if(strcmp(tmp_stream->name, streamName) == 0) {
            break;
        }
    }

    if(streamIdx < MAX_STREAMS_NEW) {
        ret_stream = streamDispatcher->streams[streamIdx];
    } else {
        int freeStreamIdx = getFirstFreeStream(streamDispatcher);
        XLINK_OUT_IF(freeStreamIdx >= MAX_STREAMS_NEW);

        ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamLock[freeStreamIdx]));

        ret_stream = streamDispatcher->streams[freeStreamIdx];
        streamDispatcher->freeStreamsIdx[freeStreamIdx] = BUSY_STREAM_FLAG;
        Stream_SetName(ret_stream, streamName);

        ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamLock[freeStreamIdx]));
    }

    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
    return ret_stream;
}

Stream* StreamDispatcher_OpenStreamById(StreamDispatcher* streamDispatcher,
    const char* streamName, streamId_t streamId) {
    Stream* ret_stream = NULL;
    XLINK_RET_WITH_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(streamName == NULL, NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));
    XLINK_RET_WITH_ERR_IF(streamDispatcher->freeStreamsIdx[streamId]
                                       == BUSY_STREAM_FLAG, NULL);

    {
        ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamLock[streamId]));

        ret_stream = streamDispatcher->streams[streamId];
        streamDispatcher->freeStreamsIdx[streamId] = BUSY_STREAM_FLAG;
        Stream_SetName(ret_stream, streamName);

        ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamLock[streamId]));
    }

    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
    return ret_stream;
}

XLinkError_t StreamDispatcher_CloseStream(StreamDispatcher* streamDispatcher, streamId_t streamId) {
    XLINK_RET_IF(streamDispatcher == NULL);
    XLINK_RET_IF(streamId >= MAX_STREAMS_NEW);

    XLinkError_t rc = X_LINK_SUCCESS;
    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));
    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamLock[streamId]));

    Stream* stream = streamDispatcher->streams[streamId];
    Stream_Destroy(stream);

    stream = Stream_Create(streamId);
    XLINK_OUT_RC_WITH_ERR_IF(stream == NULL, X_LINK_ERROR);
    streamDispatcher->streams[streamId] = stream;
    streamDispatcher->freeStreamsIdx[streamId] = FREE_STREAM_FLAG;

    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamLock[streamId]));
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
    return rc;
}

Stream* StreamDispatcher_GetStream(StreamDispatcher* streamDispatcher, streamId_t streamId) {
    Stream* ret_stream = NULL;
    XLINK_RET_WITH_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(streamId >= MAX_STREAMS_NEW, NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamLock[streamId]));

    Stream* stream = streamDispatcher->streams[streamId];
    int streamFlag = streamDispatcher->freeStreamsIdx[streamId];
    ASSERT_XLINK(streamFlag == FREE_STREAM_FLAG
    || streamFlag == BUSY_STREAM_FLAG);
    XLINK_OUT_IF(streamFlag == FREE_STREAM_FLAG);

    ret_stream = stream;

    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamLock[streamId]));
    return ret_stream;
}

Stream* StreamDispatcher_GetStreamByName(StreamDispatcher* streamDispatcher, const char* streamName) {
    Stream* ret_stream = NULL;
    XLINK_RET_WITH_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(streamName == NULL, NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));

    int streamIdx = 0;
    for (streamIdx = 0; streamIdx < MAX_STREAMS_NEW; ++streamIdx) {
        Stream* tmp_stream = streamDispatcher->streams[streamIdx];
        if(strcmp(tmp_stream->name, streamName) == 0) {
            break;
        }
    }

    if(streamIdx >= MAX_STREAMS_NEW) {
        ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
        return ret_stream;
    }

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamLock[streamIdx]));
    XLINK_OUT_IF(streamIdx >= MAX_STREAMS_NEW);

    Stream* stream = streamDispatcher->streams[streamIdx];
    XLINK_OUT_IF(stream == NULL);
    int streamFlag = streamDispatcher->freeStreamsIdx[stream->streamId];
    ASSERT_XLINK(streamFlag == FREE_STREAM_FLAG
                 || streamFlag == BUSY_STREAM_FLAG);
    XLINK_OUT_IF(streamFlag == FREE_STREAM_FLAG);

    ret_stream = stream;
    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamLock[streamIdx]));
    return ret_stream;
}

PacketNew* StreamDispatcher_GetPacket(StreamDispatcher* streamDispatcher,
                                      streamId_t streamId, ChannelType_t channelType) {
    XLINK_RET_WITH_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(streamId >= MAX_STREAMS_NEW, NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamLock[streamId]));

    PacketNew* ret_packet = NULL;
    Stream* stream = streamDispatcher->streams[streamId];
    int streamFlag = streamDispatcher->freeStreamsIdx[streamId];
    ASSERT_XLINK(streamFlag == FREE_STREAM_FLAG
                 || streamFlag == BUSY_STREAM_FLAG);

    XLINK_OUT_IF(streamFlag == FREE_STREAM_FLAG);
    ret_packet = Stream_GetPacket(stream, channelType);

    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamLock[streamId]));
    return ret_packet;
}

PacketNew* StreamDispatcher_FindPendingPacket(StreamDispatcher* streamDispatcher,
                                              streamId_t streamId, packetId_t packetId) {
    XLINK_RET_WITH_ERR_IF(streamDispatcher == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(streamId >= MAX_STREAMS_NEW, NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamLock[streamId]));

    PacketNew* ret_packet = NULL;
    Stream* stream = streamDispatcher->streams[streamId];
    int streamFlag = streamDispatcher->freeStreamsIdx[streamId];
    ASSERT_XLINK(streamFlag == FREE_STREAM_FLAG
                 || streamFlag == BUSY_STREAM_FLAG);
    XLINK_OUT_IF(streamFlag == FREE_STREAM_FLAG);

    ret_packet = Stream_FindPendingPacket(stream, packetId);

    XLINK_OUT:
    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamLock[streamId]));
    return ret_packet;
}

XLinkError_t StreamDispatcher_FreePendingPackets(StreamDispatcher* streamDispatcher,
                                                 streamId_t streamId, packetStatus_t status) {
    XLINK_RET_IF(streamDispatcher == NULL);
    XLINK_RET_IF(streamId >= MAX_STREAMS_NEW);

    Stream* stream = streamDispatcher->streams[streamId];
    int streamFlag = streamDispatcher->freeStreamsIdx[streamId];
    ASSERT_XLINK(streamFlag == FREE_STREAM_FLAG
                 || streamFlag == BUSY_STREAM_FLAG);
    XLINK_RET_IF(streamFlag == FREE_STREAM_FLAG);

    return Stream_FreePendingPackets(stream, status);
}

XLinkError_t StreamDispatcher_Lock(StreamDispatcher* streamDispatcher) {
    XLINK_RET_IF(streamDispatcher == NULL);

    ASSERT_RC_XLINK(pthread_mutex_lock(&streamDispatcher->streamDispatcherLock));

    return X_LINK_SUCCESS;
}

XLinkError_t StreamDispatcher_Unlock(StreamDispatcher* streamDispatcher) {
    XLINK_RET_IF(streamDispatcher == NULL);

    ASSERT_RC_XLINK(pthread_mutex_unlock(&streamDispatcher->streamDispatcherLock));

    return X_LINK_SUCCESS;
}

XLinkError_t StreamDispatcher_GetOpenedStreamIds(StreamDispatcher* streamDispatcher,
                                                 int openedStreamIds[MAX_STREAMS_NEW], int* count) {
    XLINK_RET_IF(streamDispatcher == NULL);

    int out_idx = 0;
    for (int i = 0; i < MAX_STREAMS_NEW; ++i) {
        if(streamDispatcher->freeStreamsIdx[i] == BUSY_STREAM_FLAG) {
            openedStreamIds[out_idx] = i;
            out_idx++;
        }
    }

    *count = out_idx;

    return X_LINK_SUCCESS;
}

int getFirstFreeStream(StreamDispatcher* streamDispatcher) {
    ASSERT_XLINK(streamDispatcher);

    for (int i = 0; i < MAX_STREAMS_NEW; ++i) {
        if(streamDispatcher->freeStreamsIdx[i] == FREE_STREAM_FLAG) {
            return i;
        }
    }

    return MAX_STREAMS_NEW;
}

// ------------------------------------
// StreamDispatcher API implementation. End.
// ------------------------------------
