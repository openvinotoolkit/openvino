// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string.h"
#include "stdlib.h"
#include "time.h"

#if (defined(_WIN32) || defined(_WIN64))
#include "win_time.h"
#endif

#include "XLink.h"
#include "XLinkTool.h"

#include "XLinkMacros.h"
#include "XLinkPrivateFields.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif

#include "XLinkLog.h"
#include "XLinkStringUtils.h"

// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

static float timespec_diff(struct timespec *start, struct timespec *stop);
static XLinkError_t getLinkByStreamId(streamId_t streamId, Connection** out_connection);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------

streamId_t XLinkOpenStream(linkId_t id, const char* name, int stream_write_size)
{
    XLINK_RET_IF(name == NULL);
    XLINK_RET_IF(stream_write_size <= 0);

    mvLog(MVLOG_DEBUG, "linkId_t=%u, name=%s, stream_write_size=%d", id, name, stream_write_size);
    Connection* connection = getLinkById(id);
    XLINK_RET_WITH_ERR_IF(connection == NULL, INVALID_STREAM_ID);

    stream_write_size = ALIGN_UP(stream_write_size, __CACHE_LINE_SIZE);
    streamId_t streamId = Connection_OpenStream(connection, name, stream_write_size);
    XLINK_RET_WITH_ERR_IF(streamId == INVALID_STREAM_ID, INVALID_STREAM_ID);

    if (streamId > 0x0FFFFFFF) {
        mvLog(MVLOG_ERROR, "Cannot find stream id by the \"%s\" name", name);
        mvLog(MVLOG_ERROR,"Max streamId reached!");
        return INVALID_STREAM_ID;
    }

    COMBIN_IDS(streamId, id);
    return streamId;
}

// Just like open stream, when closeStream is called
// on the local size we are resetting the writeSize
// and on the remote side we are freeing the read buffer
XLinkError_t XLinkCloseStream(streamId_t streamId)
{
    Connection* connection = NULL;
    mvLog(MVLOG_DEBUG, "streamId=%u", streamId);
    XLINK_RET_IF(getLinkByStreamId(streamId, &connection));
    XLINK_RET_WITH_ERR_IF(connection == NULL, INVALID_STREAM_ID);
    streamId = EXTRACT_STREAM_ID(streamId);

    XLINK_RET_IF(Connection_CloseStream(connection, streamId));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkWriteData(streamId_t streamId, const uint8_t* buffer,
                            int size)
{
    XLINK_RET_IF(buffer == NULL);
    XLINK_RET_IF(size < 0);

    mvLog(MVLOG_DEBUG, "streamId=%u, buffer=%p, size=%d", streamId, buffer, size);
    Connection* connection = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &connection));
    XLINK_RET_WITH_ERR_IF(connection == NULL, INVALID_STREAM_ID);
    streamId = EXTRACT_STREAM_ID(streamId);

    XLINK_RET_IF(Connection_Write(connection, streamId, buffer, size));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReadData(streamId_t streamId, streamPacketDesc_t** packet)
{
    XLINK_RET_IF(packet == NULL);

    mvLog(MVLOG_DEBUG, "streamId=%d, packet=%s, stream_write_size=%p", streamId, packet);
    Connection* connection = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &connection));
    XLINK_RET_WITH_ERR_IF(connection == NULL, INVALID_STREAM_ID);
    streamId = EXTRACT_STREAM_ID(streamId);

    XLINK_RET_IF(Connection_Read(connection, streamId, packet));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReleaseData(streamId_t streamId)
{
    mvLog(MVLOG_DEBUG, "streamId=%d", streamId);
    Connection* connection = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &connection));
    XLINK_RET_WITH_ERR_IF(connection == NULL, INVALID_STREAM_ID);
    streamId = EXTRACT_STREAM_ID(streamId);

    XLINK_RET_IF(Connection_ReleaseData(connection, streamId));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkGetFillLevel(streamId_t streamId, int isRemote, int* fillLevel)
{
    Connection* connection = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &connection));
    XLINK_RET_WITH_ERR_IF(connection == NULL, INVALID_STREAM_ID);
    streamId = EXTRACT_STREAM_ID(streamId);

    XLINK_RET_IF(Connection_GetFillLevel(connection, streamId, isRemote, fillLevel));

    return X_LINK_SUCCESS;
}

// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

float timespec_diff(struct timespec *start, struct timespec *stop)
{
    if ((stop->tv_nsec - start->tv_nsec) < 0) {
        start->tv_sec = stop->tv_sec - start->tv_sec - 1;
        start->tv_nsec = stop->tv_nsec - start->tv_nsec + 1000000000;
    } else {
        start->tv_sec = stop->tv_sec - start->tv_sec;
        start->tv_nsec = stop->tv_nsec - start->tv_nsec;
    }

    return start->tv_nsec/ 1000000000.0 + start->tv_sec;
}

XLinkError_t getLinkByStreamId(streamId_t streamId, Connection** out_connection) {
    ASSERT_XLINK(out_connection);

    linkId_t id = EXTRACT_LINK_ID(streamId);
    *out_connection = getLinkById(id);

    ASSERT_XLINK(*out_connection != NULL);

    ConnectionStatus_t connectionStatus = Connection_GetStatus(*out_connection);
    XLINK_RET_IF (connectionStatus != CONNECTION_UP);

    return X_LINK_SUCCESS;
}
// ------------------------------------
// Helpers declaration. End.
// ------------------------------------
