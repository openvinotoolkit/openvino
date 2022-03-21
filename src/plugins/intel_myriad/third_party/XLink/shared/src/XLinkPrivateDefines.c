// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>

#include "XLinkErrorUtils.h"
#include "XLinkPrivateFields.h"
#include "XLinkPrivateDefines.h"
#include "XLinkPlatform.h"
#include "XLinkMacros.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif

#include "XLinkLog.h"

// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

XLinkError_t getNextAvailableStreamIndex(xLinkDesc_t* link, int* out_id);
static streamId_t getNextStreamUniqueId(xLinkDesc_t *link);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------



// ------------------------------------
// XLinkPrivateDefines API implementation. Begin.
// ------------------------------------

streamId_t XLinkAddOrUpdateStream(void *fd, const char *name,
    uint32_t writeSize, uint32_t readSize, streamId_t forcedId)
{
    mvLog(MVLOG_DEBUG, "name: %s, writeSize: %u, readSize: %u, forcedId: %u\n",
        name, writeSize, readSize, forcedId);

    streamId_t retStreamId = INVALID_STREAM_ID;
    streamDesc_t* stream = NULL;
    xLinkDesc_t* link = getLink(fd);
    XLINK_OUT_IF(link == NULL);

    stream = getStreamByName(link, name);
    if (stream != NULL) {
        int streamAlreadyExists = (writeSize > stream->writeSize && stream->writeSize != 0)
            || (readSize > stream->readSize && stream->readSize != 0);
        XLINK_OUT_WITH_LOG_IF(streamAlreadyExists,
            mvLog(MVLOG_ERROR, "Stream with name:%s already exists: id=%u\n", name, stream->id));
    } else {
        streamId_t nextStreamId = forcedId == INVALID_STREAM_ID ?
                                  getNextStreamUniqueId(link) : forcedId;
        int idx = 0;
        XLINK_OUT_IF(getNextAvailableStreamIndex(link, &idx));
        stream = &link->availableStreams[idx];

        XLINK_OUT_IF(XLinkStreamInitialize(stream, nextStreamId, name));
    }

    if (readSize && !stream->readSize) {
        stream->readSize = readSize;

#ifndef __PC__
        // FIXME: not the best solution but the simplest for now:
        // it is just for a check; real allocation will be done during receiving an usb package
        void *buffer = XLinkPlatformAllocateData(ALIGN_UP(readSize, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
        if (buffer == NULL) {
            mvLog(MVLOG_ERROR,"Cannot create stream. Requested memory = %u", stream->readSize);
            return INVALID_STREAM_ID;
        } else {
            XLinkPlatformDeallocateData(buffer, ALIGN_UP(readSize, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
        }
#endif
    }
    if (writeSize && !stream->writeSize) {
        stream->writeSize = writeSize;
    }

    retStreamId = stream->id;
    mvLog(MVLOG_DEBUG, "The stream \"%s\"  created, id = %u, writeSize = %d, readSize = %d\n",
          stream->name, stream->id, stream->writeSize, stream->readSize);

XLINK_OUT:
    if(stream != NULL) {
        releaseStream(stream);
    }
    return retStreamId;
}

// ------------------------------------
// XLinkPrivateDefines API implementation. End.
// ------------------------------------



// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------

XLinkError_t getNextAvailableStreamIndex(xLinkDesc_t* link, int* out_id)
{
    ASSERT_XLINK(link);

    *out_id = XLINK_MAX_STREAMS;
    for (int idx = 0; idx < XLINK_MAX_STREAMS; idx++) {
        if (link->availableStreams[idx].id == INVALID_STREAM_ID) {
            *out_id = idx;
            return X_LINK_SUCCESS;
        }
    }

    mvLog(MVLOG_DEBUG,"No next available stream!\n");
    return X_LINK_ERROR;
}

streamId_t getNextStreamUniqueId(xLinkDesc_t *link)
{
    XLINK_RET_ERR_IF(link == NULL, INVALID_STREAM_ID);
    uint32_t start = link->nextUniqueStreamId;
    uint32_t curr = link->nextUniqueStreamId;
    do
    {
        int i;
        for (i = 0; i < XLINK_MAX_STREAMS; i++)
        {
            if (link->availableStreams[i].id != INVALID_STREAM_ID &&
                link->availableStreams[i].id == curr)
                break;
        }
        if (i >= XLINK_MAX_STREAMS)
        {
            link->nextUniqueStreamId = curr;
            return curr;
        }
        curr++;

        if (curr == INVALID_STREAM_ID)
        {
            curr = 0;
        }
    } while (start != curr);
    mvLog(MVLOG_ERROR, "%s():- no next available stream unique id!\n", __func__);
    return INVALID_STREAM_ID;
}

// ------------------------------------
// Helpers implementation. End.
// ------------------------------------
