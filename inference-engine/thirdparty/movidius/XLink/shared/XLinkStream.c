// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string.h"
#include "stdlib.h"
#include "time.h"

#if (defined(_WIN32) || defined(_WIN64))
#include "gettime.h"
#endif

#include "XLink.h"
#include "XLinkTool.h"

#include "mvMacros.h"
#include "XLinkPrivateFields.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "mvLog.h"
#include "mvStringUtils.h"

// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

#ifdef __PC__
static XLinkError_t checkEventHeader(xLinkEventHeader_t header);
#endif

static float timespec_diff(struct timespec *start, struct timespec *stop);
static XLinkError_t addEvent(xLinkEvent_t *event);
static XLinkError_t addEventWithPerf(xLinkEvent_t *event, float* opTime);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------

streamId_t XLinkOpenStream(linkId_t id, const char* name, int stream_write_size)
{
    ASSERT_X_LINK(name);
    XLINK_RET_IF_RC(stream_write_size < 0,
        X_LINK_ERROR);

    xLinkDesc_t* link = getLinkById(id);
    mvLog(MVLOG_DEBUG,"%s() id %d link %p\n", __func__, id, link);
    ASSERT_X_LINK_R(link != NULL, INVALID_STREAM_ID);
    if (getXLinkState(link) != XLINK_UP) {
        /*no link*/
        mvLog(MVLOG_DEBUG,"%s() no link up\n", __func__);
        return INVALID_STREAM_ID;
    }

    if(strlen(name) > MAX_STREAM_NAME_LENGTH) {
        mvLog(MVLOG_WARN,"name too long\n");
        return INVALID_STREAM_ID;
    }

    if(stream_write_size > 0)
    {
        stream_write_size = ALIGN_UP(stream_write_size, __CACHE_LINE_SIZE);

        xLinkEvent_t event = {0};
        XLINK_INIT_EVENT(event, INVALID_STREAM_ID, XLINK_CREATE_STREAM_REQ,
                         stream_write_size, NULL, link->deviceHandle);
        mv_strncpy(event.header.streamName, MAX_STREAM_NAME_LENGTH,
                   name, MAX_STREAM_NAME_LENGTH - 1);

        dispatcherAddEvent(EVENT_LOCAL, &event);
        if (dispatcherWaitEventComplete(&link->deviceHandle))
            return INVALID_STREAM_ID;

#ifdef __PC__
        XLinkError_t eventStatus = checkEventHeader(event.header);
        if (eventStatus != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Got wrong package from device, error code = %s", XLinkErrorToStr(eventStatus));
            // FIXME: not good solution, but seems the only in the case of such XLink API
            if (eventStatus == X_LINK_OUT_OF_MEMORY) {
                return INVALID_STREAM_ID_OUT_OF_MEMORY;
            } else {
                return INVALID_STREAM_ID;
            }
        }
#endif
    }
    streamId_t streamId = getStreamIdByName(link, name);

#ifdef __PC__
    if (streamId > 0x0FFFFFFF) {
        mvLog(MVLOG_ERROR, "Cannot find stream id by the \"%s\" name", name);
        mvLog(MVLOG_ERROR,"Max streamId reached!");
        return INVALID_STREAM_ID;
    }
#else
    if (streamId == INVALID_STREAM_ID) {
        mvLog(MVLOG_ERROR,"Max streamId reached %x!", streamId);
        return INVALID_STREAM_ID;
    }
#endif

    COMBIN_IDS(streamId, id);
    return streamId;
}

// Just like open stream, when closeStream is called
// on the local size we are resetting the writeSize
// and on the remote side we are freeing the read buffer
XLinkError_t XLinkCloseStream(streamId_t streamId)
{
    xLinkDesc_t* link = getLinkByStreamId(streamId);
    ASSERT_X_LINK(link != NULL);
    XLINK_RET_IF_RC(getXLinkState(link) != XLINK_UP,
        X_LINK_COMMUNICATION_NOT_OPEN);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_CLOSE_STREAM_REQ,
        0, NULL, link->deviceHandle);

    XLINK_RET_IF(addEvent(&event));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkWriteData(streamId_t streamId, const uint8_t* buffer,
                            int size)
{
    ASSERT_X_LINK(buffer);

    float opTime = 0;
    xLinkDesc_t* link = getLinkByStreamId(streamId);
    ASSERT_X_LINK(link != NULL);
    XLINK_RET_IF_RC(getXLinkState(link) != XLINK_UP,
        X_LINK_COMMUNICATION_NOT_OPEN);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_WRITE_REQ,
        size,(void*)buffer, link->deviceHandle);

    XLINK_RET_IF(addEventWithPerf(&event, &opTime));

    if( glHandler->profEnable)
    {
        glHandler->profilingData.totalWriteBytes += size;
        glHandler->profilingData.totalWriteTime += opTime;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReadData(streamId_t streamId, streamPacketDesc_t** packet)
{
    ASSERT_X_LINK(packet);

    float opTime = 0;
    xLinkDesc_t* link = getLinkByStreamId(streamId);
    ASSERT_X_LINK(link != NULL);
    XLINK_RET_IF_RC(getXLinkState(link) != XLINK_UP,
        X_LINK_COMMUNICATION_NOT_OPEN);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_READ_REQ,
        0, NULL, link->deviceHandle);

    XLINK_RET_IF(addEventWithPerf(&event, &opTime));

    *packet = (streamPacketDesc_t *)event.data;

    if( glHandler->profEnable)
    {
        glHandler->profilingData.totalReadBytes += (*packet)->length;
        glHandler->profilingData.totalReadTime += opTime;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReleaseData(streamId_t streamId)
{
    xLinkDesc_t* link = getLinkByStreamId(streamId);
    ASSERT_X_LINK(link != NULL);
    XLINK_RET_IF_RC(getXLinkState(link) != XLINK_UP,
        X_LINK_COMMUNICATION_NOT_OPEN);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_READ_REL_REQ,
        0, NULL, link->deviceHandle);

    XLINK_RET_IF(addEvent(&event));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkGetFillLevel(streamId_t streamId, int isRemote, int* fillLevel)
{
    xLinkDesc_t* link = getLinkByStreamId(streamId);
    ASSERT_X_LINK(link != NULL);
    XLINK_RET_IF_RC(getXLinkState(link) != XLINK_UP,
        X_LINK_COMMUNICATION_NOT_OPEN);

    streamDesc_t* stream =
        getStreamById(link->deviceHandle.xLinkFD, streamId);
    ASSERT_X_LINK(stream);

    if (isRemote) {
        *fillLevel = stream->remoteFillLevel;
    }
    else {
        *fillLevel = stream->localFillLevel;
    }

    releaseStream(stream);
    return X_LINK_SUCCESS;
}

// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

XLinkError_t checkEventHeader(xLinkEventHeader_t header) {
    mvLog(MVLOG_DEBUG, "header.flags.bitField: ack:%u, nack:%u, sizeTooBig:%u, block:%u, bufferFull:%u, localServe:%u, noSuchStream:%u, terminate:%u",
          header.flags.bitField.ack,
          header.flags.bitField.nack,
          header.flags.bitField.sizeTooBig,
          header.flags.bitField.block,
          header.flags.bitField.bufferFull,
          header.flags.bitField.localServe,
          header.flags.bitField.noSuchStream,
          header.flags.bitField.terminate);


    if (header.flags.bitField.ack) {
        return X_LINK_SUCCESS;
    } else if (header.flags.bitField.nack) {
        return X_LINK_COMMUNICATION_FAIL;
    } else if (header.flags.bitField.sizeTooBig) {
        return X_LINK_OUT_OF_MEMORY;
    } else {
        return X_LINK_ERROR;
    }
}

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

XLinkError_t addEvent(xLinkEvent_t *event)
{
    ASSERT_X_LINK(event);

    xLinkEvent_t* ev = dispatcherAddEvent(EVENT_LOCAL, event);
    if (ev == NULL) {
        mvLog(MVLOG_ERROR, "Dispatcher failed on adding event");
        return X_LINK_ERROR;
    }

    if (dispatcherWaitEventComplete(&event->deviceHandle)) {
        return X_LINK_TIMEOUT;
    }

    if (event->header.flags.bitField.ack != 1) {
        return X_LINK_COMMUNICATION_FAIL;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t addEventWithPerf(xLinkEvent_t *event, float* opTime)
{
    ASSERT_X_LINK(opTime);

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    XLinkError_t rc = addEvent(event);
    if(rc != X_LINK_SUCCESS) {
        return rc;
    }

    clock_gettime(CLOCK_REALTIME, &end);
    *opTime = timespec_diff(&start, &end);

    return X_LINK_SUCCESS;
}

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------
