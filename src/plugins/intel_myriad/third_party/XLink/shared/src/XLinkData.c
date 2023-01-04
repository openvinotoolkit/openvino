// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string.h"
#include "stdlib.h"
#include "time.h"

#if (defined(_WIN32) || defined(_WIN64))
#include "win_time.h"
#endif

#include "XLink.h"
#include "XLinkErrorUtils.h"

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

#ifdef __PC__
static XLinkError_t checkEventHeader(xLinkEventHeader_t header);
#endif

static float timespec_diff(struct timespec *start, struct timespec *stop);
static XLinkError_t addEvent(xLinkEvent_t *event, unsigned int timeoutMs);
static XLinkError_t addEventWithPerf(xLinkEvent_t *event, float* opTime, unsigned int timeoutMs);
static XLinkError_t getLinkByStreamId(streamId_t streamId, xLinkDesc_t** out_link);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------

streamId_t XLinkOpenStream(linkId_t id, const char* name, int stream_write_size)
{
    XLINK_RET_ERR_IF(name == NULL, INVALID_STREAM_ID);
    XLINK_RET_ERR_IF(stream_write_size < 0, INVALID_STREAM_ID);

    xLinkDesc_t* link = getLinkById(id);
    mvLog(MVLOG_DEBUG,"%s() id %d link %p\n", __func__, id, link);
    XLINK_RET_ERR_IF(link == NULL, INVALID_STREAM_ID);
    XLINK_RET_ERR_IF(getXLinkState(link) != XLINK_UP, INVALID_STREAM_ID);
    XLINK_RET_ERR_IF(strlen(name) >= MAX_STREAM_NAME_LENGTH, INVALID_STREAM_ID);

    if(stream_write_size > 0)
    {
        stream_write_size = ALIGN_UP(stream_write_size, __CACHE_LINE_SIZE);

        xLinkEvent_t event = {0};
        XLINK_INIT_EVENT(event, INVALID_STREAM_ID, XLINK_CREATE_STREAM_REQ,
                         stream_write_size, NULL, link->deviceHandle);
        mv_strncpy(event.header.streamName, MAX_STREAM_NAME_LENGTH,
                   name, MAX_STREAM_NAME_LENGTH - 1);

        DispatcherAddEvent(EVENT_LOCAL, &event);
        XLINK_RET_ERR_IF(
            DispatcherWaitEventComplete(&link->deviceHandle, XLINK_NO_RW_TIMEOUT),
            INVALID_STREAM_ID);

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

    COMBINE_IDS(streamId, id);
    return streamId;
}

// Just like open stream, when closeStream is called
// on the local size we are resetting the writeSize
// and on the remote side we are freeing the read buffer
XLinkError_t XLinkCloseStream(streamId_t streamId)
{
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_CLOSE_STREAM_REQ,
        0, NULL, link->deviceHandle);

    XLINK_RET_IF(addEvent(&event, XLINK_NO_RW_TIMEOUT));
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkWriteData(streamId_t streamId, const uint8_t* buffer,
                            int size)
{
    XLINK_RET_IF(buffer == NULL);

    float opTime = 0.0f;
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_WRITE_REQ,
        size,(void*)buffer, link->deviceHandle);

    XLINK_RET_IF(addEventWithPerf(&event, &opTime, XLINK_NO_RW_TIMEOUT));

    if( glHandler->profEnable) {
        glHandler->profilingData.totalWriteBytes += size;
        glHandler->profilingData.totalWriteTime += opTime;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReadData(streamId_t streamId, streamPacketDesc_t** packet)
{
    XLINK_RET_IF(packet == NULL);

    float opTime = 0.0f;
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_READ_REQ,
        0, NULL, link->deviceHandle);

    XLINK_RET_IF(addEventWithPerf(&event, &opTime, XLINK_NO_RW_TIMEOUT));

    *packet = (streamPacketDesc_t *)event.data;
    if(*packet == NULL) {
        return X_LINK_ERROR;
    }

    if( glHandler->profEnable) {
        glHandler->profilingData.totalReadBytes += (*packet)->length;
        glHandler->profilingData.totalReadTime += opTime;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkWriteDataWithTimeout(streamId_t streamId, const uint8_t* buffer,
                                       int size, unsigned int timeoutMs)
{
    XLINK_RET_IF(buffer == NULL);

    float opTime = 0.0f;
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_WRITE_REQ,
        size,(void*)buffer, link->deviceHandle);

    XLINK_RET_IF_FAIL(addEventWithPerf(&event, &opTime, timeoutMs));

    if( glHandler->profEnable) {
        glHandler->profilingData.totalWriteBytes += size;
        glHandler->profilingData.totalWriteTime += opTime;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReadDataWithTimeout(streamId_t streamId, streamPacketDesc_t** packet, unsigned int timeoutMs)
{
    XLINK_RET_IF(packet == NULL);

    float opTime = 0.0f;
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_READ_REQ,
        0, NULL, link->deviceHandle);

    XLINK_RET_IF_FAIL(addEventWithPerf(&event, &opTime, timeoutMs));

    *packet = (streamPacketDesc_t *)event.data;
    if(*packet == NULL) {
        return X_LINK_ERROR;
    }

    if( glHandler->profEnable) {
        glHandler->profilingData.totalReadBytes += (*packet)->length;
        glHandler->profilingData.totalReadTime += opTime;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReleaseData(streamId_t streamId)
{
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_READ_REL_REQ,
        0, NULL, link->deviceHandle);

    XLINK_RET_IF(addEvent(&event, XLINK_NO_RW_TIMEOUT));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReleaseSpecificData(streamId_t streamId, streamPacketDesc_t* packetDesc)
{
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    xLinkEvent_t event = {0};
    XLINK_INIT_EVENT(event, streamId, XLINK_READ_REL_SPEC_REQ,
        0, (void*)packetDesc->data, link->deviceHandle);

    XLINK_RET_IF(addEvent(&event, XLINK_NO_RW_TIMEOUT));

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkGetFillLevel(streamId_t streamId, int isRemote, int* fillLevel)
{
    xLinkDesc_t* link = NULL;
    XLINK_RET_IF(getLinkByStreamId(streamId, &link));
    streamId = EXTRACT_STREAM_ID(streamId);

    streamDesc_t* stream =
        getStreamById(link->deviceHandle.xLinkFD, streamId);
    ASSERT_XLINK(stream);

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

    return start->tv_nsec/ 1000000000.0f + start->tv_sec;
}

XLinkError_t addEvent(xLinkEvent_t *event, unsigned int timeoutMs)
{
    ASSERT_XLINK(event);

    xLinkEvent_t* ev = DispatcherAddEvent(EVENT_LOCAL, event);
    if(ev == NULL) {
        mvLog(MVLOG_ERROR, "Dispatcher failed on adding event. type: %s, id: %d, stream name: %s\n",
            TypeToStr(event->header.type), event->header.id, event->header.streamName);
        return X_LINK_ERROR;
    }
    event->header.canBeServed = 1;

    if (timeoutMs != XLINK_NO_RW_TIMEOUT) {
        xLinkDesc_t* link = NULL;
        XLINK_RET_IF(getLinkByStreamId(event->header.streamId, &link));

        if (DispatcherWaitEventComplete(&event->deviceHandle, timeoutMs))  // timeout reached
        {
            streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD,
                                                 event->header.streamId);
            ASSERT_XLINK(stream);
            event->header.dropped = 1;
            if (event->header.type == XLINK_READ_REQ)
            {
                event->header.canBeServed = 0;
                XLINK_RET_IF(DispatcherServeOrDropEvent(event->header.id, XLINK_READ_REQ, stream->id, event->deviceHandle.xLinkFD));
            }
            else if (event->header.type == XLINK_WRITE_REQ)
            {
                event->header.canBeServed = 0;
                XLINK_RET_IF(DispatcherServeOrDropEvent(event->header.id, XLINK_WRITE_REQ, stream->id, event->deviceHandle.xLinkFD));
            }
            releaseStream(stream);
            if (event->header.type == XLINK_WRITE_REQ && event->header.dropped)
            {
                mvLog(MVLOG_ERROR,"event is dropped\n");
                xLinkEvent_t dropEvent = {0};
                dropEvent.header.streamId = EXTRACT_STREAM_ID(event->header.streamId);
                XLINK_INIT_EVENT(dropEvent, event->header.streamId, XLINK_DROP_REQ,
                                 0, NULL, link->deviceHandle);
                DispatcherAddEvent(EVENT_LOCAL, &dropEvent);
                XLINK_RET_ERR_IF(DispatcherWaitEventComplete(&link->deviceHandle, XLINK_NO_RW_TIMEOUT), dropEvent.header.streamId);
            }

            return X_LINK_TIMEOUT;
        }
    }
    else  // No timeout
    {
        if (DispatcherWaitEventComplete(&event->deviceHandle, timeoutMs))
        {
            return X_LINK_TIMEOUT;
        }
    }

    XLINK_RET_ERR_IF(
        event->header.flags.bitField.ack != 1,
        X_LINK_COMMUNICATION_FAIL);

    return X_LINK_SUCCESS;
}

XLinkError_t addEventWithPerf(xLinkEvent_t *event, float* opTime, unsigned int timeoutMs)
{
    ASSERT_XLINK(opTime);

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    XLINK_RET_IF_FAIL(addEvent(event, timeoutMs));

    clock_gettime(CLOCK_REALTIME, &end);
    *opTime = timespec_diff(&start, &end);

    return X_LINK_SUCCESS;
}

static XLinkError_t getLinkByStreamId(streamId_t streamId, xLinkDesc_t** out_link) {
    ASSERT_XLINK(out_link != NULL);

    linkId_t id = EXTRACT_LINK_ID(streamId);
    *out_link = getLinkById(id);

    XLINK_RET_ERR_IF(*out_link == NULL, X_LINK_ERROR);
    XLINK_RET_ERR_IF(getXLinkState(*out_link) != XLINK_UP,
                    X_LINK_COMMUNICATION_NOT_OPEN);

    return X_LINK_SUCCESS;
}
// ------------------------------------
// Helpers declaration. End.
// ------------------------------------
