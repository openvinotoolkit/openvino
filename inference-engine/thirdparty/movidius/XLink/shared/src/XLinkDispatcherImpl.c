// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>
#include "stdlib.h"

#include "XLinkMacros.h"
#include "XLinkTool.h"
#include "XLinkPlatform.h"
#include "XLinkDispatcherImpl.h"
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

static int isStreamSpaceEnoughFor(streamDesc_t* stream, uint32_t size);

static streamPacketDesc_t* getPacketFromStream(streamDesc_t* stream);
static int releasePacketFromStream(streamDesc_t* stream, uint32_t* releasedSize);
static int addNewPacketToStream(streamDesc_t* stream, void* buffer, uint32_t size);

static int handleIncomingEvent(xLinkEvent_t* event);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------



// ------------------------------------
// XLinkDispatcherImpl.h implementation. Begin.
// ------------------------------------

//adds a new event with parameters and returns event id
int dispatcherEventSend(xLinkEvent_t *event)
{
    mvLog(MVLOG_DEBUG, "Send event: %s, size %d, streamId %ld.\n",
        TypeToStr(event->header.type), event->header.size, event->header.streamId);

    int rc = XLinkPlatformWrite(&event->deviceHandle,
        &event->header, sizeof(event->header));

    if(rc < 0) {
        mvLog(MVLOG_ERROR,"Write failed (header) (err %d) | event %s\n", rc, TypeToStr(event->header.type));
        return rc;
    }

    if (event->header.type == XLINK_WRITE_REQ) {
        rc = XLinkPlatformWrite(&event->deviceHandle,
            event->data, event->header.size);
        if(rc < 0) {
            mvLog(MVLOG_ERROR,"Write failed %d\n", rc);
            return rc;
        }
    }

    return 0;
}

int dispatcherEventReceive(xLinkEvent_t* event){
    static xLinkEvent_t prevEvent = {0};
    int rc = XLinkPlatformRead(&event->deviceHandle,
        &event->header, sizeof(event->header));

    mvLog(MVLOG_DEBUG,"Incoming event %p: %s %d %p prevEvent: %s %d %p\n",
          event,
          TypeToStr(event->header.type),
          (int)event->header.id,
          event->deviceHandle.xLinkFD,
          TypeToStr(prevEvent.header.type),
          (int)prevEvent.header.id,
          prevEvent.deviceHandle.xLinkFD);

    if(rc < 0) {
        mvLog(MVLOG_DEBUG,"%s() Read failed %d\n", __func__, (int)rc);
        return rc;
    }

    if (prevEvent.header.id == event->header.id &&
        prevEvent.header.type == event->header.type &&
        prevEvent.deviceHandle.xLinkFD == event->deviceHandle.xLinkFD) {
        mvLog(MVLOG_FATAL,"Duplicate id detected. \n");
    }

    prevEvent = *event;
    return handleIncomingEvent(event);
}

//this function should be called only for remote requests
int dispatcherLocalEventGetResponse(xLinkEvent_t* event, xLinkEvent_t* response)
{
    streamDesc_t* stream;
    response->header.id = event->header.id;
    mvLog(MVLOG_DEBUG, "%s\n",TypeToStr(event->header.type));
    switch (event->header.type){
        case XLINK_WRITE_REQ:
        {

            //in case local tries to write after it issues close (writeSize is zero)
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);

            if(!stream) {
                mvLog(MVLOG_DEBUG, "stream %d has been closed!\n", event->header.streamId);
                XLINK_SET_EVENT_FAILED_AND_SERVE(event);
                break;
            }

            if (stream->writeSize == 0)
            {
                XLINK_EVENT_NOT_ACKNOWLEDGE(event);
                // return -1 to don't even send it to the remote
                releaseStream(stream);
                return -1;
            }
            XLINK_EVENT_ACKNOWLEDGE(event);
            event->header.flags.bitField.localServe = 0;

            if(!isStreamSpaceEnoughFor(stream, event->header.size)){
                mvLog(MVLOG_FATAL,"local NACK RTS. stream '%s' is full (event %d)\n", stream->name, event->header.id);
                event->header.flags.bitField.block = 1;
                event->header.flags.bitField.localServe = 1;
                // TODO: easy to implement non-blocking read here, just return nack
                mvLog(MVLOG_WARN, "Blocked event would cause dispatching thread to wait on semaphore infinitely\n");
            }else{
                event->header.flags.bitField.block = 0;
                stream->remoteFillLevel += event->header.size;
                stream->remoteFillPacketLevel++;
                mvLog(MVLOG_DEBUG,"S%d: Got local write of %ld , remote fill level %ld out of %ld %ld\n",
                      event->header.streamId, event->header.size, stream->remoteFillLevel, stream->writeSize, stream->readSize);
            }
            releaseStream(stream);
            break;
        }
        case XLINK_READ_REQ:
        {
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            if(!stream) {
                mvLog(MVLOG_DEBUG, "stream %d has been closed!\n", event->header.streamId);
                XLINK_SET_EVENT_FAILED_AND_SERVE(event);
                break;
            }
            streamPacketDesc_t* packet = getPacketFromStream(stream);
            if (packet){
                //the read can be served with this packet
                event->data = packet;
                XLINK_EVENT_ACKNOWLEDGE(event);
                event->header.flags.bitField.block = 0;
            }
            else{
                event->header.flags.bitField.block = 1;
                // TODO: easy to implement non-blocking read here, just return nack
            }
            event->header.flags.bitField.localServe = 1;
            releaseStream(stream);
            break;
        }
        case XLINK_READ_REL_REQ:
        {
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            ASSERT_X_LINK(stream);
            XLINK_EVENT_ACKNOWLEDGE(event);
            uint32_t releasedSize = 0;
            releasePacketFromStream(stream, &releasedSize);
            event->header.size = releasedSize;
            releaseStream(stream);
            break;
        }
        case XLINK_CREATE_STREAM_REQ:
        {
            XLINK_EVENT_ACKNOWLEDGE(event);
            mvLog(MVLOG_DEBUG,"XLINK_CREATE_STREAM_REQ - do nothing\n");
            break;
        }
        case XLINK_CLOSE_STREAM_REQ:
        {
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);

            ASSERT_X_LINK(stream);
            XLINK_EVENT_ACKNOWLEDGE(event);
            if (stream->remoteFillLevel != 0){
                stream->closeStreamInitiated = 1;
                event->header.flags.bitField.block = 1;
                event->header.flags.bitField.localServe = 1;
            }else{
                event->header.flags.bitField.block = 0;
                event->header.flags.bitField.localServe = 0;
            }
            releaseStream(stream);
            break;
        }
        case XLINK_RESET_REQ:
        {
            XLINK_EVENT_ACKNOWLEDGE(event);
            mvLog(MVLOG_DEBUG,"XLINK_RESET_REQ - do nothing\n");
            break;
        }
        case XLINK_PING_REQ:
        {
            XLINK_EVENT_ACKNOWLEDGE(event);
            mvLog(MVLOG_DEBUG,"XLINK_PING_REQ - do nothing\n");
            break;
        }
        case XLINK_WRITE_RESP:
        case XLINK_READ_RESP:
        case XLINK_READ_REL_RESP:
        case XLINK_CREATE_STREAM_RESP:
        case XLINK_CLOSE_STREAM_RESP:
        case XLINK_PING_RESP:
            break;
        case XLINK_RESET_RESP:
            //should not happen
            event->header.flags.bitField.localServe = 1;
            break;
        default:
            ASSERT_X_LINK(0);
    }
    return 0;
}

//this function should be called only for remote requests
int dispatcherRemoteEventGetResponse(xLinkEvent_t* event, xLinkEvent_t* response)
{
    streamDesc_t* stream;
    response->header.id = event->header.id;
    response->header.flags.raw = 0;
    mvLog(MVLOG_DEBUG, "%s\n",TypeToStr(event->header.type));

    switch (event->header.type)
    {
        case XLINK_WRITE_REQ:
            {
                //let remote write immediately as we have a local buffer for the data
                response->header.type = XLINK_WRITE_RESP;
                response->header.size = event->header.size;
                response->header.streamId = event->header.streamId;
                response->deviceHandle = event->deviceHandle;
                XLINK_EVENT_ACKNOWLEDGE(response);


                // we got some data. We should unblock a blocked read
                int xxx = DispatcherUnblockEvent(-1,
                                                 XLINK_READ_REQ,
                                                 response->header.streamId,
                                                 event->deviceHandle.xLinkFD);
                (void) xxx;
                mvLog(MVLOG_DEBUG,"unblocked from stream %d %d\n",
                      (int)response->header.streamId, (int)xxx);
            }
            break;
        case XLINK_READ_REQ:
            break;
        case XLINK_READ_REL_REQ:
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_READ_REL_RESP;
            response->deviceHandle = event->deviceHandle;
            stream = getStreamById(event->deviceHandle.xLinkFD,
                                   event->header.streamId);
            ASSERT_X_LINK(stream);
            stream->remoteFillLevel -= event->header.size;
            stream->remoteFillPacketLevel--;

            mvLog(MVLOG_DEBUG,"S%d: Got remote release of %ld, remote fill level %ld out of %ld %ld\n",
                  event->header.streamId, event->header.size, stream->remoteFillLevel, stream->writeSize, stream->readSize);
            releaseStream(stream);

            DispatcherUnblockEvent(-1, XLINK_WRITE_REQ, event->header.streamId,
                                   event->deviceHandle.xLinkFD);
            //with every released packet check if the stream is already marked for close
            if (stream->closeStreamInitiated && stream->localFillLevel == 0)
            {
                mvLog(MVLOG_DEBUG,"%s() Unblock close STREAM\n", __func__);
                int xxx = DispatcherUnblockEvent(-1,
                                                 XLINK_CLOSE_STREAM_REQ,
                                                 event->header.streamId,
                                                 event->deviceHandle.xLinkFD);
                (void) xxx;
            }
            break;
        case XLINK_CREATE_STREAM_REQ:
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_CREATE_STREAM_RESP;
            //write size from remote means read size for this peer
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               0, event->header.size,
                                                               INVALID_STREAM_ID);

            if (response->header.streamId == INVALID_STREAM_ID) {
                response->header.flags.bitField.ack = 0;
                response->header.flags.bitField.sizeTooBig = 1;
                break;
            }

            response->deviceHandle = event->deviceHandle;
            mv_strncpy(response->header.streamName, MAX_STREAM_NAME_LENGTH,
                       event->header.streamName, MAX_STREAM_NAME_LENGTH - 1);
            response->header.size = event->header.size;
            mvLog(MVLOG_DEBUG,"creating stream %x\n", (int)response->header.streamId);
            break;
        case XLINK_CLOSE_STREAM_REQ:
        {
            response->header.type = XLINK_CLOSE_STREAM_RESP;
            response->header.streamId = event->header.streamId;
            response->deviceHandle = event->deviceHandle;

            streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD,
                                                 event->header.streamId);
            if (!stream) {
                //if we have sent a NACK before, when the event gets unblocked
                //the stream might already be unavailable
                XLINK_EVENT_ACKNOWLEDGE(response);
                mvLog(MVLOG_DEBUG,"%s() got a close stream on aready closed stream\n", __func__);
            } else {
                if (stream->localFillLevel == 0)
                {
                    XLINK_EVENT_ACKNOWLEDGE(response);

                    if (stream->readSize)
                    {
                        stream->readSize = 0;
                        stream->closeStreamInitiated = 0;
                    }

                    if (!stream->writeSize) {
                        stream->id = INVALID_STREAM_ID;
                        stream->name[0] = '\0';
                    }
#ifndef __PC__
                    if(sem_destroy(&stream->sem))
                        perror("Can't destroy semaphore");
#endif
                }
                else
                {
                    mvLog(MVLOG_DEBUG,"%s():fifo is NOT empty returning NACK \n", __func__);
                    XLINK_EVENT_NOT_ACKNOWLEDGE(response);
                    stream->closeStreamInitiated = 1;
                }

                releaseStream(stream);
            }
            break;
        }
        case XLINK_PING_REQ:
            response->header.type = XLINK_PING_RESP;
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->deviceHandle = event->deviceHandle;
            sem_post(&pingSem);
            break;
        case XLINK_RESET_REQ:
            mvLog(MVLOG_DEBUG,"reset request - received! Sending ACK *****\n");
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_RESET_RESP;
            response->deviceHandle = event->deviceHandle;
            // need to send the response, serve the event and then reset
            break;
        case XLINK_WRITE_RESP:
            break;
        case XLINK_READ_RESP:
            break;
        case XLINK_READ_REL_RESP:
            break;
        case XLINK_CREATE_STREAM_RESP:
        {
            // write_size from the response the size of the buffer from the remote
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               event->header.size, 0,
                                                               event->header.streamId);
            XLINK_RET_IF_RC(response->header.streamId
                == INVALID_STREAM_ID, X_LINK_ERROR);
            response->deviceHandle = event->deviceHandle;
            break;
        }
        case XLINK_CLOSE_STREAM_RESP:
        {
            streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD,
                                                 event->header.streamId);

            if (!stream){
                XLINK_EVENT_NOT_ACKNOWLEDGE(response);
                break;
            }
            stream->writeSize = 0;
            if (!stream->readSize) {
                XLINK_EVENT_NOT_ACKNOWLEDGE(response);
                stream->id = INVALID_STREAM_ID;
                stream->name[0] = '\0';
                break;
            }
            releaseStream(stream);
            break;
        }
        case XLINK_PING_RESP:
            break;
        case XLINK_RESET_RESP:
            break;
        default:
            ASSERT_X_LINK(0);
    }
    return 0;
}

void dispatcherCloseLink(void* fd, int fullClose)
{
    xLinkDesc_t* link = getLink(fd);

    if (!link) {
        mvLog(MVLOG_WARN, "Dispatcher link is null");
        return;
    }

    if (!fullClose) {
        link->peerState = XLINK_DOWN;
        return;
    }

    link->id = INVALID_LINK_ID;
    link->deviceHandle.xLinkFD = NULL;
    link->peerState = XLINK_NOT_INIT;
    link->nextUniqueStreamId = 0;

    for (int index = 0; index < XLINK_MAX_STREAMS; index++) {
        streamDesc_t* stream = &link->availableStreams[index];
        if (!stream) {
            continue;
        }

        while (getPacketFromStream(stream) || stream->blockedPackets) {
            releasePacketFromStream(stream, NULL);
        }

        XLinkStreamReset(stream);
    }

    if(sem_destroy(&link->dispatcherClosedSem)) {
        mvLog(MVLOG_DEBUG, "Cannot destroy dispatcherClosedSem\n");
    }
}

void dispatcherCloseDeviceFd(xLinkDeviceHandle_t* deviceHandle)
{
    XLinkPlatformCloseRemote(deviceHandle);
}

// ------------------------------------
// XLinkDispatcherImpl.h implementation. End.
// ------------------------------------



// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------

int isStreamSpaceEnoughFor(streamDesc_t* stream, uint32_t size)
{
    if(stream->remoteFillPacketLevel >= XLINK_MAX_PACKETS_PER_STREAM ||
       stream->remoteFillLevel + size > stream->writeSize){
        mvLog(MVLOG_DEBUG, "S%d: Not enough space in stream '%s' for %ld: PKT %ld, FILL %ld SIZE %ld\n",
              stream->id, stream->name, size, stream->remoteFillPacketLevel, stream->remoteFillLevel, stream->writeSize);
        return 0;
    }

    return 1;
}

streamPacketDesc_t* getPacketFromStream(streamDesc_t* stream)
{
    streamPacketDesc_t* ret = NULL;
    if (stream->availablePackets)
    {
        ret = &stream->packets[stream->firstPacketUnused];
        stream->availablePackets--;
        CIRCULAR_INCREMENT(stream->firstPacketUnused,
                           XLINK_MAX_PACKETS_PER_STREAM);
        stream->blockedPackets++;
    }
    return ret;
}

int releasePacketFromStream(streamDesc_t* stream, uint32_t* releasedSize)
{
    streamPacketDesc_t* currPack = &stream->packets[stream->firstPacket];
    if(stream->blockedPackets == 0){
        mvLog(MVLOG_ERROR,"There is no packet to release\n");
        return 0; // ignore this, although this is a big problem on application side
    }

    stream->localFillLevel -= currPack->length;
    mvLog(MVLOG_DEBUG, "S%d: Got release of %ld , current local fill level is %ld out of %ld %ld\n",
          stream->id, currPack->length, stream->localFillLevel, stream->readSize, stream->writeSize);

    XLinkPlatformDeallocateData(currPack->data,
                                ALIGN_UP_INT32((int32_t) currPack->length, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);

    CIRCULAR_INCREMENT(stream->firstPacket, XLINK_MAX_PACKETS_PER_STREAM);
    stream->blockedPackets--;
    if (releasedSize) {
        *releasedSize = currPack->length;
    }
    return 0;
}

int addNewPacketToStream(streamDesc_t* stream, void* buffer, uint32_t size) {
    if (stream->availablePackets + stream->blockedPackets < XLINK_MAX_PACKETS_PER_STREAM)
    {
        stream->packets[stream->firstPacketFree].data = buffer;
        stream->packets[stream->firstPacketFree].length = size;
        CIRCULAR_INCREMENT(stream->firstPacketFree, XLINK_MAX_PACKETS_PER_STREAM);
        stream->availablePackets++;
        return 0;
    }
    return -1;
}

int handleIncomingEvent(xLinkEvent_t* event) {
    //this function will be dependent whether this is a client or a Remote
    //specific actions to this peer
    mvLog(MVLOG_DEBUG, "%s, size %u, streamId %u.\n", TypeToStr(event->header.type), event->header.size, event->header.streamId);

    ASSERT_X_LINK(event->header.type >= XLINK_WRITE_REQ
        && event->header.type != XLINK_REQUEST_LAST
        && event->header.type < XLINK_RESP_LAST);

    if(event->header.type == XLINK_WRITE_REQ) {
        streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
        ASSERT_X_LINK(stream);

        stream->localFillLevel += event->header.size;
        mvLog(MVLOG_DEBUG,"S%d: Got write of %ld, current local fill level is %ld out of %ld %ld\n",
              event->header.streamId, event->header.size, stream->localFillLevel, stream->readSize, stream->writeSize);

        void* buffer = XLinkPlatformAllocateData(ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
        if (buffer == NULL){
            mvLog(MVLOG_FATAL,"out of memory\n");
            releaseStream(stream);
            return -1;
        }
        const int sc = XLinkPlatformRead(&event->deviceHandle, buffer, event->header.size);
        if(sc < 0){
            mvLog(MVLOG_ERROR,"%s() Read failed %d\n", __func__, (int)sc);
            XLinkPlatformDeallocateData(buffer, ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
            releaseStream(stream);
            return -1;
        }

        event->data = buffer;
        if (addNewPacketToStream(stream, buffer, event->header.size)){
            mvLog(MVLOG_WARN,"No more place in stream. release packet\n");
            XLinkPlatformDeallocateData(buffer, ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
            XLINK_EVENT_NOT_ACKNOWLEDGE(event);
            releaseStream(stream);
            return -1;
        }
        releaseStream(stream);
    }

    return 0;
}

// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------
