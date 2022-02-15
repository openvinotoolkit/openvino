// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>
#include "stdlib.h"

#include "XLinkMacros.h"
#include "XLinkErrorUtils.h"
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
static int releaseSpecificPacketFromStream(streamDesc_t* stream, uint32_t* releasedSize, uint8_t* data);
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
    mvLog(MVLOG_DEBUG, "Send event: %s, size %u, streamId %u.\n",
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

            if (!isStreamSpaceEnoughFor(stream, event->header.size)) {
                mvLog(MVLOG_DEBUG,"local NACK RTS. stream '%s' is full (event %d)\n", stream->name, event->header.id);
                event->header.flags.bitField.block = 1;
                event->header.flags.bitField.localServe = 1;
                // TODO: easy to implement non-blocking read here, just return nack
                mvLog(MVLOG_WARN, "Blocked event would cause dispatching thread to wait on semaphore infinitely\n");
            } else {
                event->header.flags.bitField.block = 0;
                stream->remoteFillLevel += event->header.size;
                stream->remoteFillPacketLevel++;
                mvLog(MVLOG_DEBUG,"S%u: Got local write of %u , remote fill level %u out of %u %u\n",
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
            ASSERT_XLINK(stream);
            XLINK_EVENT_ACKNOWLEDGE(event);
            uint32_t releasedSize = 0;
            releasePacketFromStream(stream, &releasedSize);
            event->header.size = releasedSize;
            releaseStream(stream);
            break;
        }
        case XLINK_READ_REL_SPEC_REQ:
        {
            uint8_t* data = (uint8_t*)event->data;
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            ASSERT_XLINK(stream);
            XLINK_EVENT_ACKNOWLEDGE(event);
            uint32_t releasedSize = 0;
            releaseSpecificPacketFromStream(stream, &releasedSize, data);
            event->header.size = releasedSize;
            releaseStream(stream);
            break;
        }
        case XLINK_CREATE_STREAM_REQ:
        {
            XLINK_EVENT_ACKNOWLEDGE(event);
#ifdef __PC__
            event->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                            event->header.streamName,
                                                            event->header.size, 0,
                                                            INVALID_STREAM_ID);
            mvLog(MVLOG_DEBUG, "XLINK_CREATE_STREAM_REQ - stream has been just opened with id %u\n",
                  event->header.streamId);
#else
            mvLog(MVLOG_DEBUG, "XLINK_CREATE_STREAM_REQ - do nothing. Stream will be "
                  "opened with forced id accordingly to response from the host\n");
#endif
            break;
        }
        case XLINK_CLOSE_STREAM_REQ:
        {
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);

            ASSERT_XLINK(stream);
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
        case XLINK_READ_REL_SPEC_RESP:
        case XLINK_CREATE_STREAM_RESP:
        case XLINK_CLOSE_STREAM_RESP:
        case XLINK_PING_RESP:
            break;
        case XLINK_RESET_RESP:
            //should not happen
            event->header.flags.bitField.localServe = 1;
            break;
        default:
        {
            mvLog(MVLOG_ERROR,
                  "Fail to get response for local event. type: %d, stream name: %s\n",
                  event->header.type, event->header.streamName);
            ASSERT_XLINK(0);
        }
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
        case XLINK_READ_REL_SPEC_REQ:
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_READ_REL_SPEC_RESP;
            response->deviceHandle = event->deviceHandle;
            stream = getStreamById(event->deviceHandle.xLinkFD,
                                   event->header.streamId);
            ASSERT_XLINK(stream);
            stream->remoteFillLevel -= event->header.size;
            stream->remoteFillPacketLevel--;

            mvLog(MVLOG_DEBUG,"S%u: Got remote release of %u, remote fill level %u out of %u %u\n",
                  event->header.streamId, event->header.size, stream->remoteFillLevel, stream->writeSize, stream->readSize);
            releaseStream(stream);

            DispatcherUnblockEvent(-1, XLINK_WRITE_REQ, event->header.streamId,
                                   event->deviceHandle.xLinkFD);
            //with every released packet check if the stream is already marked for close
            if (stream->closeStreamInitiated && stream->localFillLevel == 0)
            {
                mvLog(MVLOG_DEBUG,"%s() Unblock close STREAM\n", __func__);
                DispatcherUnblockEvent(-1,
                                       XLINK_CLOSE_STREAM_REQ,
                                       event->header.streamId,
                                       event->deviceHandle.xLinkFD);
            }
            break;
        case XLINK_READ_REL_REQ:
            XLINK_EVENT_ACKNOWLEDGE(response);
            response->header.type = XLINK_READ_REL_RESP;
            response->deviceHandle = event->deviceHandle;
            stream = getStreamById(event->deviceHandle.xLinkFD,
                                   event->header.streamId);
            ASSERT_XLINK(stream);
            stream->remoteFillLevel -= event->header.size;
            stream->remoteFillPacketLevel--;

            mvLog(MVLOG_DEBUG,"S%u: Got remote release of %u, remote fill level %u out of %u %u\n",
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
#ifndef __PC__
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               0, event->header.size,
                                                               event->header.streamId);
#else
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               0, event->header.size,
                                                               INVALID_STREAM_ID);
#endif
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
                    if(XLink_sem_destroy(&stream->sem))
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
        case XLINK_READ_REL_SPEC_RESP:
            break;
        case XLINK_CREATE_STREAM_RESP:
        {
            // write_size from the response the size of the buffer from the remote
#ifndef __PC__
            response->header.streamId = XLinkAddOrUpdateStream(event->deviceHandle.xLinkFD,
                                                               event->header.streamName,
                                                               event->header.size, 0,
                                                               event->header.streamId);
            XLINK_RET_IF(response->header.streamId
                == INVALID_STREAM_ID);
            mvLog(MVLOG_DEBUG, "XLINK_CREATE_STREAM_REQ - stream has been just opened "
                  "with forced id=%ld accordingly to response from the host\n",
                  response->header.streamId);
#endif
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
        {
            mvLog(MVLOG_ERROR,
                "Fail to get response for remote event. type: %d, stream name: %s\n",
                event->header.type, event->header.streamName);
            ASSERT_XLINK(0);
        }
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

    if(XLink_sem_destroy(&link->dispatcherClosedSem)) {
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
        mvLog(MVLOG_DEBUG, "S%u: Not enough space in stream '%s' for %u: PKT %u, FILL %u SIZE %u\n",
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
    mvLog(MVLOG_DEBUG, "S%u: Got release of %u , current local fill level is %u out of %u %u\n",
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

int releaseSpecificPacketFromStream(streamDesc_t* stream, uint32_t* releasedSize, uint8_t* data) {
    if (stream->blockedPackets == 0) {
        mvLog(MVLOG_ERROR,"There is no packet to release\n");
        return 0; // ignore this, although this is a big problem on application side
    }

    uint32_t packetId = stream->firstPacket;
    uint32_t found = 0;
    do {
        if (stream->packets[packetId].data == data) {
            found = 1;
            break;
        }
        CIRCULAR_INCREMENT(packetId, XLINK_MAX_PACKETS_PER_STREAM);
    } while (packetId != stream->firstPacketUnused);
    ASSERT_XLINK(found);

    streamPacketDesc_t* currPack = &stream->packets[packetId];
    if (currPack->length == 0) {
        mvLog(MVLOG_ERROR, "Packet with ID %d is empty\n", packetId);
    }

    stream->localFillLevel -= currPack->length;

  mvLog(MVLOG_DEBUG, "S%u: Got release of %u , current local fill level is %u out of %u %u\n",
          stream->id, currPack->length, stream->localFillLevel, stream->readSize, stream->writeSize);
    XLinkPlatformDeallocateData(currPack->data,
                                ALIGN_UP_INT32((int32_t) currPack->length, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
    stream->blockedPackets--;
    if (releasedSize) {
        *releasedSize = currPack->length;
    }

    if (packetId != stream->firstPacket) {
        uint32_t currIndex = packetId;
        uint32_t nextIndex = currIndex;
        CIRCULAR_INCREMENT(nextIndex, XLINK_MAX_PACKETS_PER_STREAM);
        while (currIndex != stream->firstPacketFree) {
            stream->packets[currIndex] = stream->packets[nextIndex];
            currIndex = nextIndex;
            CIRCULAR_INCREMENT(nextIndex, XLINK_MAX_PACKETS_PER_STREAM);
        }
        CIRCULAR_DECREMENT(stream->firstPacketUnused, (XLINK_MAX_PACKETS_PER_STREAM - 1));
        CIRCULAR_DECREMENT(stream->firstPacketFree, (XLINK_MAX_PACKETS_PER_STREAM - 1));

    } else {
        CIRCULAR_INCREMENT(stream->firstPacket, XLINK_MAX_PACKETS_PER_STREAM);
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

    ASSERT_XLINK(event->header.type >= XLINK_WRITE_REQ
               && event->header.type != XLINK_REQUEST_LAST
               && event->header.type < XLINK_RESP_LAST);

    // Then read the data buffer, which is contained only in the XLINK_WRITE_REQ event
    if(event->header.type != XLINK_WRITE_REQ) {
        return 0;
    }

    int rc = -1;
    streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
    ASSERT_XLINK(stream);

    stream->localFillLevel += event->header.size;
    mvLog(MVLOG_DEBUG,"S%u: Got write of %u, current local fill level is %u out of %u %u\n",
          event->header.streamId, event->header.size, stream->localFillLevel, stream->readSize, stream->writeSize);

    void* buffer = XLinkPlatformAllocateData(ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
    XLINK_OUT_WITH_LOG_IF(buffer == NULL,
        mvLog(MVLOG_FATAL,"out of memory to receive data of size = %lu\n", event->header.size));

    const int sc = XLinkPlatformRead(&event->deviceHandle, buffer, event->header.size);
    XLINK_OUT_WITH_LOG_IF(sc < 0, mvLog(MVLOG_ERROR,"%s() Read failed %d\n", __func__, sc));

    event->data = buffer;
    XLINK_OUT_WITH_LOG_IF(addNewPacketToStream(stream, buffer, event->header.size),
        mvLog(MVLOG_WARN,"No more place in stream. release packet\n"));
    rc = 0;

XLINK_OUT:
    releaseStream(stream);

    if(rc != 0) {
        if(buffer != NULL) {
            XLinkPlatformDeallocateData(buffer,
                ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
        }
        XLINK_EVENT_NOT_ACKNOWLEDGE(event);
    }

    return rc;
}

// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------
