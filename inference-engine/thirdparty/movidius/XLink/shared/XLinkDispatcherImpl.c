// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>
#include "stdlib.h"

#include "mvMacros.h"
#include "XLinkTool.h"
#include "XLinkPlatform.h"
#include "XLinkDispatcherImpl.h"
#include "XLinkPrivateFields.h"
#include "XLinkDispatcher.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "mvLog.h"
#include "mvStringUtils.h"

// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

static int isStreamSpaceEnoughFor(streamDesc_t* stream, uint32_t size);
static int is_semaphore_initialized(const streamDesc_t *stream);

static streamPacketDesc_t* getPacketFromStream(streamDesc_t* stream);
static int releasePacketFromStream(streamDesc_t* stream, uint32_t* releasedSize);
static int addNewPacketToStream(streamDesc_t* stream, void* buffer, uint32_t size);

static int handleIncomingEvent(xLinkEvent_t* event);

#ifdef __PC__
static void setEventFailed(xLinkEvent_t * event );
#endif

static int getNextAvailableStreamIndex(xLinkDesc_t* link);

static void deallocateStream(streamDesc_t* stream);
static streamId_t allocateNewStream(void* fd,
                                    const char* name,
                                    uint32_t writeSize,
                                    uint32_t readSize,
                                    streamId_t forcedId);


// ------------------------------------
// Helpers declaration. End.
// ------------------------------------



// ------------------------------------
// XLinkDispatcherImpl.h implementation. Begin.
// ------------------------------------

//adds a new event with parameters and returns event id
int dispatcherEventSend(xLinkEvent_t *event)
{
    mvLog(MVLOG_DEBUG, "%s, size %d, streamId %d.\n", TypeToStr(event->header.type), event->header.size, event->header.streamId);
    int rc = XLinkPlatformWrite(&event->deviceHandle, &event->header, sizeof(event->header));

    if(rc < 0)
    {
        mvLog(MVLOG_ERROR,"Write failed (header) (err %d) | event %s\n", rc, TypeToStr(event->header.type));
        return rc;
    }
    if (event->header.type == XLINK_WRITE_REQ)
    {
        //write requested data
        rc = XLinkPlatformWrite(&event->deviceHandle, event->data,
                                event->header.size);
        if(rc < 0) {
            mvLog(MVLOG_ERROR,"Write failed %d\n", rc);
#ifndef __PC__
            return rc;
#endif
        }
    }
    // this function will send events to the remote node
    return 0;
}

int dispatcherEventReceive(xLinkEvent_t* event){
    static xLinkEvent_t prevEvent = {0};
    int sc = XLinkPlatformRead(&event->deviceHandle, &event->header, sizeof(event->header));

    mvLog(MVLOG_DEBUG,"Incoming event %p: %s %d %p prevEvent: %s %d %p\n",
          event,
          TypeToStr(event->header.type),
          (int)event->header.id,
          event->deviceHandle.xLinkFD,
          TypeToStr(prevEvent.header.type),
          (int)prevEvent.header.id,
          prevEvent.deviceHandle.xLinkFD);


    if(sc < 0) {
        xLinkDesc_t* link = getLink(&event->deviceHandle.xLinkFD);
        if (event->header.type == XLINK_RESET_RESP || link == NULL) {
            return sc;
        } else if (link->hostClosedFD) {
            //host intentionally closed usb, finish normally
            event->header.type = XLINK_RESET_RESP;
            return 0;
        }
    }

    if(sc < 0) {
        mvLog(MVLOG_ERROR,"%s() Read failed %d\n", __func__, (int)sc);
        return sc;
    }

    if (prevEvent.header.id == event->header.id &&
        prevEvent.header.type == event->header.type &&
        prevEvent.deviceHandle.xLinkFD == event->deviceHandle.xLinkFD)
    {
        mvLog(MVLOG_FATAL,"Duplicate id detected. \n");
    }

    prevEvent = *event;
    if (handleIncomingEvent(event) != 0) {
        mvLog(MVLOG_WARN,"Failed to handle incoming event");
    }

    if(event->header.type == XLINK_RESET_REQ)
    {
        if(event->deviceHandle.protocol == X_LINK_PCIE) {
            mvLog(MVLOG_DEBUG,"XLINK_RESET_REQ received - doing nothing, we dont want to reset device");
        }
        else {
            return -1;
        }
    }

    return 0;
}

//this function should be called only for remote requests
int dispatcherLocalEventGetResponse(xLinkEvent_t* event, xLinkEvent_t* response)
{
    streamDesc_t* stream;
    response->header.id = event->header.id;
    mvLog(MVLOG_DEBUG, "%s\n",TypeToStr(event->header.type));
    switch (event->header.type){
        case XLINK_WRITE_REQ:
            //in case local tries to write after it issues close (writeSize is zero)
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);

#ifdef __PC__
            if(!stream){
                mvLog(MVLOG_DEBUG, "stream %d has been closed!\n", event->header.streamId);
                setEventFailed(event);
                break;
            }
#else
            ASSERT_X_LINK(stream);
#endif

            if (stream->writeSize == 0)
            {
                event->header.flags.bitField.nack = 1;
                event->header.flags.bitField.ack = 0;
                // return -1 to don't even send it to the remote
                releaseStream(stream);
                return -1;
            }
            event->header.flags.bitField.ack = 1;
            event->header.flags.bitField.nack = 0;
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
        case XLINK_READ_REQ:
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
#ifdef __PC__
            if(!stream){
                mvLog(MVLOG_DEBUG, "stream %d has been closed!\n", event->header.streamId);
                setEventFailed(event);
                break;
            }
#else
            ASSERT_X_LINK(stream);
#endif
            streamPacketDesc_t* packet = getPacketFromStream(stream);
            if (packet){
                //the read can be served with this packet
                event->data = packet;
                event->header.flags.bitField.ack = 1;
                event->header.flags.bitField.nack = 0;
                event->header.flags.bitField.block = 0;
            }
            else{
                event->header.flags.bitField.block = 1;
                // TODO: easy to implement non-blocking read here, just return nack
            }
            event->header.flags.bitField.localServe = 1;
            releaseStream(stream);
            break;
        case XLINK_READ_REL_REQ:
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            ASSERT_X_LINK(stream);
            uint32_t releasedSize = 0;
            releasePacketFromStream(stream, &releasedSize);
            event->header.size = releasedSize;
            releaseStream(stream);
            break;
        case XLINK_CREATE_STREAM_REQ:
            break;
        case XLINK_CLOSE_STREAM_REQ:
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);

            ASSERT_X_LINK(stream);
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
        case XLINK_RESET_REQ:
            mvLog(MVLOG_DEBUG,"XLINK_RESET_REQ - do nothing\n");
            break;
        case XLINK_PING_REQ:
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
            //let remote write immediately as we have a local buffer for the data
            response->header.type = XLINK_WRITE_RESP;
            response->header.size = event->header.size;
            response->header.streamId = event->header.streamId;
            response->header.flags.bitField.ack = 1;
            response->deviceHandle = event->deviceHandle;

            // we got some data. We should unblock a blocked read
            int xxx = dispatcherUnblockEvent(-1,
                                             XLINK_READ_REQ,
                                             response->header.streamId,
                                             event->deviceHandle.xLinkFD);
            (void) xxx;
            mvLog(MVLOG_DEBUG,"unblocked from stream %d %d\n",
                  (int)response->header.streamId, (int)xxx);
            break;
        case XLINK_READ_REQ:
            break;
        case XLINK_READ_REL_REQ:
            response->header.flags.bitField.ack = 1;
            response->header.flags.bitField.nack = 0;
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

            dispatcherUnblockEvent(-1, XLINK_WRITE_REQ, event->header.streamId,
                                   event->deviceHandle.xLinkFD);
            //with every released packet check if the stream is already marked for close
            if (stream->closeStreamInitiated && stream->localFillLevel == 0)
            {
                mvLog(MVLOG_DEBUG,"%s() Unblock close STREAM\n", __func__);
                int xxx = dispatcherUnblockEvent(-1,
                                                 XLINK_CLOSE_STREAM_REQ,
                                                 event->header.streamId,
                                                 event->deviceHandle.xLinkFD);
                (void) xxx;
            }
            break;
        case XLINK_CREATE_STREAM_REQ:
            response->header.flags.bitField.ack = 1;
            response->header.type = XLINK_CREATE_STREAM_RESP;
            //write size from remote means read size for this peer
            response->header.streamId = allocateNewStream(event->deviceHandle.xLinkFD,
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
                response->header.flags.bitField.ack = 1; //All is good, we are done
                response->header.flags.bitField.nack = 0;
                mvLog(MVLOG_DEBUG,"%s() got a close stream on aready closed stream\n", __func__);
            } else {
                if (stream->localFillLevel == 0)
                {
                    response->header.flags.bitField.ack = 1;
                    response->header.flags.bitField.nack = 0;

                    deallocateStream(stream);
                    if (!stream->writeSize) {
                        stream->id = INVALID_STREAM_ID;
                        stream->name[0] = '\0';
                    }
                }
                else
                {
                    mvLog(MVLOG_DEBUG,"%s():fifo is NOT empty returning NACK \n", __func__);
                    response->header.flags.bitField.nack = 1;
                    stream->closeStreamInitiated = 1;
                }

                releaseStream(stream);
            }
            break;
        }
        case XLINK_PING_REQ:
            response->header.type = XLINK_PING_RESP;
            response->header.flags.bitField.ack = 1;
            response->deviceHandle = event->deviceHandle;
            sem_post(&pingSem);
            break;
        case XLINK_RESET_REQ:
            mvLog(MVLOG_DEBUG,"reset request - received! Sending ACK *****\n");
            response->header.flags.bitField.ack = 1;
            response->header.flags.bitField.nack = 0;
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
            response->header.streamId = allocateNewStream(event->deviceHandle.xLinkFD,
                                                          event->header.streamName,
                                                          event->header.size,0,
                                                          event->header.streamId);
#ifndef __PC__
            ASSERT_X_LINK_R(response->header.streamId != INVALID_STREAM_ID, X_LINK_ERROR);
#endif
            response->deviceHandle = event->deviceHandle;
            break;
        }
        case XLINK_CLOSE_STREAM_RESP:
        {
            streamDesc_t* stream = getStreamById(event->deviceHandle.xLinkFD,
                                                 event->header.streamId);

            if (!stream){
                response->header.flags.bitField.nack = 1;
                response->header.flags.bitField.ack = 0;
                break;
            }
            stream->writeSize = 0;
            if (!stream->readSize) {
                response->header.flags.bitField.nack = 1;
                response->header.flags.bitField.ack = 0;
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

#ifndef __PC__
    link->peerState = X_LINK_COMMUNICATION_NOT_OPEN;
#else
    link->peerState = XLINK_NOT_INIT;
#endif

    link->id = INVALID_LINK_ID;
    link->deviceHandle.xLinkFD = NULL;
    link->nextUniqueStreamId = 0;

    for (int index = 0; index < XLINK_MAX_STREAMS; index++) {
        streamDesc_t* stream = &link->availableStreams[index];
        if (!stream) {
            continue;
        }

        while (getPacketFromStream(stream) || stream->blockedPackets) {
            releasePacketFromStream(stream, NULL);
        }

        if (is_semaphore_initialized(stream)) {
            sem_destroy(&stream->sem);
            stream->name[0] = '\0';
        }

        stream->id = INVALID_STREAM_ID;
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
    else
        return 1;
}

int is_semaphore_initialized(const streamDesc_t *stream) {
    return stream && strnlen(stream->name, MAX_STREAM_NAME_LENGTH) != 0;
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
    void* buffer ;
    streamDesc_t* stream ;
    int sc = 0 ;
    switch (event->header.type){
        case XLINK_WRITE_REQ:
            /*If we got here, we will read the data no matter what happens.
              If we encounter any problems we will still read the data to keep
              the communication working but send a NACK.*/
            stream = getStreamById(event->deviceHandle.xLinkFD, event->header.streamId);
            ASSERT_X_LINK(stream);

            stream->localFillLevel += event->header.size;
            mvLog(MVLOG_DEBUG,"S%d: Got write of %ld, current local fill level is %ld out of %ld %ld\n",
                  event->header.streamId, event->header.size, stream->localFillLevel, stream->readSize, stream->writeSize);

            buffer = XLinkPlatformAllocateData(ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
            if (buffer == NULL){
                mvLog(MVLOG_FATAL,"out of memory\n");
                ASSERT_X_LINK(0);
            }
            sc = XLinkPlatformRead(&event->deviceHandle, buffer, event->header.size);
            if(sc < 0){
                mvLog(MVLOG_ERROR,"%s() Read failed %d\n", __func__, (int)sc);
                XLinkPlatformDeallocateData(buffer, ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
                ASSERT_X_LINK(0);
            }

            event->data = buffer;
            if (addNewPacketToStream(stream, buffer, event->header.size)){
                mvLog(MVLOG_WARN,"No more place in stream. release packet\n");
                XLinkPlatformDeallocateData(buffer, ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
                event->header.flags.bitField.ack = 0;
                event->header.flags.bitField.nack = 1;
                ASSERT_X_LINK(0);
            }
            releaseStream(stream);
            break;
        case XLINK_READ_REQ:
            break;
        case XLINK_READ_REL_REQ:
            break;
        case XLINK_CREATE_STREAM_REQ:
            break;
        case XLINK_CLOSE_STREAM_REQ:
            break;
        case XLINK_PING_REQ:
            break;
        case XLINK_RESET_REQ:
            break;
        case XLINK_WRITE_RESP:
            break;
        case XLINK_READ_RESP:
            break;
        case XLINK_READ_REL_RESP:
            break;
        case XLINK_CREATE_STREAM_RESP:
            break;
        case XLINK_CLOSE_STREAM_RESP:
            break;
        case XLINK_PING_RESP:
            break;
        case XLINK_RESET_RESP:
            break;
        default:
            ASSERT_X_LINK(0);
    }
    //adding event for the scheduler. We let it know that this is a remote event
    dispatcherAddEvent(EVENT_REMOTE, event);
    return 0;
}

#ifdef __PC__
void setEventFailed(xLinkEvent_t * event )
{
    event->header.flags.bitField.localServe = 1;
    event->header.flags.bitField.ack = 0;
    event->header.flags.bitField.nack = 1;
}
#endif

int getNextAvailableStreamIndex(xLinkDesc_t* link)
{
    if (link == NULL)
        return -1;

    int idx;
    for (idx = 0; idx < XLINK_MAX_STREAMS; idx++) {
        if (link->availableStreams[idx].id == INVALID_STREAM_ID)
            return idx;
    }

    mvLog(MVLOG_DEBUG,"%s(): - no next available stream!\n", __func__);
    return -1;
}

void deallocateStream(streamDesc_t* stream)
{
    if (stream && stream->id != INVALID_STREAM_ID)
    {
        if (stream->readSize)
        {
            stream->readSize = 0;
            stream->closeStreamInitiated = 0;
        }

#ifndef __PC__
        if (is_semaphore_initialized(stream)) {
            if(sem_destroy(&stream->sem))
                perror("Can't destroy semaphore");
        }
#endif
    }
}

streamId_t allocateNewStream(void* fd,
                             const char* name,
                             uint32_t writeSize,
                             uint32_t readSize,
                             streamId_t forcedId)
{
    streamId_t streamId;
    streamDesc_t* stream;
    xLinkDesc_t* link = getLink(fd);
    ASSERT_X_LINK_R(link != NULL, INVALID_STREAM_ID);

    stream = getStreamByName(link, name);

    if (stream != NULL)
    {
        /*the stream already exists*/
        if ((writeSize > stream->writeSize && stream->writeSize != 0) ||
            (readSize > stream->readSize && stream->readSize != 0))
        {
            mvLog(MVLOG_ERROR, "%s(): streamName Exists %d\n", __func__, (int)stream->id);
            return INVALID_STREAM_ID;
        }
    }
    else
    {
        int idx = getNextAvailableStreamIndex(link);

        if (idx == -1)
        {
            return INVALID_STREAM_ID;
        }
        stream = &link->availableStreams[idx];
        if (forcedId == INVALID_STREAM_ID)
            stream->id = link->nextUniqueStreamId;
        else
            stream->id = forcedId;
        link->nextUniqueStreamId++; //even if we didnt use a new one, we need to align with total number of  unique streams
        if (!is_semaphore_initialized(stream)) //if sem_init is called for already initiated sem, behavior is undefined
        {
            if(sem_init(&stream->sem, 0, 0))
                perror("Can't create semaphore\n");
        }
        else
        {
            mvLog(MVLOG_INFO, "is_semaphore_initialized\n");
        }

        mv_strncpy(stream->name, MAX_STREAM_NAME_LENGTH,
                   name, MAX_STREAM_NAME_LENGTH - 1);
        stream->readSize = 0;
        stream->writeSize = 0;
        stream->remoteFillLevel = 0;
        stream->remoteFillPacketLevel = 0;

        stream->localFillLevel = 0;
        stream->closeStreamInitiated = 0;
    }
    if (readSize && !stream->readSize)
    {
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
    if (writeSize && !stream->writeSize)
    {
        stream->writeSize = writeSize;
    }

    mvLog(MVLOG_DEBUG, "The stream \"%s\"  created, id = %u, readSize = %d, writeSize = %d\n",
          stream->name, stream->id, stream->readSize, stream->writeSize);

    streamId = stream->id;
    releaseStream(stream);
    return streamId;
}

// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------
