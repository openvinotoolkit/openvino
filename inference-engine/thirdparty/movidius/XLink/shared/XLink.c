// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///

#include "XLink.h"
#include "XLink_tool.h"

#include "stdio.h"
#include "stdint.h"
#include "string.h"
#include "time.h"

#include <assert.h>
#include <stdlib.h>


#if (defined(_WIN32) || defined(_WIN64))
#include "gettime.h"
#include "win_pthread.h"
#include "win_semaphore.h"
#else
# ifdef __APPLE__
#  include "pthread_semaphore.h"
# else
#  include <semaphore.h>
# endif
#endif

#include "mvMacros.h"
#include "XLinkPlatform.h"
#include "XLinkDispatcher.h"
#include "XLinkPublicDefines.h"

#define _XLINK_ENABLE_PRIVATE_INCLUDE_
#include "XLinkPrivateDefines.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "mvLog.h"
#include "mvStringUtils.h"

#ifndef XLINK_USB_DATA_TIMEOUT
#define XLINK_USB_DATA_TIMEOUT 0
#endif

#ifndef XLINK_COMMON_TIMEOUT_MSEC
#define XLINK_COMMON_TIMEOUT_MSEC (1*60*1000)
#endif

#define DEFAULT_TIMEOUT ((unsigned int)-1)
#define MAX_PATH_LENGTH (255)

static unsigned int glCommonTimeOutMsec = XLINK_COMMON_TIMEOUT_MSEC;
static unsigned int glDeviceOpenTimeOutMsec = 5000;
static unsigned int glAllocateGraphTimeOutMsec = 12000;

XLinkError_t XLinkSetCommonTimeOutMsec(unsigned int msec) {
    glCommonTimeOutMsec = msec;
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkSetDeviceOpenTimeOutMsec(unsigned int msec) {
    glDeviceOpenTimeOutMsec = msec;
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkSetAllocateGraphTimeOutMsec(unsigned int msec) {
    glAllocateGraphTimeOutMsec = msec;
    return X_LINK_SUCCESS;
}

int XLinkWaitSem(sem_t* sem)
{
#ifdef __PC__
    ASSERT_X_LINK_R(sem != NULL, -1);

    if (glCommonTimeOutMsec == 0)
    {
        return sem_wait(sem);
    }
    else
    {
        struct timespec ts;
        uint64_t timeout_counter =  (uint64_t)glCommonTimeOutMsec * 1000 * 1000;
        uint64_t overflow;

        if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
        {
            return -1;
        }
        overflow = timeout_counter + ts.tv_nsec;
        ts.tv_sec += overflow / 1000000000ul;
        ts.tv_nsec = overflow % 1000000000ul;

        return sem_timedwait(sem, &ts);
    }
#else
    return sem_wait(sem);
#endif
}

int XLinkWaitSemUserMode(sem_t* sem, unsigned int timeout)
{
    ASSERT_X_LINK_R(sem != NULL, -1);

    if (timeout == 0)
    {
        return sem_wait(sem);
    }
    else if (timeout == DEFAULT_TIMEOUT)
    {
        return XLinkWaitSem(sem);
    }
    else
    {
        struct timespec ts;
        uint64_t timeout_counter =  (uint64_t)timeout * 1000 * 1000;
        uint64_t overflow;

        if (clock_gettime(CLOCK_REALTIME, &ts) == -1)
        {
            return -1;
        }
        overflow = timeout_counter + ts.tv_nsec;
        ts.tv_sec += overflow / 1000000000ul;
        ts.tv_nsec = overflow % 1000000000ul;

        return sem_timedwait(sem, &ts);
    }
}

static int is_semaphore_initialized(const streamDesc_t *stream) {
    return stream && strnlen(stream->name, MAX_STREAM_NAME_LENGTH) != 0;
}

int dispatcherLocalEventGetResponse(xLinkEvent_t* event, xLinkEvent_t* response);
int dispatcherRemoteEventGetResponse(xLinkEvent_t* event, xLinkEvent_t* response);
//adds a new event with parameters and returns event id
int dispatcherEventSend(xLinkEvent_t* event);
streamDesc_t* getStreamById(void* fd, streamId_t id);
void releaseStream(streamDesc_t*);
int addNewPacketToStream(streamDesc_t* stream, void* buffer, uint32_t size);
xLinkDesc_t* getLink(void* fd);

#ifdef __PC__
static XLinkError_t checkEventHeader(xLinkEventHeader_t header);
#endif

struct dispatcherControlFunctions controlFunctionTbl;
XLinkGlobalHandler_t* glHandler; //TODO need to either protect this with semaphor
                                 //or make profiling data per device
linkId_t nextUniqueLinkId = 0; //incremental number, doesn't get decremented.

xLinkDesc_t availableXLinks[MAX_LINKS];

xLinkDesc_t* getLinkById(linkId_t id);
xLinkDesc_t* getLink(void* fd);
sem_t  pingSem; //to b used by myriad


char* TypeToStr(int type)
{
    switch(type)
    {
        case XLINK_WRITE_REQ:     return "XLINK_WRITE_REQ";
        case XLINK_READ_REQ:      return "XLINK_READ_REQ";
        case XLINK_READ_REL_REQ:  return "XLINK_READ_REL_REQ";
        case XLINK_CREATE_STREAM_REQ:return "XLINK_CREATE_STREAM_REQ";
        case XLINK_CLOSE_STREAM_REQ: return "XLINK_CLOSE_STREAM_REQ";
        case XLINK_PING_REQ:         return "XLINK_PING_REQ";
        case XLINK_RESET_REQ:        return "XLINK_RESET_REQ";
        case XLINK_REQUEST_LAST:     return "XLINK_REQUEST_LAST";
        case XLINK_WRITE_RESP:   return "XLINK_WRITE_RESP";
        case XLINK_READ_RESP:     return "XLINK_READ_RESP";
        case XLINK_READ_REL_RESP: return "XLINK_READ_REL_RESP";
        case XLINK_CREATE_STREAM_RESP: return "XLINK_CREATE_STREAM_RESP";
        case XLINK_CLOSE_STREAM_RESP:  return "XLINK_CLOSE_STREAM_RESP";
        case XLINK_PING_RESP:  return "XLINK_PING_RESP";
        case XLINK_RESET_RESP: return "XLINK_RESET_RESP";
        case XLINK_RESP_LAST:  return "XLINK_RESP_LAST";
        default:
            break;
    }
    return "";
}

static XLinkError_t parseUsbLinkPlatformError(xLinkPlatformErrorCode_t rc) {
    switch (rc) {
        case X_LINK_PLATFORM_SUCCESS:
            return X_LINK_SUCCESS;
        case X_LINK_PLATFORM_DEVICE_NOT_FOUND:
            return X_LINK_DEVICE_NOT_FOUND;
        case X_LINK_PLATFORM_TIMEOUT:
            return X_LINK_TIMEOUT;
        default:
            return X_LINK_ERROR;
    }
}

const char* XLinkErrorToStr(XLinkError_t rc) {
    switch (rc) {
        case X_LINK_SUCCESS:
            return "X_LINK_SUCCESS";
        case X_LINK_ALREADY_OPEN:
            return "X_LINK_ALREADY_OPEN";
        case X_LINK_COMMUNICATION_NOT_OPEN:
            return "X_LINK_COMMUNICATION_NOT_OPEN";
        case X_LINK_COMMUNICATION_FAIL:
            return "X_LINK_COMMUNICATION_FAIL";
        case X_LINK_COMMUNICATION_UNKNOWN_ERROR:
            return "X_LINK_COMMUNICATION_UNKNOWN_ERROR";
        case X_LINK_DEVICE_NOT_FOUND:
            return "X_LINK_DEVICE_NOT_FOUND";
        case X_LINK_TIMEOUT:
            return "X_LINK_TIMEOUT";
        case X_LINK_OUT_OF_MEMORY:
            return "X_LINK_OUT_OF_MEMORY";
        case X_LINK_ERROR:
        default:
            return "X_LINK_ERROR";
    }
}

/*#################################################################################
###################################### INTERNAL ###################################
##################################################################################*/

static float timespec_diff(struct timespec *start, struct timespec *stop)
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

int handleIncomingEvent(xLinkEvent_t* event){
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

            buffer = allocateData(ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
            if (buffer == NULL){
                mvLog(MVLOG_FATAL,"out of memory\n");
                ASSERT_X_LINK(0);
            }
            sc = XLinkRead(&event->deviceHandle, buffer, event->header.size, XLINK_USB_DATA_TIMEOUT);
            if(sc < 0){
                mvLog(MVLOG_ERROR,"%s() Read failed %d\n", __func__, (int)sc);
                deallocateData(buffer, ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
                ASSERT_X_LINK(0);
            }

            event->data = buffer;
            if (addNewPacketToStream(stream, buffer, event->header.size)){
                mvLog(MVLOG_WARN,"No more place in stream. release packet\n");
                deallocateData(buffer, ALIGN_UP(event->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
                event->header.flags.bitField.ack = 0;
                event->header.flags.bitField.nack = 1;
                assert(0);
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

int dispatcherEventReceive(xLinkEvent_t* event){
    static xLinkEvent_t prevEvent = {0};
#ifdef __PC__
    int sc = XLinkRead(&event->deviceHandle, &event->header, sizeof(event->header), 0);
#else
    int sc = XLinkRead(&event->deviceHandle, &event->header, sizeof(event->header), XLINK_USB_DATA_TIMEOUT);
#endif

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

int getLinkIndex(void* fd)
{
    int i;
    for (i = 0; i < MAX_LINKS; i++)
        if (availableXLinks[i].deviceHandle.xLinkFD == fd)
            return i;
    return -1;
}

xLinkDesc_t* getLinkById(linkId_t id)
{
    int i;
    for (i = 0; i < MAX_LINKS; i++)
        if (availableXLinks[i].id == id)
            return &availableXLinks[i];
    return NULL;
}
xLinkDesc_t* getLink(void* fd)
{
    int i;
    for (i = 0; i < MAX_LINKS; i++)
        if (availableXLinks[i].deviceHandle.xLinkFD == fd)
            return &availableXLinks[i];
    return NULL;
}

static linkId_t getNextAvailableLinkUniqueId()
{
    linkId_t start = nextUniqueLinkId;
    do
    {
        int i;
        for (i = 0; i < MAX_LINKS; i++)
        {
            if (availableXLinks[i].id != INVALID_LINK_ID &&
                availableXLinks[i].id == nextUniqueLinkId)
                break;
        }
        if (i >= MAX_LINKS)
        {
            return nextUniqueLinkId;
        }
        nextUniqueLinkId++;
        if (nextUniqueLinkId == INVALID_LINK_ID)
        {
            nextUniqueLinkId = 0;
        }
    } while (start != nextUniqueLinkId);
    mvLog(MVLOG_ERROR, "%s():- no next available link!\n", __func__);
    return INVALID_LINK_ID;
}

int getNextAvailableLinkIndex()
{
    int i;
    for (i = 0; i < MAX_LINKS; i++)
        if (availableXLinks[i].id == INVALID_LINK_ID)
            return i;

    mvLog(MVLOG_ERROR,"%s():- no next available link!\n", __func__);
    return -1;
}

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

streamDesc_t* getStreamById(void* fd, streamId_t id)
{
    xLinkDesc_t* link = getLink(fd);
    ASSERT_X_LINK_R(link != NULL, NULL);
    int stream;
    for (stream = 0; stream < XLINK_MAX_STREAMS; stream++) {
        if (link->availableStreams[stream].id == id) {
            if (XLinkWaitSem(&link->availableStreams[stream].sem)) {
#ifdef __PC__
                return NULL;
#endif
            }
            return &link->availableStreams[stream];
        }
    }
    return NULL;
}

streamDesc_t* getStreamByName(xLinkDesc_t* link, const char* name)
{
    ASSERT_X_LINK_R(link != NULL, NULL);
    int stream;
    for (stream = 0; stream < XLINK_MAX_STREAMS; stream++) {
        if (link->availableStreams[stream].id != INVALID_STREAM_ID &&
            strcmp(link->availableStreams[stream].name, name) == 0) {
            if (XLinkWaitSem(&link->availableStreams[stream].sem)) {
#ifdef __PC__
                return NULL;
#endif
            }
            return &link->availableStreams[stream];
        }
    }
    return NULL;
}

void releaseStream(streamDesc_t* stream)
{
    if (stream && stream->id != INVALID_STREAM_ID) {
        sem_post(&stream->sem);
    }
    else {
        mvLog(MVLOG_DEBUG,"trying to release a semaphore for a released stream\n");
    }
}

streamId_t getStreamIdByName(xLinkDesc_t* link, const char* name)
{
    streamDesc_t* stream = getStreamByName(link, name);
    streamId_t id;
    if (stream) {
        id = stream->id;
        releaseStream(stream);
        return id;
    }
    else
        return INVALID_STREAM_ID;
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

    deallocateData(currPack->data,
                   ALIGN_UP_INT32((int32_t)currPack->length, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);

    CIRCULAR_INCREMENT(stream->firstPacket, XLINK_MAX_PACKETS_PER_STREAM);
    stream->blockedPackets--;
    if (releasedSize) {
        *releasedSize = currPack->length;
    }
    return 0;
}

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

int addNewPacketToStream(streamDesc_t* stream, void* buffer, uint32_t size){
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
        void *buffer = allocateData(ALIGN_UP(readSize, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
        if (buffer == NULL) {
            mvLog(MVLOG_ERROR,"Cannot create stream. Requested memory = %u", stream->readSize);
            return INVALID_STREAM_ID;
        } else {
            deallocateData(buffer, ALIGN_UP(readSize, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
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

#ifdef __PC__
static void setEventFailed(xLinkEvent_t * event )
{
    event->header.flags.bitField.localServe = 1;
    event->header.flags.bitField.ack = 0;
    event->header.flags.bitField.nack = 1;
}
#endif

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
//adds a new event with parameters and returns event id
int dispatcherEventSend(xLinkEvent_t *event)
{
    mvLog(MVLOG_DEBUG, "%s, size %d, streamId %d.\n", TypeToStr(event->header.type), event->header.size, event->header.streamId);
#ifdef __PC__
    int rc = XLinkWrite(&event->deviceHandle, &event->header, sizeof(event->header), XLINK_USB_DATA_TIMEOUT);
#else
    int rc = XLinkWrite(&event->deviceHandle, &event->header, sizeof(event->header), 0);
#endif

    if(rc < 0)
    {
        mvLog(MVLOG_ERROR,"Write failed (header) (err %d) | event %s\n", rc, TypeToStr(event->header.type));
        return rc;
    }
    if (event->header.type == XLINK_WRITE_REQ)
    {
        //write requested data
        rc = XLinkWrite(&event->deviceHandle, event->data,
                        event->header.size, XLINK_USB_DATA_TIMEOUT);
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

static xLinkState_t getXLinkState(xLinkDesc_t* link)
{
    ASSERT_X_LINK_R(link != NULL, XLINK_NOT_INIT);
    mvLog(MVLOG_DEBUG,"%s() link %p link->peerState %d\n", __func__,link, link->peerState);
    return link->peerState;
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

/*#################################################################################
###################################### EXTERNAL ###################################
##################################################################################*/

//Called only from app - per device
XLinkError_t XLinkConnect(XLinkHandler_t* handler)
{
    ASSERT_X_LINK(handler);
    if (strnlen(handler->devicePath, MAX_PATH_LENGTH) < 2) {
        mvLog(MVLOG_ERROR, "Device path is incorrect");
        return X_LINK_ERROR;
    }

    int index = getNextAvailableLinkIndex();
    ASSERT_X_LINK(index != -1);

    xLinkDesc_t* link = &availableXLinks[index];
    mvLog(MVLOG_DEBUG,"%s() device name %s glHandler %p protocol %d\n", __func__, handler->devicePath, glHandler, handler->protocol);

    link->deviceHandle.protocol = handler->protocol;
    if (XLinkPlatformConnect(handler->devicePath2, handler->devicePath,
                             link->deviceHandle.protocol, &link->deviceHandle.xLinkFD) < 0) {
        return X_LINK_ERROR;
    }

    if (dispatcherStart(&link->deviceHandle))
        return X_LINK_TIMEOUT;

    xLinkEvent_t event = {0};

    event.header.type = XLINK_PING_REQ;
    event.deviceHandle = link->deviceHandle;
    dispatcherAddEvent(EVENT_LOCAL, &event);

    if (dispatcherWaitEventComplete(&link->deviceHandle, glDeviceOpenTimeOutMsec)) {
        dispatcherClean(link->deviceHandle.xLinkFD);
        return X_LINK_TIMEOUT;
    }

    link->id = getNextAvailableLinkUniqueId();
    link->peerState = XLINK_UP;
    link->hostClosedFD = 0;
    handler->linkId = link->id;
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkInitialize(XLinkGlobalHandler_t* handler)
{
#ifndef __PC__
    mvLogLevelSet(MVLOG_FATAL);
    mvLogDefaultLevelSet(MVLOG_FATAL);
#endif

    ASSERT_X_LINK(handler);
    ASSERT_X_LINK(XLINK_MAX_STREAMS <= MAX_POOLS_ALLOC);
    glHandler = handler;
    if (sem_init(&pingSem,0,0)) {
        mvLog(MVLOG_ERROR, "Can't create semaphore\n");
    }
    int i;

    XLinkPlatformInit();

    //Using deprecated fields. Begin.
    int loglevel = handler->loglevel;
    int protocol = handler->protocol;
    //Using deprecated fields. End.

    memset((void*)handler, 0, sizeof(XLinkGlobalHandler_t));

    //Using deprecated fields. Begin.
    handler->loglevel = loglevel;
    handler->protocol = protocol;
    //Using deprecated fields. End.

    //initialize availableStreams
    xLinkDesc_t* link;
    for (i = 0; i < MAX_LINKS; i++) {
        link = &availableXLinks[i];
        link->id = INVALID_LINK_ID;
        link->deviceHandle.xLinkFD = NULL;
        link->peerState = XLINK_NOT_INIT;
        int stream;
        for (stream = 0; stream < XLINK_MAX_STREAMS; stream++)
            link->availableStreams[stream].id = INVALID_STREAM_ID;
    }

    controlFunctionTbl.eventReceive = &dispatcherEventReceive;
    controlFunctionTbl.eventSend = &dispatcherEventSend;
    controlFunctionTbl.localGetResponse = &dispatcherLocalEventGetResponse;
    controlFunctionTbl.remoteGetResponse = &dispatcherRemoteEventGetResponse;
    controlFunctionTbl.closeLink = &dispatcherCloseLink;
    controlFunctionTbl.closeDeviceFd = &dispatcherCloseDeviceFd;

    if (dispatcherInitialize(&controlFunctionTbl))
    {
#ifdef __PC__
        return X_LINK_TIMEOUT;
#endif
    }

#ifndef __PC__
    int index = getNextAvailableLinkIndex();
    if (index == -1)
        return X_LINK_COMMUNICATION_NOT_OPEN;

    link = &availableXLinks[index];
    link->deviceHandle.xLinkFD = NULL;
    link->id = nextUniqueLinkId++;
    link->peerState = XLINK_UP;

    sem_wait(&pingSem);
#endif
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkGetFillLevel(streamId_t streamId, int isRemote, int* fillLevel)
{
    linkId_t id;
    EXTRACT_IDS(streamId,id);
    xLinkDesc_t* link = getLinkById(id);
    streamDesc_t* stream;

    ASSERT_X_LINK(link != NULL);
    if (getXLinkState(link) != XLINK_UP)
    {
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }
    stream = getStreamById(link->deviceHandle.xLinkFD, streamId);
    ASSERT_X_LINK(stream);

    if (isRemote)
        *fillLevel = stream->remoteFillLevel;
    else
        *fillLevel = stream->localFillLevel;
    releaseStream(stream);
    return X_LINK_SUCCESS;
}

streamId_t XLinkOpenStream(linkId_t id, const char* name, int stream_write_size)
{
    ASSERT_X_LINK(name);
    if (stream_write_size < 0) {
        return X_LINK_ERROR;
    }

    xLinkEvent_t event = {0};
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
        event.header.type = XLINK_CREATE_STREAM_REQ;
        mv_strncpy(event.header.streamName, MAX_STREAM_NAME_LENGTH,
                   name, MAX_STREAM_NAME_LENGTH - 1);
        event.header.size = stream_write_size;
        event.header.streamId = INVALID_STREAM_ID;
        event.deviceHandle = link->deviceHandle;

        dispatcherAddEvent(EVENT_LOCAL, &event);
        if (dispatcherWaitEventComplete(&link->deviceHandle, DEFAULT_TIMEOUT))
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

// Just like open stream, when closeStream is called
// on the local size we are resetting the writeSize
// and on the remote side we are freeing the read buffer
XLinkError_t XLinkCloseStream(streamId_t streamId)
{
    linkId_t id;
    EXTRACT_IDS(streamId,id);
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);

    mvLog(MVLOG_DEBUG,"%s(): streamId %d\n", __func__, (int)streamId);
    if (getXLinkState(link) != XLINK_UP)
        return X_LINK_COMMUNICATION_NOT_OPEN;

    xLinkEvent_t event = {0};
    event.header.type = XLINK_CLOSE_STREAM_REQ;
    event.header.streamId = streamId;
    event.deviceHandle = link->deviceHandle;
    xLinkEvent_t* ev = dispatcherAddEvent(EVENT_LOCAL, &event);
    if (ev == NULL) {
        mvLog(MVLOG_ERROR, "Dispatcher failed on adding event");
        return X_LINK_ERROR;
    }

    if (dispatcherWaitEventComplete(&link->deviceHandle, DEFAULT_TIMEOUT))
        return X_LINK_TIMEOUT;

    if (event.header.flags.bitField.ack != 1)
        return X_LINK_COMMUNICATION_FAIL;

    return X_LINK_SUCCESS;
}


XLinkError_t XLinkGetAvailableStreams(linkId_t id)
{
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);
    if (getXLinkState(link) != XLINK_UP)
    {
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }
    /*...get other statuses*/
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkFindFirstSuitableDevice(XLinkDeviceState_t state,
                                          const deviceDesc_t in_deviceRequirements,
                                          deviceDesc_t *out_foundDevice)
{
    ASSERT_X_LINK(out_foundDevice);

    xLinkPlatformErrorCode_t rc;
    rc = XLinkPlatformFindDeviceName(state, in_deviceRequirements, out_foundDevice);
    return parseUsbLinkPlatformError(rc);
}

XLinkError_t XLinkFindAllSuitableDevices(XLinkDeviceState_t state,
                                         deviceDesc_t in_deviceRequirements,
                                         deviceDesc_t *out_foundDevicesPtr,
                                         const unsigned int devicesArraySize,
                                         unsigned int* out_amountOfFoundDevices) {
    ASSERT_X_LINK(out_foundDevicesPtr);
    ASSERT_X_LINK(devicesArraySize > 0);
    ASSERT_X_LINK(out_amountOfFoundDevices);

    xLinkPlatformErrorCode_t rc;
    rc = XLinkPlatformFindArrayOfDevicesNames(
        state, in_deviceRequirements,
        out_foundDevicesPtr, devicesArraySize, out_amountOfFoundDevices);

    return parseUsbLinkPlatformError(rc);
}

static XLinkError_t writeData(streamId_t streamId, const uint8_t* buffer,
                              int size, unsigned int timeout)
{
    ASSERT_X_LINK(buffer);

    linkId_t id;
    EXTRACT_IDS(streamId,id);
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);
    if (getXLinkState(link) != XLINK_UP)
    {
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    xLinkEvent_t event = {0};
    event.header.type = XLINK_WRITE_REQ;
    event.header.size = size;
    event.header.streamId = streamId;
    event.deviceHandle = link->deviceHandle;
    event.data = (void*)buffer;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    xLinkEvent_t* ev = dispatcherAddEvent(EVENT_LOCAL, &event);
    if (ev == NULL) {
        mvLog(MVLOG_ERROR, "Dispatcher failed on adding event");
        return X_LINK_ERROR;
    }

    if (dispatcherWaitEventComplete(&link->deviceHandle, timeout))
        return X_LINK_TIMEOUT;

    clock_gettime(CLOCK_REALTIME, &end);

    if (event.header.flags.bitField.ack == 1)
    {
        //profile only on success
        if( glHandler->profEnable)
        {
            glHandler->profilingData.totalWriteBytes += size;
            glHandler->profilingData.totalWriteTime += timespec_diff(&start, &end);
        }
        return X_LINK_SUCCESS;
    }
    else
        return X_LINK_COMMUNICATION_FAIL;
}

XLinkError_t XLinkWriteData(streamId_t streamId, const uint8_t* buffer,
                            int size)
{
    return writeData(streamId, buffer, size, DEFAULT_TIMEOUT);
}

XLinkError_t XLinkWriteDataWithTimeout(streamId_t streamId, const uint8_t* buffer,
                                       int size, unsigned int timeout)
{
    return writeData(streamId, buffer, size, timeout);
}

XLinkError_t XLinkWriteGraphData(streamId_t streamId, const uint8_t* buffer, int size)
{
    return writeData(streamId, buffer, size, glAllocateGraphTimeOutMsec);
}


XLinkError_t XLinkAsyncWriteData()
{
    if (getXLinkState(NULL) != XLINK_UP)
    {
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReadData(streamId_t streamId, streamPacketDesc_t** packet)
{
    return XLinkReadDataWithTimeOut(streamId, packet, DEFAULT_TIMEOUT);
}

XLinkError_t XLinkReadDataWithTimeOut(streamId_t streamId, streamPacketDesc_t** packet, unsigned int timeout)
{
    linkId_t id;
    EXTRACT_IDS(streamId,id);
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);
    if (getXLinkState(link) != XLINK_UP)
    {
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    xLinkEvent_t event = {0};
    event.header.type = XLINK_READ_REQ;
    event.header.size = 0;
    event.header.streamId = streamId;
    event.deviceHandle = link->deviceHandle;
    event.data = NULL;

    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);

    xLinkEvent_t* ev = dispatcherAddEvent(EVENT_LOCAL, &event);
    if (ev == NULL) {
        mvLog(MVLOG_ERROR, "Dispatcher failed on adding event");
        return X_LINK_ERROR;
    }

    if (dispatcherWaitEventComplete(&link->deviceHandle, timeout))
        return X_LINK_TIMEOUT;

    if (event.data == NULL) {
        mvLog(MVLOG_ERROR, "Event data is invalid");
        return X_LINK_ERROR;
    }

    *packet = (streamPacketDesc_t *)event.data;
    clock_gettime(CLOCK_REALTIME, &end);

    if (event.header.flags.bitField.ack != 1)
        return X_LINK_COMMUNICATION_FAIL;

    if( glHandler->profEnable)
    {
        glHandler->profilingData.totalReadBytes += (*packet)->length;
        glHandler->profilingData.totalReadTime += timespec_diff(&start, &end);
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkReleaseData(streamId_t streamId)
{
    linkId_t id;
    EXTRACT_IDS(streamId,id);
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);
    if (getXLinkState(link) != XLINK_UP)
    {
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    xLinkEvent_t event = {0};
    event.header.type = XLINK_READ_REL_REQ;
    event.header.streamId = streamId;
    event.deviceHandle = link->deviceHandle;

    xLinkEvent_t* ev = dispatcherAddEvent(EVENT_LOCAL, &event);
    if (ev == NULL) {
        mvLog(MVLOG_ERROR, "Dispatcher failed on adding event");
        return X_LINK_ERROR;
    }

    if (dispatcherWaitEventComplete(&link->deviceHandle, DEFAULT_TIMEOUT))
        return X_LINK_TIMEOUT;

    if (event.header.flags.bitField.ack == 1)
        return X_LINK_SUCCESS;
    else
        return X_LINK_COMMUNICATION_FAIL;
}

XLinkError_t XLinkBoot(deviceDesc_t* deviceDesc, const char* binaryPath)
{
    if (XLinkPlatformBootRemote(deviceDesc, binaryPath) == 0)
        return X_LINK_SUCCESS;
    else
        return X_LINK_COMMUNICATION_FAIL;
}

XLinkError_t XLinkResetRemote(linkId_t id)
{
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);
    if (getXLinkState(link) != XLINK_UP)
    {
        mvLog(MVLOG_WARN, "Link is down, close connection to device without reset");
        XLinkPlatformCloseRemote(&link->deviceHandle);
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    // Add event to reset device. After sending it, dispatcher will close fd link
    xLinkEvent_t event = {0};
    event.header.type = XLINK_RESET_REQ;
    event.deviceHandle = link->deviceHandle;
    mvLog(MVLOG_DEBUG, "sending reset remote event\n");
    dispatcherAddEvent(EVENT_LOCAL, &event);
    if (dispatcherWaitEventComplete(&link->deviceHandle, glDeviceOpenTimeOutMsec))
        return X_LINK_TIMEOUT;

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkResetAll()
{
#if defined(NO_BOOT)
    mvLog(MVLOG_INFO, "Devices will not be restarted for this configuration (NO_BOOT)");
#else
    int i;
    for (i = 0; i < MAX_LINKS; i++) {
        if (availableXLinks[i].id != INVALID_LINK_ID) {
            xLinkDesc_t* link = &availableXLinks[i];
            int stream;
            for (stream = 0; stream < XLINK_MAX_STREAMS; stream++) {
                if (link->availableStreams[stream].id != INVALID_STREAM_ID) {
                    streamId_t streamId = link->availableStreams[stream].id;
                    mvLog(MVLOG_DEBUG,"%s() Closing stream (stream = %d) %d on link %d\n",
                          __func__, stream, (int) streamId, (int) link->id);
                    COMBIN_IDS(streamId, link->id);
                    if (XLinkCloseStream(streamId) != X_LINK_SUCCESS) {
                        mvLog(MVLOG_WARN,"Failed to close stream");
                    }
                }
            }
            if (XLinkResetRemote(link->id) != X_LINK_SUCCESS) {
                mvLog(MVLOG_WARN,"Failed to reset");
            }
        }
    }
#endif
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkProfStart()
{
    glHandler->profEnable = 1;
    glHandler->profilingData.totalReadBytes = 0;
    glHandler->profilingData.totalWriteBytes = 0;
    glHandler->profilingData.totalWriteTime = 0;
    glHandler->profilingData.totalReadTime = 0;
    glHandler->profilingData.totalBootCount = 0;
    glHandler->profilingData.totalBootTime = 0;

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkProfStop()
{
    glHandler->profEnable = 0;
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkProfPrint()
{
    printf("XLink profiling results:\n");
    if (glHandler->profilingData.totalWriteTime)
    {
        printf("Average write speed: %f MB/Sec\n",
               glHandler->profilingData.totalWriteBytes /
               glHandler->profilingData.totalWriteTime /
               1024.0 /
               1024.0 );
    }
    if (glHandler->profilingData.totalReadTime)
    {
        printf("Average read speed: %f MB/Sec\n",
               glHandler->profilingData.totalReadBytes /
               glHandler->profilingData.totalReadTime /
               1024.0 /
               1024.0);
    }
    if (glHandler->profilingData.totalBootCount)
    {
        printf("Average boot speed: %f sec\n",
               glHandler->profilingData.totalBootTime /
               glHandler->profilingData.totalBootCount);
    }
    return X_LINK_SUCCESS;
}

// ------------------------------------
// Deprecated API. Begin.
// ------------------------------------

XLinkError_t getDeviceName(int index, char* name, int nameSize, XLinkPlatform_t platform, XLinkDeviceState_t state)
{
    ASSERT_X_LINK(name != NULL);
    ASSERT_X_LINK(index >= 0);
    ASSERT_X_LINK(nameSize >= 0 && nameSize <= XLINK_MAX_NAME_SIZE);

    deviceDesc_t in_deviceRequirements = {};
    in_deviceRequirements.protocol = glHandler != NULL ? glHandler->protocol : USB_VSC;
    in_deviceRequirements.platform = platform;
    memset(name, 0, nameSize);

    if(index == 0)
    {
        deviceDesc_t deviceToBoot = {};
        XLinkError_t rc =
            XLinkFindFirstSuitableDevice(state, in_deviceRequirements, &deviceToBoot);
        if(rc != X_LINK_SUCCESS)
        {
            return rc;
        }

        return mv_strcpy(name, nameSize, deviceToBoot.name) == EOK ? X_LINK_SUCCESS : X_LINK_ERROR;
    }
    else
    {
        deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {};
        unsigned int numberOfDevices = 0;
        XLinkError_t rc =
            XLinkFindAllSuitableDevices(state, in_deviceRequirements,
                                        deviceDescArray, XLINK_MAX_DEVICES, &numberOfDevices);
        if(rc != X_LINK_SUCCESS)
        {
            return rc;
        }

        if((unsigned int)index >= numberOfDevices)
        {
            return X_LINK_DEVICE_NOT_FOUND;
        }

        return mv_strcpy(name, nameSize, deviceDescArray[index].name) == EOK ? X_LINK_SUCCESS : X_LINK_ERROR;
    }
}

XLinkError_t XLinkGetDeviceName(int index, char* name, int nameSize)
{
    return getDeviceName(index, name, nameSize, X_LINK_ANY_PLATFORM, X_LINK_ANY_STATE);
}
XLinkError_t XLinkGetDeviceNameExtended(int index, char* name, int nameSize, int pid)
{
    XLinkDeviceState_t state = XLinkPlatformPidToState(pid);
    XLinkPlatform_t platform = XLinkPlatformPidToPlatform(pid);

    return getDeviceName(index, name, nameSize, platform, state);
}

XLinkError_t XLinkBootRemote(const char* deviceName, const char* binaryPath)
{
    ASSERT_X_LINK(deviceName != NULL);
    ASSERT_X_LINK(binaryPath != NULL);

    deviceDesc_t deviceDesc = {};
    deviceDesc.protocol = glHandler != NULL ? glHandler->protocol : USB_VSC;
    mv_strcpy(deviceDesc.name, XLINK_MAX_NAME_SIZE, deviceName);

    return XLinkBoot(&deviceDesc, binaryPath);
}

XLinkError_t XLinkDisconnect(linkId_t id)
{
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);

    link->hostClosedFD = 1;
    return XLinkPlatformCloseRemote(&link->deviceHandle);
}

// ------------------------------------
// Deprecated API. End.
// ------------------------------------

/* end of file */
