// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // fix for warning: implicit declaration of function ‘pthread_setname_np’
#endif

#if (defined(_WIN32) || defined(_WIN64))
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif

#include <time.h>
#include <errno.h>
#include <assert.h>
#include <stdlib.h>
#include <semaphore.h>
#include <stdio.h>
#include <string.h>

#include "XLinkDispatcherNew.h"
#include "XLinkPrivateDefines.h"
#include "XLinkTool.h"
#include "XLinkPlatform.h"
#include "XLinkMacros.h"
#include "XLinkStream.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLinkDispatcherNew
#endif
#include "XLinkLog.h"

struct DispatcherNew_t {
    DispatcherStatus_t status;
    xLinkDeviceHandle_t* deviceHandle;

    StreamDispatcher* streamDispatcher;
    BlockingQueue* packetsToSendQueue;
    BlockingQueue* receivedPacketsQueue[MAX_STREAMS_NEW];

    pthread_t sendPacketsThread;
    pthread_t receivePacketsThread;
};

// ------------------------------------
// Private methods declaration. Begin.
// ------------------------------------

static PacketNew* _dispatcher_ReadPacketData(DispatcherNew* dispatcher);
static XLinkError_t _dispatcher_WritePacketData(DispatcherNew* dispatcher, PacketNew* packet);

static XLinkError_t _dispatcher_HandleReadPacketError(DispatcherNew* dispatcher);
static XLinkError_t _dispatcher_HandleRequest(DispatcherNew* dispatcher, PacketNew* receivedPacket);
static XLinkError_t _dispatcher_HandleResponse(DispatcherNew* dispatcher, PacketNew* receivedPacket);
static XLinkError_t _dispatcher_SendResponse(DispatcherNew* dispatcher, PacketNew* receivedPacket);

static XLinkError_t _dispatcher_StartThread(DispatcherNew* dispatcher, void* (*start_routine) (void*),
                                     pthread_t* newThread, const char* threadName);

static void* _dispatcher_SendPacketsThr(void* arg);
static void* _dispatcher_ReceivePacketsThr(void* arg);

static xLinkEventType_t _dispatcher_GetResponseType(xLinkEventType_t requestType);
static char* _dispatcher_TypeToStr(int type);

// ------------------------------------
// Private methods declaration. End.
// ------------------------------------

// ------------------------------------
// API methods implementation. Begin.
// ------------------------------------

DispatcherNew* Dispatcher_Create(StreamDispatcher* streamDispatcher,
                                 BlockingQueue* packetsToSendQueue,
                                 BlockingQueue* receivedPacketsQueue[MAX_STREAMS_NEW]) {
    DispatcherNew* ret_dispatcher = NULL;

    XLINK_RET_WITH_ERR_IF(receivedPacketsQueue == NULL, NULL);
    XLINK_RET_WITH_ERR_IF(packetsToSendQueue == NULL, NULL);

    DispatcherNew* dispatcher = malloc(sizeof(DispatcherNew));
    if (dispatcher == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate BlockingQueue\n");
        return ret_dispatcher;
    }

    dispatcher->deviceHandle = NULL;
    dispatcher->streamDispatcher = streamDispatcher;
    dispatcher->packetsToSendQueue = packetsToSendQueue;
    dispatcher->status = DISPATCHER_INITIALIZED;

    memcpy(dispatcher->receivedPacketsQueue,
        receivedPacketsQueue, MAX_STREAMS_NEW * sizeof(BlockingQueue*));

    ret_dispatcher = dispatcher;
    XLINK_OUT:
    if(ret_dispatcher == NULL
       && dispatcher != NULL) {
        Dispatcher_Destroy(dispatcher);
    }
    return ret_dispatcher;
}

void Dispatcher_Destroy(DispatcherNew* dispatcher) {
    if(dispatcher == NULL) {
        return;
    }

    free(dispatcher);

    return;
}

XLinkError_t Dispatcher_Start(DispatcherNew* dispatcher, xLinkDeviceHandle_t* deviceHandle) {
    ASSERT_XLINK(dispatcher);
#ifdef __PC__
    ASSERT_XLINK(deviceHandle);
#endif

    dispatcher->deviceHandle = deviceHandle;
    dispatcher->status = DISPATCHER_UP;

    char packetSenderThrName[MVLOG_MAXIMUM_THREAD_NAME_SIZE];
#ifdef __PC__
    snprintf(packetSenderThrName,
        sizeof(packetSenderThrName), "Sender%dThr", 1); //TODO id
#else
    snprintf(packetSenderThrName,
        sizeof(packetSenderThrName), "DevicePacketSenderThr");
#endif

    XLINK_RET_IF(_dispatcher_StartThread(dispatcher,  _dispatcher_SendPacketsThr,
        &dispatcher->sendPacketsThread, packetSenderThrName));

    char packetReceiverThrName[MVLOG_MAXIMUM_THREAD_NAME_SIZE];
#ifdef __PC__
    snprintf(packetReceiverThrName,
             sizeof(packetReceiverThrName), "Receiver%dThr", 1);//TODO id
#else
    snprintf(packetReceiverThrName,
             sizeof(packetReceiverThrName), "DevicePacketReceiverThr");
#endif
    XLINK_RET_IF(_dispatcher_StartThread(dispatcher, _dispatcher_ReceivePacketsThr,
        &dispatcher->receivePacketsThread, packetReceiverThrName));

    return X_LINK_SUCCESS;
}

XLinkError_t Dispatcher_Stop(DispatcherNew* dispatcher) {
    ASSERT_XLINK(dispatcher);

    dispatcher->status = DISPATCHER_WAITING_TO_CLOSE;

    if(pthread_join(dispatcher->sendPacketsThread, NULL)) {
        mvLog(MVLOG_ERROR, "Waiting for ""_senEventsThread"" thread failed");
    }
    if(pthread_join(dispatcher->receivePacketsThread, NULL)) {
        mvLog(MVLOG_ERROR, "Waiting for ""_senEventsThread"" thread failed");
    }

    dispatcher->status = DISPATCHER_DOWN;

    return X_LINK_SUCCESS;
}

DispatcherStatus_t Dispatcher_GetStatus(DispatcherNew* dispatcher) {
    ASSERT_XLINK(dispatcher);

    return dispatcher->status;
}

// ------------------------------------
// API methods implementation. End.
// ------------------------------------

// ------------------------------------
// Private methods implementation. Begin.
// ------------------------------------

static PacketNew* _dispatcher_ReadPacketData(DispatcherNew* dispatcher) {
    ASSERT_XLINK(dispatcher);

    XLinkError_t rc = X_LINK_SUCCESS;
    int platformRc = 0;
    PacketNew* out_packet = NULL;
    XLINK_ALIGN_TO_BOUNDARY(64) PacketHeader header = {0};
    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;

    platformRc = XLinkPlatformRead(dispatcher->deviceHandle,
                               &header, sizeof(header));

    XLINK_RET_WITH_ERR_AND_LOG_IF(platformRc, NULL,
        mvLog(MVLOG_ERROR,"Read packet header failed %d\n", platformRc));

    mvLog(MVLOG_DEBUG, "Read new packet. Packet: %s, id=%d, size=%u, streamId=%u, streamName=%s.\n",
          _dispatcher_TypeToStr(header.type), header.id, header.size, header.streamId, header.streamName);

    // TODO For support old XLink dispatcher
    if(header.type == XLINK_PING_RESP || header.type == XLINK_RESET_RESP
    || header.type == XLINK_PING_REQ || header.type == XLINK_RESET_REQ) {
        header.streamId = CONTROL_SRTEAM_ID;
    }

    if(header.type == XLINK_CREATE_STREAM_REQ) {
        Stream* stream = StreamDispatcher_OpenStream(streamDispatcher, header.streamName);
        header.streamId = Stream_GetId(stream);
    }

    out_packet = StreamDispatcher_GetPacket(streamDispatcher, header.streamId, IN_CHANNEL);
    XLINK_RET_WITH_ERR_IF(out_packet == NULL, NULL);

    out_packet->header = header;

    if(out_packet->header.type == XLINK_WRITE_REQ) {
        XLINK_OUT_RC_IF(Packet_AllocateData(out_packet));

        platformRc = XLinkPlatformRead(
            dispatcher->deviceHandle, out_packet->data, out_packet->header.size);
        XLINK_OUT_RC_WITH_ERR_AND_LOG_IF(platformRc, X_LINK_ERROR,
            mvLog(MVLOG_ERROR,"Read packet data failed %d\n", platformRc));
    }

XLINK_OUT:
    if(rc != X_LINK_SUCCESS) {
        Packet_ReleaseData(out_packet);
    }
    return out_packet;
}

static XLinkError_t _dispatcher_WritePacketData(DispatcherNew* dispatcher, PacketNew* packet) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(packet);

    // TODO For support old XLink dispatcher
    if(packet->header.type == XLINK_CREATE_STREAM_REQ) {
        packet->header.streamId = INVALID_STREAM_ID;
    }

    mvLog(MVLOG_DEBUG, "Write new packet. Packet: %s, id=%d, size=%u, streamId=%u, streamName=%s.\n",
          _dispatcher_TypeToStr(packet->header.type), packet->header.id,
          packet->header.size, packet->header.streamId, packet->header.streamName);

    XLinkError_t rc = X_LINK_SUCCESS;
    XLINK_RET_WITH_ERR_IF(XLinkPlatformWrite(dispatcher->deviceHandle,
                                &packet->header, sizeof(packet->header)), X_LINK_ERROR);

    if (packet->header.type == XLINK_WRITE_REQ) {
        XLINK_RET_WITH_ERR_IF(XLinkPlatformWrite(dispatcher->deviceHandle,
                                packet->data, packet->header.size), X_LINK_ERROR);
    }

    // TODO For support old XLink dispatcher
    if(packet->header.type == XLINK_CREATE_STREAM_REQ) {
        Stream* stream = StreamDispatcher_GetStreamByName(dispatcher->streamDispatcher, packet->header.streamName);
        packet->header.streamId = Stream_GetId(stream);
    }

    return rc;
}

static XLinkError_t _dispatcher_HandleReadPacketError(DispatcherNew* dispatcher) {
    ASSERT_XLINK(dispatcher);

    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;
    ASSERT_XLINK(streamDispatcher);
    ASSERT_RC_XLINK(StreamDispatcher_Lock(streamDispatcher));

    XLinkError_t rc = X_LINK_ERROR;
    int openedStreamIds[MAX_STREAMS_NEW] = {0};
    int count = 0;
    XLINK_OUT_IF(StreamDispatcher_GetOpenedStreamIds(streamDispatcher, openedStreamIds, &count));

    PacketNew* errorPacket = NULL;
    for (int i = 0; i < count; ++i) {

        streamId_t streamId = openedStreamIds[i];
        errorPacket = StreamDispatcher_GetPacket(streamDispatcher, streamId, IN_CHANNEL);
        XLINK_OUT_IF(errorPacket == NULL);

        errorPacket->privateFields.status = PACKET_DROPED;
        XLINK_OUT_IF(BlockingQueue_Push(dispatcher->receivedPacketsQueue[streamId], errorPacket));

        XLINK_OUT_IF(StreamDispatcher_FreePendingPackets(streamDispatcher, streamId, PACKET_DROPED));
    }

    rc = X_LINK_SUCCESS;
    XLINK_OUT:
    ASSERT_RC_XLINK(StreamDispatcher_Unlock(streamDispatcher));
    return rc;
}

static XLinkError_t _dispatcher_HandleRequest(DispatcherNew* dispatcher, PacketNew* receivedPacket) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(receivedPacket);

    xLinkEventType_t reqType = receivedPacket->header.type;

    mvLog(MVLOG_DEBUG, "Handle request: %s\n", _dispatcher_TypeToStr(receivedPacket->header.type));

    switch (reqType) {
        case XLINK_WRITE_REQ:
        {
            BlockingQueue* streamReceivedPacketsQueue =
                dispatcher->receivedPacketsQueue[receivedPacket->header.streamId];
            mvLog(MVLOG_DEBUG, "Push packet to streamReceivedPacketsQueue: id=%d, idx=%d\n",
                  receivedPacket->header.id, receivedPacket->privateFields.idx);
            BlockingQueue_Push(streamReceivedPacketsQueue, receivedPacket);

            return _dispatcher_SendResponse(dispatcher, receivedPacket);
        }
        case XLINK_RESET_REQ:
        {
            dispatcher->status = DISPATCHER_NEED_TO_CLOOSE;
            break;
        }
    }

    if(_dispatcher_SendResponse(dispatcher, receivedPacket)) {
        mvLog(MVLOG_ERROR, "Failed to send response for request: %s", _dispatcher_TypeToStr(receivedPacket->header.type));
    }

    return Packet_Release(receivedPacket);
}

static XLinkError_t _dispatcher_HandleResponse(DispatcherNew* dispatcher, PacketNew* receivedPacket) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(receivedPacket);

    mvLog(MVLOG_DEBUG, "Handle response: %s\n", _dispatcher_TypeToStr(receivedPacket->header.type));

    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;
    PacketNew* pendingPacket = StreamDispatcher_FindPendingPacket(streamDispatcher,
        receivedPacket->header.streamId, receivedPacket->header.id);

    if(pendingPacket != NULL) {
        ASSERT_RC_XLINK(Packet_FreePending(pendingPacket, PACKET_COMPLETED));
    } else {
        mvLog(MVLOG_DEBUG, "Just release packet packet\n");
    }

    return Packet_Release(receivedPacket);
}

static XLinkError_t _dispatcher_SendResponse(DispatcherNew* dispatcher, PacketNew* receivedPacket) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(receivedPacket);

    BlockingQueue* packetsToSendQueue = dispatcher->packetsToSendQueue;
    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;
    PacketNew* respPacket = StreamDispatcher_GetPacket(streamDispatcher,
        receivedPacket->header.streamId, OUT_CHANNEL);

    ASSERT_XLINK(respPacket);
    respPacket->header.id = receivedPacket->header.id;
    respPacket->header.size = receivedPacket->header.size;
    respPacket->header.flags = receivedPacket->header.flags;
    respPacket->header.type = _dispatcher_GetResponseType(receivedPacket->header.type);

    mvLog(MVLOG_DEBUG, "Push packet to packetsToSendQueue: id=%d, idx=%d\n",
          receivedPacket->header.id, receivedPacket->privateFields.idx);
    return BlockingQueue_Push(packetsToSendQueue, respPacket);
}

static XLinkError_t _dispatcher_StartThread(DispatcherNew* dispatcher, void* (*start_routine) (void*),
                                     pthread_t* newThread,  const char* threadName) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(newThread);
    ASSERT_XLINK(threadName);

    XLinkError_t rc = X_LINK_SUCCESS;
    pthread_attr_t attr;
    XLINK_RET_IF(pthread_attr_init(&attr));

    mvLog(MVLOG_DEBUG,"Starting a new thread. %s\n", threadName);
    XLINK_OUT_RC_IF(pthread_create(newThread, &attr,
                                   start_routine, (void*)dispatcher));

#ifndef __APPLE__
    XLINK_OUT_RC_IF(pthread_setname_np(*newThread, threadName));
#endif

XLINK_OUT:
    ASSERT_RC_XLINK(pthread_attr_destroy(&attr));
    return rc;
}

static void* _dispatcher_SendPacketsThr(void* arg) {
    DispatcherNew* dispatcher = (DispatcherNew*) arg;
    BlockingQueue* packetsToSendQueue = dispatcher->packetsToSendQueue;
    packetCommType_t commType = PACKET_REQUEST;
    XLinkError_t isPacketSent = X_LINK_SUCCESS;

    while(dispatcher->status == DISPATCHER_UP) {
        PacketNew* packet = NULL;
        if(BlockingQueue_TimedPop(packetsToSendQueue, (void**)&packet, 500)) {
            ASSERT_XLINK(packet);
        } else {
            continue;
        }

        mvLog(MVLOG_DEBUG, "Pop packet from packetsToSendQueue: id=%d, idx=%d\n",
              packet->header.id, packet->privateFields.idx);

        isPacketSent = _dispatcher_WritePacketData(dispatcher, packet);
        commType = Packet_GetCommType(packet);

        if(commType == PACKET_RESPONSE) {
            Packet_Release(packet);
            continue;
        }

        if(isPacketSent == X_LINK_SUCCESS) {
            packet->privateFields.status = PACKET_PENDING;
        } else {
            mvLog(MVLOG_DEBUG, "Fail to write packet. Packet: %s, id=%d, size=%u, streamId=%u, streamName=%s.\n",
                  _dispatcher_TypeToStr(packet->header.type), packet->header.id,
                  packet->header.size, packet->header.streamId, packet->header.streamName);

            ASSERT_RC_XLINK(Packet_FreePending(packet, PACKET_DROPED));
        }
    }

    return NULL;
}

static void* _dispatcher_ReceivePacketsThr(void* arg) {
    DispatcherNew* dispatcher = (DispatcherNew*) arg;
    packetCommType_t commType;

    while(dispatcher->status == DISPATCHER_UP) {
        PacketNew* packet = _dispatcher_ReadPacketData(dispatcher); // TODO: What if error??

        if(dispatcher->status == DISPATCHER_UP) {
            if(packet == NULL) {
                _dispatcher_HandleReadPacketError(dispatcher);
                continue;
            }
        } else {
            break;
        }

        commType = Packet_GetCommType(packet);
        if(commType == PACKET_REQUEST) {
            _dispatcher_HandleRequest(dispatcher, packet);
        } else {
            _dispatcher_HandleResponse(dispatcher, packet);
        }
    }

    return NULL;
}

static xLinkEventType_t _dispatcher_GetResponseType(xLinkEventType_t requestType) {
    switch(requestType)
    {
        case XLINK_WRITE_REQ:        return XLINK_WRITE_RESP;
        case XLINK_READ_REQ:         return XLINK_READ_RESP;
        case XLINK_READ_REL_REQ:     return XLINK_READ_REL_RESP;
        case XLINK_CREATE_STREAM_REQ:return XLINK_CREATE_STREAM_RESP;
        case XLINK_CLOSE_STREAM_REQ: return XLINK_CLOSE_STREAM_RESP;
        case XLINK_PING_REQ:         return XLINK_PING_RESP;
        case XLINK_RESET_REQ:        return XLINK_RESET_RESP;
        default:
            break;
    }
    return XLINK_RESP_LAST;
}

static char* _dispatcher_TypeToStr(int type) {
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

// ------------------------------------
// Private methods implementation. End.
// ------------------------------------
