// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _GNU_SOURCE
#define _GNU_SOURCE // fix for warning: implicit declaration of function ‘pthread_setname_np’
#endif

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLinkDispatcherNew
#endif

#include "XLinkDispatcherNew.h"
#include "XLinkErrorUtils.h"
#include "XLinkPrivateDefines.h"
#include "XLinkPrivateFields.h"
#include "XLinkPlatform.h"
#include "XLinkStream.h"
#include "XLinkLog.h"

#if (defined(_WIN32) || defined(_WIN64))
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <XLinkMacros.h>

// ------------------------------------
// Private methods declaration. Begin.
// ------------------------------------

static XLinkError_t _dispatcher_ReadPacketData(DispatcherNew* dispatcher, Packet** receivedPacket);
static XLinkError_t _dispatcher_WritePacketData(DispatcherNew* dispatcher, Packet* packet);

static XLinkError_t _dispatcher_HandleReadPacketError(DispatcherNew* dispatcher);
static XLinkError_t _dispatcher_HandleRequest(DispatcherNew* dispatcher, Packet* receivedPacket);
static XLinkError_t _dispatcher_HandleResponse(DispatcherNew* dispatcher, Packet* receivedPacket);
static XLinkError_t _dispatcher_SendResponse(DispatcherNew* dispatcher, Packet* receivedPacket,
                                             Packet** respPacket, XLinkError_t serviceInfo,
                                             packetStatus_t respPacketStatus);

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

XLinkError_t Dispatcher_Create(DispatcherNew* dispatcher,
                               StreamDispatcher* streamDispatcher,
                               BlockingQueue* packetsToSendQueue,
                               BlockingQueue* receivedPacketsQueue) {
    XLINK_RET_IF(streamDispatcher == NULL);
    XLINK_RET_IF(packetsToSendQueue == NULL);
    XLINK_RET_IF(receivedPacketsQueue == NULL);

    if (dispatcher == NULL) {
        mvLog(MVLOG_ERROR, "Cannot allocate Dispatcher");
        return X_LINK_ERROR;
    }

    dispatcher->deviceHandle = NULL;
    dispatcher->streamDispatcher = streamDispatcher;
    dispatcher->packetsToSendQueue = packetsToSendQueue;
    dispatcher->receivedPacketsQueue = receivedPacketsQueue;
    dispatcher->status = DISPATCHER_INITIALIZED;

    return X_LINK_SUCCESS;
}

void Dispatcher_Destroy(DispatcherNew* dispatcher) {
    if (dispatcher == NULL) {
        mvLog(MVLOG_WARN, "Dispatcher was already destroyed");
    }
}

XLinkError_t Dispatcher_Start(DispatcherNew* dispatcher, xLinkDeviceHandle_t* deviceHandle, linkId_t connectionId) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(deviceHandle);
#ifdef __PC__
    ASSERT_XLINK(deviceHandle->xLinkFD);
#endif

    dispatcher->deviceHandle = deviceHandle;
    dispatcher->status = DISPATCHER_UP;

    char packetSenderThrName[MVLOG_MAXIMUM_THREAD_NAME_SIZE];
    snprintf(packetSenderThrName,
             sizeof(packetSenderThrName), "Sender%uThr", connectionId);

    XLINK_RET_IF(_dispatcher_StartThread(dispatcher,  _dispatcher_SendPacketsThr,
        &dispatcher->sendPacketsThread, packetSenderThrName));

    char packetReceiverThrName[MVLOG_MAXIMUM_THREAD_NAME_SIZE];
    snprintf(packetReceiverThrName,
             sizeof(packetReceiverThrName), "Receiver%uThr", connectionId);

    XLINK_RET_IF(_dispatcher_StartThread(dispatcher, _dispatcher_ReceivePacketsThr,
        &dispatcher->receivePacketsThread, packetReceiverThrName));

    return X_LINK_SUCCESS;
}

XLinkError_t Dispatcher_Stop(DispatcherNew* dispatcher) {
    ASSERT_XLINK(dispatcher);

    dispatcher->status = DISPATCHER_WAITING_TO_CLOSE;

    if (pthread_join(dispatcher->sendPacketsThread, NULL)) {
        mvLog(MVLOG_ERROR, "Waiting for ""sendPacketsThread"" thread failed");
    }
    if (pthread_join(dispatcher->receivePacketsThread, NULL)) {
        mvLog(MVLOG_ERROR, "Waiting for ""receivePacketsThread"" thread failed");
    }

    dispatcher->status = DISPATCHER_DOWN;

    return X_LINK_SUCCESS;
}

DispatcherStatus_t Dispatcher_GetStatus(DispatcherNew* dispatcher) {
    ASSERT_XLINK(dispatcher);

    return dispatcher->status;
}

void Dispatcher_SetStatus(DispatcherNew* dispatcher, DispatcherStatus_t status) {
    ASSERT_XLINK(dispatcher);

    dispatcher->status = status;
}

// ------------------------------------
// API methods implementation. End.
// ------------------------------------

// ------------------------------------
// Private methods implementation. Begin.
// ------------------------------------

static XLinkError_t _dispatcher_ReadPacketData(DispatcherNew* dispatcher, Packet** receivedPacket) {
    ASSERT_XLINK(dispatcher);

    XLINK_ALIGN_TO_BOUNDARY(64) PacketHeader header = {0};
    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;

    int platformRc = XLinkPlatformRead(dispatcher->deviceHandle, &header, sizeof(header));
    if (platformRc < 0) {
        mvLog(MVLOG_DEBUG, "Read packet header failed %d", platformRc);
        return X_LINK_COMMUNICATION_FAIL;
    }

    mvLog(MVLOG_INFO, "Read new packet. Packet: %s, id=%d, size=%u, streamId=%u, streamName=%s, serviceInfo %d.",
          _dispatcher_TypeToStr(header.type), header.id, header.size, header.streamId,
          header.streamName, header.serviceInfo);

    (*receivedPacket) = StreamDispatcher_GetPacket(streamDispatcher, header.streamId, IN_CHANNEL);
    XLINK_RET_IF((*receivedPacket) == NULL);
    (*receivedPacket)->header = header;

    if ((*receivedPacket)->header.type == XLINK_WRITE_REQ) {
        if (Packet_AllocateData(*receivedPacket) != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Cannot allocate packet data");
            Packet_Release((*receivedPacket));
            return X_LINK_ERROR;
        }

        platformRc = XLinkPlatformRead(
            dispatcher->deviceHandle, (*receivedPacket)->data, (*receivedPacket)->header.size);
        if (platformRc < 0) {
            mvLog(MVLOG_ERROR,"Read packet data failed %d", platformRc);
            Packet_Release((*receivedPacket));
            return X_LINK_COMMUNICATION_FAIL;
        }
    }

    return X_LINK_SUCCESS;
}

static XLinkError_t _dispatcher_WritePacketData(DispatcherNew* dispatcher, Packet* packet) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(packet);

    mvLog(MVLOG_INFO, "Write new packet. Packet: %s, id=%d, size=%u, streamId=%u, streamName=%s, serviceInfo %d.",
          _dispatcher_TypeToStr(packet->header.type), packet->header.id, packet->header.size,
          packet->header.streamId, packet->header.streamName, packet->header.serviceInfo);

    int platformRc = XLinkPlatformWrite(dispatcher->deviceHandle,
                                        &packet->header, sizeof(packet->header));
    if (platformRc < 0) {
        mvLog(MVLOG_DEBUG, "Write packet header failed %d", platformRc);
        return X_LINK_COMMUNICATION_FAIL;
    }

    if (packet->header.type == XLINK_WRITE_REQ) {
        platformRc = XLinkPlatformWrite(
                dispatcher->deviceHandle, packet->data, packet->header.size);
        if (platformRc < 0) {
            mvLog(MVLOG_ERROR, "Write packet data failed %d", platformRc);
            return X_LINK_COMMUNICATION_FAIL;
        }
    }

    return X_LINK_SUCCESS;
}

static XLinkError_t _dispatcher_HandleReadPacketError(DispatcherNew* dispatcher) {
    ASSERT_XLINK(dispatcher);

    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;
    ASSERT_XLINK(streamDispatcher);
    ASSERT_XLINK(!StreamDispatcher_Lock(streamDispatcher));

    XLinkError_t rc = X_LINK_ERROR;
    int openedStreamIds[XLINK_MAX_STREAMS] = {0};
    int count = 0;
    XLINK_OUT_IF(StreamDispatcher_GetOpenedStreamIds(streamDispatcher, openedStreamIds, &count));

    Packet* errorPacket = NULL;
    for (int i = 0; i < count; ++i) {
        streamId_t streamId = openedStreamIds[i];

        int pendingToPop = dispatcher->receivedPacketsQueue[streamId].pendingToPop;
        for (int pendingToPopIdx = 0; pendingToPopIdx < pendingToPop; ++pendingToPopIdx) {
            errorPacket = StreamDispatcher_GetPacket(streamDispatcher, streamId, IN_CHANNEL);
            XLINK_OUT_IF(errorPacket == NULL);

            Packet_SetPacketStatus(errorPacket, PACKET_DROPPED);
            XLINK_OUT_IF(BlockingQueue_Push(&dispatcher->receivedPacketsQueue[streamId], errorPacket));
        }

        XLINK_OUT_IF(StreamDispatcher_FreePendingPackets(streamDispatcher, streamId, PACKET_DROPPED));
    }

    rc = X_LINK_SUCCESS;
    XLINK_OUT:
    ASSERT_XLINK(!StreamDispatcher_Unlock(streamDispatcher));
    return rc;
}

static XLinkError_t _dispatcher_HandleRequest(DispatcherNew* dispatcher, Packet* receivedPacket) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(receivedPacket);

    xLinkEventType_t reqType = receivedPacket->header.type;
    streamId_t streamId = receivedPacket->header.streamId;
    packetStatus_t respPacketStatus = PACKET_PROCESSING;
    int32_t serviceInfo = 0;
    Packet* respPacket = NULL;

    mvLog(MVLOG_DEBUG, "Handle request: %s", _dispatcher_TypeToStr(reqType));

    // Preamble
    switch (reqType) {
        case XLINK_WRITE_REQ: {
            BlockingQueue* streamReceivedPacketsQueue =
                &dispatcher->receivedPacketsQueue[streamId];
            BlockingQueue_Push(streamReceivedPacketsQueue, receivedPacket);

            break;
        }
        case XLINK_PING_REQ: {
            sem_post(&pingSem);
            break;
        }
        case XLINK_CREATE_STREAM_REQ: {
            if (XLink_isOnHostSide()) {
                Stream* stream = StreamDispatcher_OpenStream(
                        dispatcher->streamDispatcher, receivedPacket->header.streamName);
                XLINK_RET_IF(stream == NULL);
                serviceInfo = (int32_t)Stream_GetId(stream);
            } else {
                Stream* stream = StreamDispatcher_OpenStreamById(
                        dispatcher->streamDispatcher, receivedPacket->header.streamName,
                        (streamId_t)receivedPacket->header.serviceInfo);
                XLINK_RET_IF(stream == NULL);

                void *buffer = XLinkPlatformAllocateData(ALIGN_UP(receivedPacket->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
                if (buffer == NULL) {
                    mvLog(MVLOG_ERROR,"Cannot create stream. Requested memory = %u", receivedPacket->header.size);
                    serviceInfo = X_LINK_OUT_OF_MEMORY;
                    StreamDispatcher_CloseStream(dispatcher->streamDispatcher, stream->streamId);
                } else {
                    XLinkPlatformDeallocateData(buffer, ALIGN_UP(receivedPacket->header.size, __CACHE_LINE_SIZE), __CACHE_LINE_SIZE);
                }
            }
            break;
        }
        case XLINK_CLOSE_STREAM_REQ:
        case XLINK_RESET_REQ: {
            respPacketStatus = PACKET_PENDING_TO_SEND;
            break;
        }
        case XLINK_READ_REQ:
        case XLINK_READ_REL_REQ:
            break;

        default:
            mvLog(MVLOG_ERROR, "Invalid event request type %d", reqType);
            return X_LINK_ERROR;
    }

    if (_dispatcher_SendResponse(dispatcher, receivedPacket, &respPacket, serviceInfo, respPacketStatus)) {
        mvLog(MVLOG_ERROR, "Failed to send response for request: %s", _dispatcher_TypeToStr(receivedPacket->header.type));
    }

    if (respPacketStatus == PACKET_PENDING_TO_SEND) {
        Packet_WaitPacketComplete(respPacket);
        ASSERT_XLINK(!Packet_Release(respPacket));
    }

    // Postamble
    switch (reqType) {
        case XLINK_PING_REQ:
        case XLINK_CREATE_STREAM_REQ:
        case XLINK_READ_REQ:
        case XLINK_READ_REL_REQ:
            break;
        case XLINK_WRITE_REQ: {
            return X_LINK_SUCCESS;
        }
        case XLINK_RESET_REQ: {
            dispatcher->status = DISPATCHER_NEED_TO_CLOSE;
            XLinkPlatformCloseRemote(dispatcher->deviceHandle);
            break;
        }
        case XLINK_CLOSE_STREAM_REQ: {
            streamId_t closingStreamId = (streamId_t)receivedPacket->header.serviceInfo;
            ASSERT_XLINK(!Packet_Release(receivedPacket));
            return StreamDispatcher_CloseStream(dispatcher->streamDispatcher, closingStreamId);
        }
        default:
            mvLog(MVLOG_ERROR, "Invalid event request type %d", reqType);
            Packet_Release(receivedPacket);
            return X_LINK_ERROR;
    }

    return Packet_Release(receivedPacket);
}

static XLinkError_t _dispatcher_HandleResponse(DispatcherNew* dispatcher, Packet* receivedPacket) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(receivedPacket);

    mvLog(MVLOG_DEBUG, "Handle response: %s", _dispatcher_TypeToStr(receivedPacket->header.type));

    xLinkEventType_t reqType = receivedPacket->header.type;
    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;

    Packet* pendingPacket = StreamDispatcher_FindPendingPacket(
            streamDispatcher, receivedPacket->header.streamId, receivedPacket);

    if (reqType == XLINK_RESET_RESP) {
        dispatcher->status = DISPATCHER_NEED_TO_CLOSE;
    }
    if (reqType == XLINK_PING_RESP && dispatcher->status == DISPATCHER_NEED_TO_CLOSE) {
        dispatcher->status = DISPATCHER_WAITING_TO_CLOSE;
    }

    if (pendingPacket != NULL) {
        pendingPacket->header.serviceInfo = receivedPacket->header.serviceInfo;
        ASSERT_XLINK(!Packet_FreePending(pendingPacket, PACKET_COMPLETED));
    } else {
        mvLog(MVLOG_DEBUG, "Just release packet: streamId=%u, packet type %s",
              receivedPacket->header.streamId, _dispatcher_TypeToStr(reqType));
    }

    return Packet_Release(receivedPacket);
}

static XLinkError_t _dispatcher_SendResponse(DispatcherNew* dispatcher, Packet* receivedPacket,
                                             Packet** respPacket, XLinkError_t serviceInfo,
                                             packetStatus_t respPacketStatus) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(receivedPacket);

    BlockingQueue* packetsToSendQueue = dispatcher->packetsToSendQueue;
    StreamDispatcher* streamDispatcher = dispatcher->streamDispatcher;

    *respPacket = StreamDispatcher_GetPacket(streamDispatcher,
        receivedPacket->header.streamId, OUT_CHANNEL);

    ASSERT_XLINK(*respPacket);
    (*respPacket)->header.id = receivedPacket->header.id;
    (*respPacket)->header.size = receivedPacket->header.size;
    (*respPacket)->header.serviceInfo = serviceInfo;
    (*respPacket)->header.type = _dispatcher_GetResponseType(receivedPacket->header.type);
    (*respPacket)->privateFields.status = respPacketStatus;
    mvLog(MVLOG_DEBUG, "Push packet to packetsToSendQueue: id=%d, idx=%d %s",
          receivedPacket->header.id, receivedPacket->privateFields.idx, _dispatcher_TypeToStr((*respPacket)->header.type));
    return BlockingQueue_Push(packetsToSendQueue, (*respPacket));
}

static XLinkError_t _dispatcher_StartThread(DispatcherNew* dispatcher, void* (*start_routine) (void*),
                                            pthread_t* newThread,  const char* threadName) {
    ASSERT_XLINK(dispatcher);
    ASSERT_XLINK(newThread);
    ASSERT_XLINK(threadName);

    XLinkError_t rc = X_LINK_SUCCESS;
    pthread_attr_t attr;
    XLINK_RET_IF(pthread_attr_init(&attr));
#ifndef __PC__
    if (pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_setinheritsched error");
        pthread_attr_destroy(&attr);
    }
    if (pthread_attr_setschedpolicy(&attr, SCHED_RR) != 0) {
        mvLog(MVLOG_ERROR,"pthread_attr_setschedpolicy error");
        pthread_attr_destroy(&attr);
    }
#endif

    mvLog(MVLOG_DEBUG,"Starting a new thread. %s", threadName);
    XLINK_OUT_IF(pthread_create(newThread, &attr,
                                   start_routine, (void*)dispatcher));

#ifndef __APPLE__
    XLINK_OUT_IF(pthread_setname_np(*newThread, threadName));
#endif

XLINK_OUT:
    ASSERT_XLINK(!pthread_attr_destroy(&attr));
    return rc;
}

static void* _dispatcher_SendPacketsThr(void* arg) {
    DispatcherNew* dispatcher = (DispatcherNew*) arg;
    BlockingQueue* packetsToSendQueue = dispatcher->packetsToSendQueue;

    while (dispatcher->status == DISPATCHER_UP) {
        Packet* packet = NULL;
        XLinkError_t rc = BlockingQueue_TimedPop(packetsToSendQueue, (void**)&packet, 100);
        if (rc) {
            if (rc == X_LINK_TIMEOUT) {
                continue;
            } else {
                mvLog(MVLOG_ERROR, "BlockingQueue_TimedPop returned %d, stopping...", rc);
                break;
            }
        }

        ASSERT_XLINK(packet);
        mvLog(MVLOG_DEBUG, "Pop packet from packetsToSendQueue: id=%d, idx=%d, streamName=%s",
              packet->header.id, packet->privateFields.idx, packet->header.streamName);

        XLinkError_t isPacketSent = _dispatcher_WritePacketData(dispatcher, packet);

        packetStatus_t status;
        Packet_GetPacketStatus(packet, &status);

        if (isPacketSent != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Failed to write packet. Packet: %s, id=%d, size=%u, streamId=%u, streamName=%s.",
                  _dispatcher_TypeToStr(packet->header.type), packet->header.id,
                  packet->header.size, packet->header.streamId, packet->header.streamName);


            if (status == PACKET_PENDING_TO_SEND || status == PACKET_PENDING_RESPONSE) {
                ASSERT_XLINK(!Packet_FreePending(packet, PACKET_DROPPED));
            } else if (status == PACKET_PROCESSING) {
                ASSERT_XLINK(!Packet_Release(packet));
            }
            continue;
        }

        if (status == PACKET_PENDING_TO_SEND) {
            ASSERT_XLINK(!Packet_FreePending(packet, PACKET_COMPLETED));
        } else if (status == PACKET_PROCESSING) {
            ASSERT_XLINK(!Packet_Release(packet));
        }
    }

    return NULL;
}

static void* _dispatcher_ReceivePacketsThr(void* arg) {
    DispatcherNew* dispatcher = (DispatcherNew*) arg;
    packetCommType_t commType;

    while (dispatcher->status == DISPATCHER_UP) {
        Packet* receivedPacket = NULL;
        XLinkError_t rc = _dispatcher_ReadPacketData(dispatcher, &receivedPacket);

        if (rc != X_LINK_SUCCESS) {
            _dispatcher_HandleReadPacketError(dispatcher);
            continue;
        }

        commType = Packet_GetCommType(receivedPacket);
        if (commType == PACKET_REQUEST) {
            _dispatcher_HandleRequest(dispatcher, receivedPacket);
        } else {
            _dispatcher_HandleResponse(dispatcher, receivedPacket);
        }
    }

    return NULL;
}

static xLinkEventType_t _dispatcher_GetResponseType(xLinkEventType_t requestType) {
    switch(requestType) {
        case XLINK_WRITE_REQ:          return XLINK_WRITE_RESP;
        case XLINK_READ_REQ:           return XLINK_READ_RESP;
        case XLINK_READ_REL_REQ:       return XLINK_READ_REL_RESP;
        case XLINK_CREATE_STREAM_REQ:  return XLINK_CREATE_STREAM_RESP;
        case XLINK_CLOSE_STREAM_REQ:   return XLINK_CLOSE_STREAM_RESP;
        case XLINK_PING_REQ:           return XLINK_PING_RESP;
        case XLINK_RESET_REQ:          return XLINK_RESET_RESP;
        default:
            break;
    }
    return XLINK_RESP_LAST;
}

static char* _dispatcher_TypeToStr(int type) {
    switch(type) {
        case XLINK_WRITE_REQ:          return "XLINK_WRITE_REQ";
        case XLINK_READ_REQ:           return "XLINK_READ_REQ";
        case XLINK_READ_REL_REQ:       return "XLINK_READ_REL_REQ";
        case XLINK_CREATE_STREAM_REQ:  return "XLINK_CREATE_STREAM_REQ";
        case XLINK_CLOSE_STREAM_REQ:   return "XLINK_CLOSE_STREAM_REQ";
        case XLINK_PING_REQ:           return "XLINK_PING_REQ";
        case XLINK_RESET_REQ:          return "XLINK_RESET_REQ";
        case XLINK_REQUEST_LAST:       return "XLINK_REQUEST_LAST";
        case XLINK_WRITE_RESP:         return "XLINK_WRITE_RESP";
        case XLINK_READ_RESP:          return "XLINK_READ_RESP";
        case XLINK_READ_REL_RESP:      return "XLINK_READ_REL_RESP";
        case XLINK_CREATE_STREAM_RESP: return "XLINK_CREATE_STREAM_RESP";
        case XLINK_CLOSE_STREAM_RESP:  return "XLINK_CLOSE_STREAM_RESP";
        case XLINK_PING_RESP:          return "XLINK_PING_RESP";
        case XLINK_RESET_RESP:         return "XLINK_RESET_RESP";
        case XLINK_RESP_LAST:          return "XLINK_RESP_LAST";
        default:
            break;
    }
    return "";
}

// ------------------------------------
// Private methods implementation. End.
// ------------------------------------
