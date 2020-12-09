// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdlib.h>
#include <string.h>

#if (defined(_WIN32) || defined(_WIN64))
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif


#include "XLinkConnection.h"
#include "XLinkPrivateDefines.h"
#include "XLinkDispatcherNew.h"
#include "XLinkBlockingQueue.h"
#include "XLinkTool.h"
#include "XLinkPlatform.h"
#include "XLinkStringUtils.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLinkConnection
#endif
#include "XLinkLog.h"

#define MAX_PATH_LENGTH (255)



// ------------------------------------
// Private methods declaration. Begin.
// ------------------------------------

static XLinkError_t _connection_SendPacket(Connection *connection, streamId_t streamId, xLinkEventType_t type,
                                           const uint8_t *buffer, int size);

// ------------------------------------
// Private methods declaration. End.
// ------------------------------------



// ------------------------------------
// API methods implementation. Begin.
// ------------------------------------

XLinkError_t Connection_Init(Connection* connection, linkId_t id) {
    XLINK_RET_IF(connection == NULL);

    XLinkError_t rc = X_LINK_ERROR;
    memset(connection, 0, sizeof(Connection));

    connection->id = id;
    connection->status = CONNECTION_INITIALIZED;
    connection->streamDispatcher = StreamDispatcher_Create();
    XLINK_OUT_IF(connection->streamDispatcher == NULL);

    connection->packetsToSendQueue = BlockingQueue_Create("packetsToSendQueue");
    XLINK_OUT_IF(connection->packetsToSendQueue == NULL);

    char name[MAX_QUEUE_NAME_LENGHT] = {0};
    for (int i = 0; i < MAX_STREAMS_NEW; ++i) {
        snprintf(name, sizeof(name), "receivedPacketsQueue[%d]", i);
        connection->receivedPacketsQueue[i] = BlockingQueue_Create(name);
        XLINK_OUT_IF(connection->receivedPacketsQueue[i] == NULL);

        snprintf(name, sizeof(name), "userPacketQueue[%d]", i);
        connection->userPacketQueue[i] = BlockingQueue_Create(name);
        XLINK_OUT_IF(connection->userPacketQueue[i] == NULL);
    }

    connection->dispatcher = Dispatcher_Create(connection->streamDispatcher,
        connection->packetsToSendQueue, connection->receivedPacketsQueue);
    XLINK_OUT_IF(connection->dispatcher == NULL);

    XLINK_OUT_IF(StreamDispatcher_OpenStreamById(
        connection->streamDispatcher, "controlStream", CONTROL_SRTEAM_ID) == NULL);

    rc = X_LINK_SUCCESS;
    XLINK_OUT:
    if(rc != X_LINK_SUCCESS) {
        Connection_Clean(connection);
    }
    return rc;
}

XLinkError_t Connection_Clean(Connection* connection) {
    XLINK_RET_IF(connection == NULL);

    StreamDispatcher_Destroy(connection->streamDispatcher);
    BlockingQueue_Destroy(connection->packetsToSendQueue);

    for (int i = 0; i < MAX_STREAMS_NEW; ++i) {
        BlockingQueue_Destroy(connection->receivedPacketsQueue[i]);
        BlockingQueue_Destroy(connection->userPacketQueue[i]);
    }

    Dispatcher_Destroy(connection->dispatcher);

    return X_LINK_SUCCESS;
}

XLinkError_t Connection_Connect(Connection* connection, XLinkHandler_t* handler) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(handler == NULL);

    if (strnlen(handler->devicePath, MAX_PATH_LENGTH) < 2) {
        mvLog(MVLOG_ERROR, "Device path is incorrect");
        return X_LINK_ERROR;
    }

    connection->deviceHandle.protocol = handler->protocol;
    XLINK_RET_IF(XLinkPlatformConnect(handler->devicePath2, handler->devicePath,
        connection->deviceHandle.protocol, &connection->deviceHandle.xLinkFD));

    XLINK_RET_IF(Dispatcher_Start(connection->dispatcher,
        &connection->deviceHandle));

    XLinkError_t isCompleted = _connection_SendPacket(connection,
        CONTROL_SRTEAM_ID, XLINK_PING_REQ, NULL, 0);

    connection->status = isCompleted == X_LINK_SUCCESS ? CONNECTION_UP : CONNECTION_NEED_TO_CLOOSE;

    return isCompleted;
}

XLinkError_t Connection_Reset(Connection* connection) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(connection->status != CONNECTION_UP);

    connection->status = CONNECTION_WAITING_TO_CLOSE;
    XLinkError_t isCompleted = _connection_SendPacket(
        connection, CONTROL_SRTEAM_ID, XLINK_RESET_REQ, NULL, 0);

    XLINK_RET_IF(Dispatcher_Stop(connection->dispatcher));
    connection->status = CONNECTION_DOWN;

    return isCompleted;
}

streamId_t Connection_OpenStream(Connection* connection, const char* name, int stream_write_size) {
    XLINK_RET_WITH_ERR_IF(connection == NULL, INVALID_STREAM_ID);
    XLINK_RET_WITH_ERR_IF(name == NULL, INVALID_STREAM_ID);
    XLINK_RET_WITH_ERR_IF(stream_write_size < 0, INVALID_STREAM_ID);

    streamId_t  ret_id = INVALID_STREAM_ID;
    Stream* stream = StreamDispatcher_OpenStream(connection->streamDispatcher, name);
    XLINK_OUT_IF(stream == NULL);
    XLinkError_t isCompleted = _connection_SendPacket(connection,
        Stream_GetId(stream), XLINK_CREATE_STREAM_REQ, NULL, stream_write_size);
    XLINK_OUT_IF(isCompleted != X_LINK_SUCCESS);

    ret_id = Stream_GetId(stream);
    XLINK_OUT:
    return ret_id;
}

XLinkError_t Connection_CloseStream(Connection* connection, streamId_t streamId) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > MAX_STREAMS_NEW);

    XLinkError_t isCompleted = _connection_SendPacket(connection, streamId, XLINK_CLOSE_STREAM_REQ, NULL, 0);

    //TODO clear Queues!!!!
    XLINK_RET_IF(StreamDispatcher_CloseStream(connection->streamDispatcher, streamId));

    return isCompleted;
}

XLinkError_t Connection_Write(Connection* connection, streamId_t streamId, const uint8_t* buffer, int size) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > MAX_STREAMS_NEW);
    XLINK_RET_IF(buffer == NULL);
    XLINK_RET_IF(size < 0);

    XLinkError_t isCompleted = _connection_SendPacket(connection, streamId, XLINK_WRITE_REQ, buffer, size);

    return isCompleted;
}

XLinkError_t Connection_Read(Connection* connection, streamId_t streamId, streamPacketDesc_t** packet) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > MAX_STREAMS_NEW);
    XLINK_RET_IF(packet == NULL);

    PacketNew* receivedPacket = NULL;

    XLINK_RET_IF(BlockingQueue_Pop(connection->receivedPacketsQueue[streamId], (void**)&receivedPacket));
    XLINK_RET_IF(receivedPacket == NULL);

    if(receivedPacket->privateFields.status == PACKET_DROPED) {
        *packet = NULL;
        return X_LINK_ERROR;
    }

    receivedPacket->userData.data = receivedPacket->data;
    receivedPacket->userData.length = receivedPacket->header.size;
    *packet = &receivedPacket->userData;

    XLINK_RET_IF(BlockingQueue_Push(connection->userPacketQueue[streamId], receivedPacket));

    return X_LINK_SUCCESS;
}

XLinkError_t Connection_ReleaseData(Connection* connection, streamId_t streamId) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > MAX_STREAMS_NEW);

    PacketNew* userPacket = NULL;
    XLINK_RET_IF(BlockingQueue_Pop(connection->userPacketQueue[streamId], (void**)&userPacket));
    XLINK_RET_IF(userPacket == NULL);

    XLinkError_t isCompleted =  _connection_SendPacket(connection, streamId,
        XLINK_READ_REL_REQ, NULL, userPacket->header.size);

    XLINK_RET_IF(Packet_Release(userPacket));

    return isCompleted;
}

XLinkError_t Connection_GetFillLevel(Connection* connection, streamId_t streamId, int isRemote, int* fillLevel) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > MAX_STREAMS_NEW);

    *fillLevel = 0;

    return X_LINK_SUCCESS;
}

ConnectionStatus_t Connection_GetStatus(Connection* connection) {
    ASSERT_XLINK(connection != NULL);

    return connection->status;
}

linkId_t Connection_GetId(Connection* connection) {
    ASSERT_XLINK(connection != NULL);

    return connection->id;
}

// ------------------------------------
// API methods implementation. End.
// ------------------------------------

// ------------------------------------
// Private methods implementation. Begin.
// ------------------------------------

static XLinkError_t _connection_SendPacket(Connection *connection, streamId_t streamId, xLinkEventType_t type,
    const uint8_t *buffer, int size) {
    ASSERT_XLINK(connection);
    ASSERT_XLINK(streamId < MAX_STREAMS_NEW);

    XLinkError_t isCompleted = X_LINK_ERROR;
    PacketNew* packet = StreamDispatcher_GetPacket(connection->streamDispatcher, streamId, OUT_CHANNEL);
    XLINK_OUT_IF(packet == NULL);

    packet->header.type = type;
    XLINK_OUT_IF(Packet_SetData(packet, (void*)buffer, size));

    mvLog(MVLOG_DEBUG, "Push packet to packetsToSendQueue: id=%d, idx=%d\n",
          packet->header.id, packet->privateFields.idx);
    XLINK_OUT_IF(BlockingQueue_Push(connection->packetsToSendQueue, packet));
    ASSERT_RC_XLINK(sem_wait(&packet->privateFields.completedSem));

    isCompleted = packet->privateFields.status == PACKET_COMPLETED ? X_LINK_SUCCESS : X_LINK_ERROR;

XLINK_OUT:
    Packet_Release(packet);
    return isCompleted;
}

// ------------------------------------
// Private methods implementation. End.
// ------------------------------------
