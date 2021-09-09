// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLinkConnection
#endif

#include "XLinkPlatform.h"
#include "XLinkConnection.h"
#include "XLinkErrorUtils.h"
#include "XLinkPrivateDefines.h"
#include "XLinkDispatcherNew.h"
#include "XLinkBlockingQueue.h"
#include "XLinkLog.h"
#include <XLinkStringUtils.h>

#if (defined(_WIN32) || defined(_WIN64))
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif

#include <string.h>
#include <XLinkPrivateFields.h>

#define MAX_PATH_LENGTH (255)

// ------------------------------------
// Private methods declaration. Begin.
// ------------------------------------

static Packet* _connection_GetPacket(Connection *connection, streamId_t streamId,
                                     xLinkEventType_t type,
                                     const uint8_t *buffer, int size);

static XLinkError_t _connection_SendPacket(Connection *connection, Packet* packet, unsigned int timeoutMs);

static XLinkError_t _connection_ReceivePacket(Connection *connection, streamId_t streamId,
                                              streamPacketDesc_t** packet, unsigned int timeoutMs);

// ------------------------------------
// Private methods declaration. End.
// ------------------------------------


// ------------------------------------
// API methods implementation. Begin.
// ------------------------------------

XLinkError_t Connection_Init(Connection* connection, linkId_t id) {
    XLINK_RET_IF(connection == NULL);

    memset(connection, 0, sizeof(Connection));

    connection->id = id;
    connection->status = XLINK_CONNECTION_INITIALIZED;

    if (StreamDispatcher_Create(&connection->streamDispatcher) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Connection_Init: failed to create Stream Dispatcher");
        Connection_Clean(connection);
        return X_LINK_ERROR;
    }

    if (BlockingQueue_Create(&connection->packetsToSendQueue, "packetsToSendQueue") != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Connection_Init: failed to create Queue of packets to sent");
        Connection_Clean(connection);
        return X_LINK_ERROR;
    }

    char name[MAX_QUEUE_NAME_LENGTH] = {0};
    for (int i = 0; i < XLINK_MAX_STREAMS; ++i) {
        snprintf(name, sizeof(name), "receivedPacketsQueue[%d]", i);
        if (BlockingQueue_Create(&connection->receivedPacketsQueue[i], name) != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Connection_Init: failedgetNextAvailableLink to create Queue of received packets");
            Connection_Clean(connection);
            return X_LINK_ERROR;
        }

        snprintf(name, sizeof(name), "userPacketQueue[%d]", i);
        if (BlockingQueue_Create(&connection->userPacketQueue[i], name) != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Connection_Init: failed to create Queue of user packets");
            Connection_Clean(connection);
            return X_LINK_ERROR;
        }
    }

    if (Dispatcher_Create(&connection->dispatcher, &connection->streamDispatcher, &connection->packetsToSendQueue,
                          connection->receivedPacketsQueue) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Connection_Init: failed to create XLink Dispatcher");
        Connection_Clean(connection);
        return X_LINK_ERROR;
    }

    Stream* controlStream = StreamDispatcher_OpenStreamById(
            &connection->streamDispatcher,
            "controlStream",
            XLINK_CONTROL_STREAM_ID);
    if (controlStream == NULL) {
        mvLog(MVLOG_ERROR, "Connection_Init: failed to create Control stream");
        Connection_Clean(connection);
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t Connection_Clean(Connection* connection) {
    XLINK_RET_IF(connection == NULL);

    mvLog(MVLOG_DEBUG, "start cleaning connection");

    StreamDispatcher_CloseStream(&connection->streamDispatcher, XLINK_CONTROL_STREAM_ID);

    StreamDispatcher_Destroy(&connection->streamDispatcher);
    BlockingQueue_Destroy(&connection->packetsToSendQueue);

    for (int i = 0; i < XLINK_MAX_STREAMS; ++i) {
        BlockingQueue_Destroy(&connection->receivedPacketsQueue[i]);
        BlockingQueue_Destroy(&connection->userPacketQueue[i]);
    }

    Dispatcher_Destroy(&connection->dispatcher);

    return X_LINK_SUCCESS;
}

#ifdef __PC__

XLinkError_t Connection_Connect(Connection* connection, XLinkHandler_t* handler) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(handler == NULL);

    if (strnlen(handler->devicePath, MAX_PATH_LENGTH) < 2) {
        mvLog(MVLOG_ERROR, "Device path is incorrect");
        return X_LINK_ERROR;
    }

    connection->deviceHandle.protocol = handler->protocol;
    int connectStatus = XLinkPlatformConnect(handler->devicePath2, handler->devicePath,
                                             connection->deviceHandle.protocol, &connection->deviceHandle.xLinkFD);
    if (connectStatus < 0) {
        /**
         * Connection may be unsuccessful at some amount of first tries.
         * In this case, asserting the status provides enormous amount of logs in tests.
         */
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    XLINK_RET_IF(Dispatcher_Start(
            &connection->dispatcher, &connection->deviceHandle, connection->id));

    Packet* packet = _connection_GetPacket(
            connection, XLINK_CONTROL_STREAM_ID,
            XLINK_PING_REQ, NULL, 0);
    XLINK_RET_IF(packet == NULL);

    Packet_SetPacketStatus(packet, PACKET_PENDING_RESPONSE);
    XLinkError_t packetSendStatus = _connection_SendPacket(connection, packet, 0);
    if (Packet_Release(packet) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot release packet");
    }

    connection->status = packetSendStatus == X_LINK_SUCCESS
            ? XLINK_CONNECTION_UP
            : XLINK_CONNECTION_NEED_TO_CLOSE;

    return packetSendStatus;
}

XLinkError_t Connection_Reset(Connection* connection) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(connection->status != XLINK_CONNECTION_UP);

    connection->status = XLINK_CONNECTION_WAITING_TO_CLOSE;
//    Dispatcher_SetStatus(connection->dispatcher, DISPATCHER_NEED_TO_CLOSE);

#if defined(NO_BOOT)
    xLinkEventType_t type = XLINK_PING_REQ;
#else
    xLinkEventType_t type = XLINK_RESET_REQ;
#endif
    if (connection->deviceHandle.protocol == X_LINK_PCIE) {
        type = XLINK_PING_REQ;
    }

    Packet* packet = _connection_GetPacket(
            connection, XLINK_CONTROL_STREAM_ID,
            type, NULL, 0);

    XLINK_RET_IF(packet == NULL);
    Packet_SetPacketStatus(packet, PACKET_PENDING_TO_SEND);
    XLinkError_t packetSendStatus = _connection_SendPacket(connection, packet, 0);
    if (packetSendStatus != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Sending reset request failed with error %d. Just closing the connection...", packetSendStatus);
    }
    if (Packet_Release(packet) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot release packet");
    }

    XLINK_RET_IF(Dispatcher_Stop(&connection->dispatcher));
    XLinkPlatformCloseRemote(&connection->deviceHandle);
    connection->status = XLINK_CONNECTION_DOWN;

    return packetSendStatus;
}

#endif  // __PC__

streamId_t Connection_OpenStream(Connection* connection, const char* name, int stream_write_size) {
    XLINK_RET_ERR_IF(connection == NULL, INVALID_STREAM_ID);
    XLINK_RET_ERR_IF(name == NULL, INVALID_STREAM_ID);
    XLINK_RET_ERR_IF(stream_write_size < 0, INVALID_STREAM_ID);

    Stream* stream = NULL;

    Packet* packet = _connection_GetPacket(
            connection, XLINK_CONTROL_STREAM_ID,
            XLINK_CREATE_STREAM_REQ, NULL, stream_write_size);
    XLINK_RET_ERR_IF(packet == NULL, INVALID_STREAM_ID);
    packet->header.serviceInfo = (int32_t)INVALID_STREAM_ID;
    mv_strcpy(packet->header.streamName, MAX_STREAM_NAME_LENGTH, name);

    if (XLink_isOnHostSide()) {
        stream = StreamDispatcher_OpenStream(&connection->streamDispatcher, name);
        if (stream == NULL) {
            mvLog(MVLOG_ERROR, "Cannot open stream with name %s", name);
            Packet_Release(packet);
            return INVALID_STREAM_ID;
        }
        packet->header.serviceInfo = (int32_t)Stream_GetId(stream);
    }

    Packet_SetPacketStatus(packet, PACKET_PENDING_RESPONSE);
    XLinkError_t packetSendStatus = _connection_SendPacket(connection, packet, 0);
    XLINK_RET_ERR_IF(packetSendStatus != X_LINK_SUCCESS, INVALID_STREAM_ID);

    if (XLink_isOnHostSide()) {
        if (packet->header.serviceInfo == X_LINK_OUT_OF_MEMORY) {
            mvLog(MVLOG_ERROR, "Not enough memory on the device to open stream with id %u, name %s, write size %d",
                  stream->streamId, stream->name, packet->header.size);
            StreamDispatcher_CloseStream(&connection->streamDispatcher, stream->streamId);
            Packet_Release(packet);
            return INVALID_STREAM_ID_OUT_OF_MEMORY;
        }
    } else {
        stream = StreamDispatcher_OpenStreamById(&connection->streamDispatcher, name, (streamId_t)packet->header.serviceInfo);
        if (stream == NULL) {
            mvLog(MVLOG_ERROR, "Cannot open stream with name %s by id %u", name, (streamId_t)packet->header.serviceInfo);
            Packet_Release(packet);
            return INVALID_STREAM_ID;
        }
    }

    if (Packet_Release(packet) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot release packet");
    }

    mvLog(MVLOG_DEBUG, "stream with name %s, id %u was opened", name, Stream_GetId(stream));

    return Stream_GetId(stream);
}

XLinkError_t Connection_CloseStream(Connection* connection, streamId_t streamId) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > XLINK_MAX_STREAMS);

    Packet* packet = _connection_GetPacket(
            connection, XLINK_CONTROL_STREAM_ID,
            XLINK_CLOSE_STREAM_REQ, NULL, 0);
    XLINK_RET_IF(packet == NULL);
    packet->header.serviceInfo = (int32_t)streamId;

    Packet_SetPacketStatus(packet, PACKET_PENDING_TO_SEND);
    XLinkError_t packetSendStatus = _connection_SendPacket(connection, packet, 0);
    if (packetSendStatus != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Sending close stream request failed with error %d. Just closing stream...", packetSendStatus);
    }
    if (Packet_Release(packet) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot release packet");
    }

    XLINK_RET_IF(StreamDispatcher_CloseStream(&connection->streamDispatcher, streamId));

    return packetSendStatus;
}

XLinkError_t Connection_Write(Connection* connection, streamId_t streamId, const uint8_t* buffer, int size, unsigned int timeoutMs) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId >= XLINK_MAX_STREAMS);
    XLINK_RET_IF(buffer == NULL);
    XLINK_RET_IF(size < 0);

    Packet* packet = _connection_GetPacket(
            connection, streamId,
            XLINK_WRITE_REQ, buffer, size);
    XLINK_RET_IF(packet == NULL);

    Packet_SetPacketStatus(packet, PACKET_PENDING_TO_SEND);
    XLinkError_t packetSendStatus = _connection_SendPacket(connection, packet, timeoutMs);
    if (packetSendStatus != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Writing failed with error %d.", packetSendStatus);
    }
    if (Packet_Release(packet) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Cannot release packet");
    }

    return packetSendStatus;
}

XLinkError_t Connection_Read(Connection* connection, streamId_t streamId, streamPacketDesc_t** packet, unsigned int timeoutMs) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(connection->status != XLINK_CONNECTION_UP);
    XLINK_RET_IF(streamId > XLINK_MAX_STREAMS);
    XLINK_RET_IF(packet == NULL);

    return _connection_ReceivePacket(connection, streamId, packet, timeoutMs);
}

XLinkError_t Connection_ReleaseData(Connection* connection, streamId_t streamId) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > XLINK_MAX_STREAMS);

    Packet* userPacket = NULL;
    XLINK_RET_IF(BlockingQueue_Pop(&connection->userPacketQueue[streamId], (void**)&userPacket));
    XLINK_RET_IF(userPacket == NULL);

    return Packet_Release(userPacket);
}

XLinkError_t Connection_GetFillLevel(Connection* connection, streamId_t streamId, int* fillLevel) {
    XLINK_RET_IF(connection == NULL);
    XLINK_RET_IF(streamId > XLINK_MAX_STREAMS);

    *fillLevel = 0;

    return X_LINK_SUCCESS;
}

xLinkConnectionStatus_t Connection_GetStatus(Connection* connection) {
    XLINK_RET_ERR_IF(connection == NULL, XLINK_CONNECTION_DOWN);

    return connection->status;
}

linkId_t Connection_GetId(Connection* connection) {
    XLINK_RET_ERR_IF(connection == NULL, INVALID_LINK_ID);

    return connection->id;
}

// ------------------------------------
// API methods implementation. End.
// ------------------------------------

// ------------------------------------
// Private methods implementation. Begin.
// ------------------------------------

static Packet* _connection_GetPacket(Connection* connection, streamId_t streamId,
                                     xLinkEventType_t type,
                                     const uint8_t* buffer, int size) {
    XLINK_RET_ERR_IF(connection == NULL, NULL);
    XLINK_RET_ERR_IF(streamId >= XLINK_MAX_STREAMS, NULL);

    Packet* packet = StreamDispatcher_GetPacket(
            &connection->streamDispatcher,
            streamId,
            OUT_CHANNEL);
    if (packet == NULL) {
        mvLog(MVLOG_ERROR, "cannot get packet");
        return NULL;
    }

    packet->header.type = type;
    if (Packet_SetData(packet, (void*)buffer, size) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "cannot set data to packet id=%d, idx=%d",
              packet->header.id, packet->privateFields.idx);
        Packet_Release(packet);
        return NULL;
    }

    return packet;
}

static XLinkError_t _connection_SendPacket(Connection* connection, Packet* packet, unsigned int timeoutMs) {
    XLINK_RET_IF(connection == NULL);

    mvLog(MVLOG_DEBUG, "Push packet to packetsToSendQueue: id=%d, idx=%d",
          packet->header.id, packet->privateFields.idx);

    XLinkError_t packetPushedToQueueStatus = BlockingQueue_Push(&connection->packetsToSendQueue, packet);
    if (packetPushedToQueueStatus != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "_connection_SendPacket: cannot push packet to queue, id=%d, idx=%d",
              packet->header.id, packet->privateFields.idx);
        return packetPushedToQueueStatus;
    }

    if (timeoutMs) {
        XLinkError_t rc = Packet_TimedWaitPacketComplete(packet, timeoutMs);
        if (rc != X_LINK_SUCCESS) {
            return rc;
        }
    } else {
        XLINK_RET_IF(Packet_WaitPacketComplete(packet));
    }

    XLinkError_t isCompleted = X_LINK_SUCCESS;
    packetStatus_t status;
    Packet_GetPacketStatus(packet, &status);
    if (status != PACKET_COMPLETED) {
        mvLog(MVLOG_ERROR, "Packet sending failed. Packet status %d.", status);
        isCompleted = X_LINK_ERROR;
    }

    return isCompleted;
}

static XLinkError_t _connection_ReceivePacket(Connection *connection, streamId_t streamId,
                                              streamPacketDesc_t** packet, unsigned int timeoutMs) {
    Packet* receivedPacket = NULL;

    if (timeoutMs) {
        XLinkError_t rc = BlockingQueue_TimedPop(&connection->receivedPacketsQueue[streamId], (void**)&receivedPacket, timeoutMs);
        if (rc != X_LINK_SUCCESS) {
            return rc;
        }
    } else {
        XLINK_RET_IF(BlockingQueue_Pop(&connection->receivedPacketsQueue[streamId], (void**)&receivedPacket));
    }
    XLINK_RET_IF(receivedPacket == NULL);

    packetStatus_t status;
    Packet_GetPacketStatus(receivedPacket, &status);
    if (status == PACKET_DROPPED) {
        mvLog(MVLOG_ERROR, "Some error occurred, so packet has been dropped, returning NULL");
        Packet_Release(receivedPacket);
        *packet = NULL;
        return X_LINK_ERROR;
    }

    receivedPacket->userData.data = receivedPacket->data;
    receivedPacket->userData.length = receivedPacket->header.size;
    *packet = &receivedPacket->userData;

    return BlockingQueue_Push(&connection->userPacketQueue[streamId], receivedPacket);
}

// ------------------------------------
// Private methods implementation. End.
// ------------------------------------
