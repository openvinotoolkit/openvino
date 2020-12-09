// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINKCONNECTION_H
#define OPENVINO_XLINKCONNECTION_H

#include "XLinkPublicDefines.h"
#include "XLinkPrivateDefines.h"
#include "XLinkDispatcherNew.h"

typedef enum{
    CONNECTION_INITIALIZED = 0,
    CONNECTION_UP,
    CONNECTION_NEED_TO_CLOOSE,
    CONNECTION_WAITING_TO_CLOSE,
    CONNECTION_DOWN
} ConnectionStatus_t;

typedef struct Connection_t {
    linkId_t id;
    ConnectionStatus_t status;
    xLinkDeviceHandle_t deviceHandle;

    DispatcherNew* dispatcher;
    StreamDispatcher* streamDispatcher;

    BlockingQueue* packetsToSendQueue;
    BlockingQueue* receivedPacketsQueue[MAX_STREAMS_NEW];
    BlockingQueue* userPacketQueue[MAX_STREAMS_NEW];
} Connection;

XLinkError_t Connection_Init(Connection* connection, linkId_t id);
XLinkError_t Connection_Clean(Connection* connection);

XLinkError_t Connection_Connect(Connection* connection, XLinkHandler_t* handler);
XLinkError_t Connection_Reset(Connection* connection);

// For support old XLink. Begin.
streamId_t Connection_OpenStream(Connection* connection, const char* name, int stream_write_size);
XLinkError_t Connection_CloseStream(Connection* connection, streamId_t streamId);
// For support old XLink. End.

XLinkError_t Connection_Write(Connection* connection, streamId_t streamId, const uint8_t* buffer, int size);
XLinkError_t Connection_Read(Connection* connection, streamId_t streamId, streamPacketDesc_t** packet);

XLinkError_t Connection_ReleaseData(Connection* connection, streamId_t streamId);
XLinkError_t Connection_GetFillLevel(Connection* connection, streamId_t streamId, int isRemote, int* fillLevel);

ConnectionStatus_t Connection_GetStatus(Connection* connection);
linkId_t Connection_GetId(Connection* connection);

#endif //OPENVINO_XLINKCONNECTION_H
