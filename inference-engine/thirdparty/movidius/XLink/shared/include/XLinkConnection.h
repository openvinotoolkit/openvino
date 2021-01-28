// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINK_CONNECTION_H
#define OPENVINO_XLINK_CONNECTION_H

#include "XLinkPublicDefines.h"
#include "XLinkPrivateDefines.h"
#include "XLinkDispatcherNew.h"

typedef enum {
    XLINK_CONNECTION_INITIALIZED = 0,
    XLINK_CONNECTION_UP,
    XLINK_CONNECTION_NEED_TO_CLOSE,
    XLINK_CONNECTION_WAITING_TO_CLOSE,
    XLINK_CONNECTION_DOWN
} xLinkConnectionStatus_t;

typedef struct Connection_t {
    linkId_t id;
    xLinkConnectionStatus_t status;
    xLinkDeviceHandle_t deviceHandle;

    DispatcherNew dispatcher;
    StreamDispatcher streamDispatcher;

    BlockingQueue packetsToSendQueue;
    BlockingQueue receivedPacketsQueue[XLINK_MAX_STREAMS];
    BlockingQueue userPacketQueue[XLINK_MAX_STREAMS];
} Connection;


XLinkError_t Connection_Init(
        Connection* connection,
        linkId_t id);
XLinkError_t Connection_Clean(
        Connection* connection);

#ifdef __PC__

XLinkError_t Connection_Connect(
        Connection* connection,
        XLinkHandler_t* handler);
XLinkError_t Connection_Reset(
        Connection* connection);

#endif  // __PC__

// ------------------------------------
// Functions to support old API. Begin.
// ------------------------------------

streamId_t Connection_OpenStream(
        Connection* connection,
        const char* name,
        int stream_write_size);
XLinkError_t Connection_CloseStream(
        Connection* connection,
        streamId_t streamId);

// ------------------------------------
// Functions to support old API. End.
// ------------------------------------

XLinkError_t Connection_Write(
        Connection* connection,
        streamId_t streamId,
        const uint8_t* buffer,
        int size);
XLinkError_t Connection_Read(
        Connection* connection,
        streamId_t streamId,
        streamPacketDesc_t** packet);

XLinkError_t Connection_ReleaseData(
        Connection* connection,
        streamId_t streamId);
XLinkError_t Connection_GetFillLevel(
        Connection* connection,
        streamId_t streamId,
        int* fillLevel);

xLinkConnectionStatus_t Connection_GetStatus(
        Connection* connection);
linkId_t Connection_GetId(
        Connection* connection);

#endif  // OPENVINO_XLINK_CONNECTION_H
