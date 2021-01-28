// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINK_DISPATCHERNEW_H
#define OPENVINO_XLINK_DISPATCHERNEW_H

#include "XLinkPublicDefines.h"
#include "XLinkPrivateDefines.h"
#include "XLinkBlockingQueue.h"
#include "XLinkStream.h"

typedef enum {
    DISPATCHER_INITIALIZED = 0,
    DISPATCHER_UP,
    DISPATCHER_NEED_TO_CLOSE,
    DISPATCHER_WAITING_TO_CLOSE,
    DISPATCHER_DOWN
} DispatcherStatus_t;

typedef struct DispatcherNew_t {
    DispatcherStatus_t status;
    xLinkDeviceHandle_t* deviceHandle;

    StreamDispatcher* streamDispatcher;
    BlockingQueue* packetsToSendQueue;
    BlockingQueue* receivedPacketsQueue;

    pthread_t sendPacketsThread;
    pthread_t receivePacketsThread;
} DispatcherNew;

XLinkError_t Dispatcher_Create(
        DispatcherNew* dispatcher,
        StreamDispatcher* streamDispatcher,
        BlockingQueue* packetsToSendQueue,
        BlockingQueue* receivedPacketsQueue);
void Dispatcher_Destroy(
        DispatcherNew* dispatcher);

XLinkError_t Dispatcher_Start(
        DispatcherNew* dispatcher,
        xLinkDeviceHandle_t* deviceHandle,
        linkId_t connectionId);
XLinkError_t Dispatcher_Stop(
        DispatcherNew* dispatcher);

DispatcherStatus_t Dispatcher_GetStatus(
        DispatcherNew* dispatcher);
void Dispatcher_SetStatus(
        DispatcherNew* dispatcher, DispatcherStatus_t status);

#endif  // OPENVINO_XLINK_DISPATCHERNEW_H
