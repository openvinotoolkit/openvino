// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef OPENVINO_XLINKDISPATCHERNEW_H
#define OPENVINO_XLINKDISPATCHERNEW_H

#include "XLinkPublicDefines.h"
#include "XLinkPrivateDefines.h"
#include "XLinkBlockingQueue.h"
#include "XLinkStream.h"

typedef enum{
    DISPATCHER_INITIALIZED = 0,
    DISPATCHER_UP,
    DISPATCHER_NEED_TO_CLOOSE,
    DISPATCHER_WAITING_TO_CLOSE,
    DISPATCHER_DOWN
} DispatcherStatus_t;

typedef struct DispatcherNew_t DispatcherNew;

DispatcherNew* Dispatcher_Create(StreamDispatcher* streamDispatcher,
                                 BlockingQueue* packetsToSendQueue,
                                 BlockingQueue* receivedPacketsQueue[MAX_STREAMS_NEW]);
void Dispatcher_Destroy(DispatcherNew* dispatcher);

XLinkError_t Dispatcher_Start(DispatcherNew* dispatcher, xLinkDeviceHandle_t* deviceHandle);
XLinkError_t Dispatcher_Stop(DispatcherNew* dispatcher);

DispatcherStatus_t Dispatcher_GetStatus(DispatcherNew* dispatcher);

#endif //OPENVINO_XLINKDISPATCHERNEW_H
