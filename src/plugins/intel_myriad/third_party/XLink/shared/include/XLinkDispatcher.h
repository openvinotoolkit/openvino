// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///
#ifndef _XLINKDISPATCHER_H
#define _XLINKDISPATCHER_H

#include "XLinkPrivateDefines.h"

#ifdef __cplusplus
extern "C"
{
#endif
typedef int (*getRespFunction) (xLinkEvent_t*,
                xLinkEvent_t*);
typedef struct {
    int (*eventSend) (xLinkEvent_t*);
    int (*eventReceive) (xLinkEvent_t*);
    getRespFunction localGetResponse;
    getRespFunction remoteGetResponse;
    void (*closeLink) (void* fd, int fullClose);
    void (*closeDeviceFd) (xLinkDeviceHandle_t* deviceHandle);
} DispatcherControlFunctions;

XLinkError_t DispatcherInitialize(DispatcherControlFunctions *controlFunc);
XLinkError_t DispatcherStart(xLinkDeviceHandle_t *deviceHandle);
int DispatcherClean(xLinkDeviceHandle_t *deviceHandle);

xLinkEvent_t* DispatcherAddEvent(xLinkEventOrigin_t origin, xLinkEvent_t *event);
int DispatcherWaitEventComplete(xLinkDeviceHandle_t *deviceHandle, unsigned int timeoutMs);

char* TypeToStr(int type);
int DispatcherUnblockEvent(eventId_t id,
                             xLinkEventType_t type,
                             streamId_t stream,
                             void *xlinkFD);
int DispatcherServeEvent(eventId_t id,
                             xLinkEventType_t type,
                             streamId_t stream,
                             void *xlinkFD);
#ifdef __cplusplus
}
#endif

#endif