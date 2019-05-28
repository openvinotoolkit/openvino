// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///
#ifndef _XLINKDISPATCHER_H
#define _XLINKDISPATCHER_H
#define _XLINK_ENABLE_PRIVATE_INCLUDE_
#include "XLinkPrivateDefines.h"

#ifdef __cplusplus
extern "C"
{
#endif
typedef int (*getRespFunction) (xLinkEvent_t*,
                xLinkEvent_t*);
///Adds a new event with parameters and returns event.header.id
xLinkEvent_t* dispatcherAddEvent(xLinkEventOrigin_t origin,
                                    xLinkEvent_t *event);

int dispatcherWaitEventComplete(void* xlinkFD, unsigned int timeout);
int dispatcherUnblockEvent(eventId_t id,
                            xLinkEventType_t type,
                            streamId_t stream,
                            void* xlinkFD);

struct dispatcherControlFunctions {
                                int (*eventSend) (xLinkEvent_t*);
                                int (*eventReceive) (xLinkEvent_t*);
                                getRespFunction localGetResponse;
                                getRespFunction remoteGetResponse;
                                void (*closeLink) (void* fd);
                                void (*closeDeviceFd) (void* fd);
                                };

int dispatcherInitialize(struct dispatcherControlFunctions* controlFunc);
int dispatcherStart(void* fd);
int dispatcherClean(void* xLinkFD);

#ifdef __cplusplus
}
#endif

#endif
