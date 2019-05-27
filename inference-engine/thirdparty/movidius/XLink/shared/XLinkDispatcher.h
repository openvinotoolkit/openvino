/*
* Copyright 2018-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

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
