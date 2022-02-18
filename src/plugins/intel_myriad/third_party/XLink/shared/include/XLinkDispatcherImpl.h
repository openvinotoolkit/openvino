// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINKDISPATCHERIMPL_H
#define _XLINKDISPATCHERIMPL_H

#include "XLinkPrivateDefines.h"

int dispatcherEventSend (xLinkEvent_t*);
int dispatcherEventReceive (xLinkEvent_t*);
int dispatcherLocalEventGetResponse (xLinkEvent_t*,
                        xLinkEvent_t*);
int dispatcherRemoteEventGetResponse (xLinkEvent_t*,
                        xLinkEvent_t*);
void dispatcherCloseLink (void* fd, int fullClose);
void dispatcherCloseDeviceFd (xLinkDeviceHandle_t* deviceHandle);

#endif //_XLINKDISPATCHERIMPL_H
