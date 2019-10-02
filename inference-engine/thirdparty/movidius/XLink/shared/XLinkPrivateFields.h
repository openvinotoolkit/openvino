// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINKPRIVATEFIELDS_H
#define _XLINKPRIVATEFIELDS_H

#include "XLinkDispatcher.h"

// ------------------------------------
// Global fields declaration. Begin.
// ------------------------------------

extern XLinkGlobalHandler_t* glHandler; //TODO need to either protect this with semaphor
                                        //or make profiling data per device

extern xLinkDesc_t availableXLinks[MAX_LINKS];
extern DispatcherControlFunctions controlFunctionTbl;
extern sem_t  pingSem; //to b used by myriad

// ------------------------------------
// Global fields declaration. End.
// ------------------------------------


// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

streamId_t getStreamIdByName(xLinkDesc_t* link, const char* name);
xLinkDesc_t* getLinkByStreamId(streamId_t streamId);
xLinkDesc_t* getLinkById(linkId_t id);
xLinkDesc_t* getLink(void* fd);
xLinkState_t getXLinkState(xLinkDesc_t* link);


streamDesc_t* getStreamById(void* fd, streamId_t id);
streamDesc_t* getStreamByName(xLinkDesc_t* link, const char* name);

void releaseStream(streamDesc_t* stream);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------

#endif //PROJECT_XLINKPRIVATEFIELDS_H
