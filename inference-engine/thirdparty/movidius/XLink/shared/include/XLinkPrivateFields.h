// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINKPRIVATEFIELDS_H
#define _XLINKPRIVATEFIELDS_H

#include "XLinkDispatcher.h"

#define LINK_ID_MASK 0xFF
#define LINK_ID_SHIFT ((sizeof(uint32_t) - sizeof(uint8_t)) * 8)
#define STREAM_ID_MASK 0xFFFFFF

#define EXTRACT_LINK_ID(streamId) (((streamId) >> LINK_ID_SHIFT) & LINK_ID_MASK)
#define EXTRACT_STREAM_ID(streamId) ((streamId) & STREAM_ID_MASK)

#define COMBINE_IDS(streamId, linkid) \
    streamId = streamId | ((linkid & LINK_ID_MASK) << LINK_ID_SHIFT);

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

xLinkDesc_t* getLinkById(linkId_t id);
xLinkDesc_t* getLink(void* fd);
xLinkState_t getXLinkState(xLinkDesc_t* link);


streamId_t getStreamIdByName(xLinkDesc_t* link, const char* name);

streamDesc_t* getStreamById(void* fd, streamId_t id);
streamDesc_t* getStreamByName(xLinkDesc_t* link, const char* name);

void releaseStream(streamDesc_t* stream);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------

#endif //PROJECT_XLINKPRIVATEFIELDS_H
