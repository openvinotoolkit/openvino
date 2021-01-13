// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINKPRIVATEFIELDS_H
#define _XLINKPRIVATEFIELDS_H

#include "XLinkConnection.h"
#include "XLinkPrivateDefines.h"

#define LINK_ID_MASK 0xFF
#define LINK_ID_SHIFT ((sizeof(uint32_t) - sizeof(uint8_t)) * 8)
#define STREAM_ID_MASK 0xFFFFFF

#define EXTRACT_LINK_ID(streamId) (((streamId) >> (uint32_t)LINK_ID_SHIFT) & (uint32_t)LINK_ID_MASK)
#define EXTRACT_STREAM_ID(streamId) ((streamId) & (uint32_t)STREAM_ID_MASK)

#define COMBINE_IDS(streamId, linkid) \
    streamId = streamId | ((linkid & LINK_ID_MASK) << LINK_ID_SHIFT);

// ------------------------------------
// Global fields declaration. Begin.
// ------------------------------------
extern XLinkGlobalHandler_t* glHandler;
extern Connection availableConnections[MAX_LINKS];

extern sem_t pingSem;

// ------------------------------------
// Global fields declaration. End.
// ------------------------------------


// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

Connection* getLinkById(linkId_t id);

int XLink_isOnHostSide();
int XLink_isOnDeviceSide();

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------

#endif //PROJECT_XLINKPRIVATEFIELDS_H
