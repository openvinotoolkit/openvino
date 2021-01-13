// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif

#include "XLinkPrivateFields.h"
#include "XLinkPrivateDefines.h"
#include "XLinkLog.h"

#include <string.h>

Connection* getLinkById(linkId_t id)
{
    for (int i = 0; i < MAX_LINKS; i++) {
        linkId_t currId = Connection_GetId(&availableConnections[i]);
        if (currId == id) {
            return &availableConnections[i];
        }
    }

    return NULL;
}

int XLink_isOnHostSide() {
#ifdef __PC__
    return 1;
#else
    return 0;
#endif
}

int XLink_isOnDeviceSide() {
    return !XLink_isOnHostSide();
}
