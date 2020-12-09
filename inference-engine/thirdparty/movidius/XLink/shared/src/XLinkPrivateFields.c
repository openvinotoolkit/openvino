// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdio.h>
#include <string.h>
#include "stdlib.h"

#include "XLinkPrivateFields.h"
#include "XLinkPrivateDefines.h"
#include "XLinkTool.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif

#include "XLinkLog.h"

Connection* getLinkById(linkId_t id)
{
    int i;
    for (i = 0; i < MAX_LINKS; i++) {
        linkId_t currId = Connection_GetId(&availableConnections[i]);
        if(currId == id) {
            return &availableConnections[i];
        }
    }

    return NULL;
}
