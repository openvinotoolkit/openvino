// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>

#include "XLinkStream.h"
#include "XLinkTool.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif

#include "XLinkLog.h"
#include "XLinkStringUtils.h"

XLinkError_t XLinkStreamInitialize(
    streamDesc_t* stream, streamId_t id, const char* name) {
    mvLog(MVLOG_DEBUG, "name: %s, id: %ld\n", name, id);
    ASSERT_X_LINK(stream);

    memset(stream, 0, sizeof(*stream));

    if (sem_init(&stream->sem, 0, 0)) {
        mvLog(MVLOG_ERROR, "Cannot initialize semaphore\n");
        return X_LINK_ERROR;
    }

    stream->id = id;
    mv_strncpy(stream->name, MAX_STREAM_NAME_LENGTH,
               name, MAX_STREAM_NAME_LENGTH - 1);

    return X_LINK_SUCCESS;
}

void XLinkStreamReset(streamDesc_t* stream) {
    if(stream == NULL) {
        return;
    }

    if(sem_destroy(&stream->sem)) {
        mvLog(MVLOG_DEBUG, "Cannot destroy semaphore\n");
    }

    memset(stream, 0, sizeof(*stream));
    stream->id = INVALID_STREAM_ID;
}
