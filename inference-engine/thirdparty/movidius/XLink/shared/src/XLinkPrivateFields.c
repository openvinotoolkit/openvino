// Copyright (C) 2018-2020 Intel Corporation
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

xLinkDesc_t* getLinkById(linkId_t id)
{
    int i;
    for (i = 0; i < MAX_LINKS; i++)
        if (availableXLinks[i].id == id)
            return &availableXLinks[i];
    return NULL;
}

xLinkDesc_t* getLink(void* fd)
{
    int i;
    for (i = 0; i < MAX_LINKS; i++)
        if (availableXLinks[i].deviceHandle.xLinkFD == fd)
            return &availableXLinks[i];
    return NULL;
}

streamId_t getStreamIdByName(xLinkDesc_t* link, const char* name)
{
    streamDesc_t* stream = getStreamByName(link, name);

    if (stream) {
        streamId_t id = stream->id;
        releaseStream(stream);
        return id;
    }

    return INVALID_STREAM_ID;
}

streamDesc_t* getStreamById(void* fd, streamId_t id)
{
    xLinkDesc_t* link = getLink(fd);
    ASSERT_X_LINK_R(link != NULL, NULL);
    int stream;
    for (stream = 0; stream < XLINK_MAX_STREAMS; stream++) {
        if (link->availableStreams[stream].id == id) {
            sem_wait(&link->availableStreams[stream].sem);

            return &link->availableStreams[stream];
        }
    }
    return NULL;
}

streamDesc_t* getStreamByName(xLinkDesc_t* link, const char* name)
{
    ASSERT_X_LINK_R(link != NULL, NULL);
    int stream;
    for (stream = 0; stream < XLINK_MAX_STREAMS; stream++) {
        if (link->availableStreams[stream].id != INVALID_STREAM_ID &&
            strcmp(link->availableStreams[stream].name, name) == 0) {
            sem_wait(&link->availableStreams[stream].sem);

            return &link->availableStreams[stream];
        }
    }
    return NULL;
}

void releaseStream(streamDesc_t* stream)
{
    if (stream && stream->id != INVALID_STREAM_ID) {
        sem_post(&stream->sem);
    }
    else {
        mvLog(MVLOG_DEBUG,"trying to release a semaphore for a released stream\n");
    }
}

xLinkState_t getXLinkState(xLinkDesc_t* link)
{
    ASSERT_X_LINK_R(link != NULL, XLINK_NOT_INIT);
    mvLog(MVLOG_DEBUG,"%s() link %p link->peerState %d\n", __func__,link, link->peerState);
    return link->peerState;
}
