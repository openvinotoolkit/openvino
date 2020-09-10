// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stdio.h"
#include "stdint.h"
#include "string.h"
#include "stdlib.h"

#include "XLink.h"
#include "XLinkErrorUtils.h"

#include "XLinkPlatform.h"
#include "XLinkPrivateFields.h"
#include "XLinkDispatcherImpl.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "XLinkLog.h"
#include "XLinkStringUtils.h"

#define MAX_PATH_LENGTH (255)

// ------------------------------------
// Global fields. Begin.
// ------------------------------------

XLinkGlobalHandler_t* glHandler; //TODO need to either protect this with semaphor
                                 //or make profiling data per device

xLinkDesc_t availableXLinks[MAX_LINKS];
sem_t  pingSem; //to b used by myriad
DispatcherControlFunctions controlFunctionTbl;
linkId_t nextUniqueLinkId = 0; //incremental number, doesn't get decremented.

// ------------------------------------
// Global fields. End.
// ------------------------------------



// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

static linkId_t getNextAvailableLinkUniqueId();
static xLinkDesc_t* getNextAvailableLink();

#ifdef __PC__

static XLinkError_t parsePlatformError(xLinkPlatformErrorCode_t rc);

#endif // __PC__

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------



// ------------------------------------
// API implementation. Begin.
// ------------------------------------

XLinkError_t XLinkInitialize(XLinkGlobalHandler_t* globalHandler)
{
#ifndef __PC__
    mvLogLevelSet(MVLOG_FATAL);
    mvLogDefaultLevelSet(MVLOG_FATAL);
#endif

    XLINK_RET_IF(globalHandler == NULL);
    ASSERT_XLINK(XLINK_MAX_STREAMS <= MAX_POOLS_ALLOC);
    glHandler = globalHandler;
    if (sem_init(&pingSem,0,0)) {
        mvLog(MVLOG_ERROR, "Can't create semaphore\n");
    }
    int i;

    XLinkPlatformInit();

    //Using deprecated fields. Begin.
    int loglevel = globalHandler->loglevel;
    int protocol = globalHandler->protocol;
    //Using deprecated fields. End.

    memset((void*)globalHandler, 0, sizeof(XLinkGlobalHandler_t));

    //Using deprecated fields. Begin.
    globalHandler->loglevel = loglevel;
    globalHandler->protocol = protocol;
    //Using deprecated fields. End.

    controlFunctionTbl.eventReceive      = &dispatcherEventReceive;
    controlFunctionTbl.eventSend         = &dispatcherEventSend;
    controlFunctionTbl.localGetResponse  = &dispatcherLocalEventGetResponse;
    controlFunctionTbl.remoteGetResponse = &dispatcherRemoteEventGetResponse;
    controlFunctionTbl.closeLink         = &dispatcherCloseLink;
    controlFunctionTbl.closeDeviceFd     = &dispatcherCloseDeviceFd;

    XLINK_RET_IF(DispatcherInitialize(&controlFunctionTbl));

    //initialize availableStreams
    memset(availableXLinks, 0, sizeof(availableXLinks));

    xLinkDesc_t* link;
    for (i = 0; i < MAX_LINKS; i++) {
        link = &availableXLinks[i];

        link->id = INVALID_LINK_ID;
        link->deviceHandle.xLinkFD = NULL;
        link->peerState = XLINK_NOT_INIT;
        int stream;
        for (stream = 0; stream < XLINK_MAX_STREAMS; stream++)
            link->availableStreams[stream].id = INVALID_STREAM_ID;
    }

#ifndef __PC__
    link = getNextAvailableLink();
    if (link == NULL)
        return X_LINK_COMMUNICATION_NOT_OPEN;

    link->id = getNextAvailableLinkUniqueId();
    link->peerState = XLINK_UP;
    link->deviceHandle.xLinkFD = NULL;

    xLinkDeviceHandle_t temp = {0};
    temp.protocol = X_LINK_ANY_PROTOCOL;
    XLINK_RET_IF_FAIL(DispatcherStart(&temp)); //myriad has one

    sem_wait(&pingSem);
#endif

    return X_LINK_SUCCESS;
}

#ifdef __PC__

int XLinkIsDescriptionValid(const deviceDesc_t *in_deviceDesc, const XLinkDeviceState_t state) {
    return XLinkPlatformIsDescriptionValid(in_deviceDesc, state);
}

XLinkError_t XLinkFindFirstSuitableDevice(XLinkDeviceState_t state,
                                          const deviceDesc_t in_deviceRequirements,
                                          deviceDesc_t *out_foundDevice)
{
    XLINK_RET_IF(out_foundDevice == NULL);

    xLinkPlatformErrorCode_t rc;
    rc = XLinkPlatformFindDeviceName(state, in_deviceRequirements, out_foundDevice);
    return parsePlatformError(rc);
}

XLinkError_t XLinkFindAllSuitableDevices(XLinkDeviceState_t state,
                                         const deviceDesc_t in_deviceRequirements,
                                         deviceDesc_t *out_foundDevicesPtr,
                                         const unsigned int devicesArraySize,
                                         unsigned int* out_foundDevicesCount) {
    XLINK_RET_IF(out_foundDevicesPtr == NULL);
    XLINK_RET_IF(devicesArraySize <= 0);
    XLINK_RET_IF(out_foundDevicesCount == NULL);

    xLinkPlatformErrorCode_t rc;
    rc = XLinkPlatformFindArrayOfDevicesNames(
        state, in_deviceRequirements,
        out_foundDevicesPtr, devicesArraySize, out_foundDevicesCount);

    return parsePlatformError(rc);
}

//Called only from app - per device
XLinkError_t XLinkConnect(XLinkHandler_t* handler)
{
    XLINK_RET_IF(handler == NULL);
    if (strnlen(handler->devicePath, MAX_PATH_LENGTH) < 2) {
        mvLog(MVLOG_ERROR, "Device path is incorrect");
        return X_LINK_ERROR;
    }

    xLinkDesc_t* link = getNextAvailableLink();
    XLINK_RET_IF(link == NULL);
    mvLog(MVLOG_DEBUG,"%s() device name %s glHandler %p protocol %d\n", __func__, handler->devicePath, glHandler, handler->protocol);

    link->deviceHandle.protocol = handler->protocol;
    int connectStatus = XLinkPlatformConnect(handler->devicePath2, handler->devicePath,
                                             link->deviceHandle.protocol, &link->deviceHandle.xLinkFD);

    if (connectStatus < 0) {
        /**
         * Connection may be unsuccessful at some amount of first tries.
         * In this case, asserting the status provides enormous amount of logs in tests.
         */
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    XLINK_RET_ERR_IF(
        DispatcherStart(&link->deviceHandle) != X_LINK_SUCCESS, X_LINK_TIMEOUT);

    xLinkEvent_t event = {0};

    event.header.type = XLINK_PING_REQ;
    event.deviceHandle = link->deviceHandle;
    DispatcherAddEvent(EVENT_LOCAL, &event);

    if (DispatcherWaitEventComplete(&link->deviceHandle)) {
        DispatcherClean(&link->deviceHandle);
        return X_LINK_TIMEOUT;
    }

    link->id = getNextAvailableLinkUniqueId();
    link->peerState = XLINK_UP;
    link->hostClosedFD = 0;
    handler->linkId = link->id;
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkBoot(deviceDesc_t* deviceDesc, const char* binaryPath)
{
    if (XLinkPlatformBootRemote(deviceDesc, binaryPath) == 0) {
        return X_LINK_SUCCESS;
    }

    return X_LINK_COMMUNICATION_FAIL;
}

XLinkError_t XLinkBootFirmware(deviceDesc_t* deviceDesc, const char* firmware, unsigned long length) {
    if (!XLinkPlatformBootFirmware(deviceDesc, firmware, length)) {
        return X_LINK_SUCCESS;
    }

    return X_LINK_COMMUNICATION_FAIL;
}

XLinkError_t XLinkResetRemote(linkId_t id)
{
    xLinkDesc_t* link = getLinkById(id);
    XLINK_RET_IF(link == NULL);

    if (getXLinkState(link) != XLINK_UP) {
        mvLog(MVLOG_WARN, "Link is down, close connection to device without reset");
        XLinkPlatformCloseRemote(&link->deviceHandle);
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    // Add event to reset device. After sending it, dispatcher will close fd link
    xLinkEvent_t event = {0};
    event.header.type = XLINK_RESET_REQ;
    event.deviceHandle = link->deviceHandle;
    mvLog(MVLOG_DEBUG, "sending reset remote event\n");
    DispatcherAddEvent(EVENT_LOCAL, &event);
    XLINK_RET_ERR_IF(DispatcherWaitEventComplete(&link->deviceHandle),
        X_LINK_TIMEOUT);

    if(sem_wait(&link->dispatcherClosedSem)) {
        mvLog(MVLOG_ERROR,"can't wait dispatcherClosedSem\n");
        return X_LINK_ERROR;
    }

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkResetAll()
{
#if defined(NO_BOOT)
    mvLog(MVLOG_INFO, "Devices will not be restarted for this configuration (NO_BOOT)");
#else
    int i;
    for (i = 0; i < MAX_LINKS; i++) {
        if (availableXLinks[i].id != INVALID_LINK_ID) {
            xLinkDesc_t* link = &availableXLinks[i];
            int stream;
            for (stream = 0; stream < XLINK_MAX_STREAMS; stream++) {
                if (link->availableStreams[stream].id != INVALID_STREAM_ID) {
                    streamId_t streamId = link->availableStreams[stream].id;
                    mvLog(MVLOG_DEBUG,"%s() Closing stream (stream = %d) %d on link %d\n",
                          __func__, stream, (int) streamId, (int) link->id);
                    COMBINE_IDS(streamId, link->id);
                    if (XLinkCloseStream(streamId) != X_LINK_SUCCESS) {
                        mvLog(MVLOG_WARN,"Failed to close stream");
                    }
                }
            }
            if (XLinkResetRemote(link->id) != X_LINK_SUCCESS) {
                mvLog(MVLOG_WARN,"Failed to reset");
            }
        }
    }
#endif
    return X_LINK_SUCCESS;
}

#endif // __PC__

XLinkError_t XLinkProfStart()
{
    glHandler->profEnable = 1;
    glHandler->profilingData.totalReadBytes = 0;
    glHandler->profilingData.totalWriteBytes = 0;
    glHandler->profilingData.totalWriteTime = 0;
    glHandler->profilingData.totalReadTime = 0;
    glHandler->profilingData.totalBootCount = 0;
    glHandler->profilingData.totalBootTime = 0;

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkProfStop()
{
    glHandler->profEnable = 0;
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkProfPrint()
{
    printf("XLink profiling results:\n");
    if (glHandler->profilingData.totalWriteTime)
    {
        printf("Average write speed: %f MB/Sec\n",
               glHandler->profilingData.totalWriteBytes /
               glHandler->profilingData.totalWriteTime /
               1024.0 /
               1024.0 );
    }
    if (glHandler->profilingData.totalReadTime)
    {
        printf("Average read speed: %f MB/Sec\n",
               glHandler->profilingData.totalReadBytes /
               glHandler->profilingData.totalReadTime /
               1024.0 /
               1024.0);
    }
    if (glHandler->profilingData.totalBootCount)
    {
        printf("Average boot speed: %f sec\n",
               glHandler->profilingData.totalBootTime /
               glHandler->profilingData.totalBootCount);
    }
    return X_LINK_SUCCESS;
}

// ------------------------------------
// API implementation. End.
// ------------------------------------


// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------

static linkId_t getNextAvailableLinkUniqueId()
{
    linkId_t start = nextUniqueLinkId;
    do
    {
        int i;
        for (i = 0; i < MAX_LINKS; i++)
        {
            if (availableXLinks[i].id != INVALID_LINK_ID &&
                availableXLinks[i].id == nextUniqueLinkId)
                break;
        }
        if (i >= MAX_LINKS)
        {
            return nextUniqueLinkId;
        }
        nextUniqueLinkId++;
        if (nextUniqueLinkId == INVALID_LINK_ID)
        {
            nextUniqueLinkId = 0;
        }
    } while (start != nextUniqueLinkId);
    mvLog(MVLOG_ERROR, "%s():- no next available unique link id!\n", __func__);
    return INVALID_LINK_ID;
}

static xLinkDesc_t* getNextAvailableLink() {
    int i;
    for (i = 0; i < MAX_LINKS; i++) {
        if (availableXLinks[i].id == INVALID_LINK_ID) {
            break;
        }
    }

    if(i >= MAX_LINKS) {
        mvLog(MVLOG_ERROR,"%s():- no next available link!\n", __func__);
        return NULL;
    }

    xLinkDesc_t* link = &availableXLinks[i];

    if (sem_init(&link->dispatcherClosedSem, 0 ,0)) {
        mvLog(MVLOG_ERROR, "Cannot initialize semaphore\n");
        return NULL;
    }

    return link;
}

#ifdef __PC__

static XLinkError_t parsePlatformError(xLinkPlatformErrorCode_t rc) {
    switch (rc) {
        case X_LINK_PLATFORM_SUCCESS:
            return X_LINK_SUCCESS;
        case X_LINK_PLATFORM_DEVICE_NOT_FOUND:
            return X_LINK_DEVICE_NOT_FOUND;
        case X_LINK_PLATFORM_TIMEOUT:
            return X_LINK_TIMEOUT;
        default:
            return X_LINK_ERROR;
    }
}

#endif // __PC__

// ------------------------------------
// Helpers implementation. End.
// ------------------------------------
