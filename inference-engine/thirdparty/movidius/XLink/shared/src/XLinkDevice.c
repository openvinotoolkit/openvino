// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "XLinkLog.h"
#include "XLinkStringUtils.h"

#include "XLink.h"

#include "XLinkConnection.h"
#include "XLinkErrorUtils.h"
#include "XLinkPlatform.h"
#include "XLinkPrivateFields.h"

#if (defined(_WIN32) || defined(_WIN64))
# include "win_pthread.h"
# include "win_semaphore.h"
#else
# include <pthread.h>
# ifndef __APPLE__
#  include <semaphore.h>
# endif
#endif

#include "stdio.h"
#include "stdint.h"
#include "string.h"
#include "stdlib.h"

#define MAX_PATH_LENGTH (255)

// ------------------------------------
// Global fields. Begin.
// ------------------------------------

XLinkGlobalHandler_t* glHandler; //TODO need to either protect this with semaphor
                                 //or make profiling data per device
int g_IsInitialized = 0;

#define FREE_CONNECTION_FLAG 0
#define BUSY_CONNECTION_FLAG 1

Connection availableConnections[MAX_LINKS];
int freeConnectionsIds[MAX_LINKS];
linkId_t nextUniqueLinkId = 0;  // incremental number, doesn't get decremented.
sem_t pingSem;  // to be used by myriad

// ------------------------------------
// Global fields. End.
// ------------------------------------



// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

static linkId_t getNextAvailableLinkUniqueId();
static Connection* getNextAvailableConnection();

#ifdef __PC__

static XLinkError_t releaseConnection(Connection* connection);

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
    // mvLogLevelSet(MVLOG_DEBUG);
    // mvLogDefaultLevelSet(MVLOG_DEBUG);

    XLINK_RET_IF(globalHandler == NULL);
    ASSERT_XLINK(XLINK_MAX_STREAMS <= MAX_POOLS_ALLOC);
    glHandler = globalHandler;

    if (g_IsInitialized) {
        return X_LINK_SUCCESS;
    }

    XLinkPlatformInit();

    // Using deprecated fields. Begin.
    int loglevel = globalHandler->loglevel;
    int protocol = globalHandler->protocol;
    // Using deprecated fields. End.

    memset((void*)globalHandler, 0, sizeof(XLinkGlobalHandler_t));
    memset(availableConnections, 0, MAX_LINKS * sizeof(Connection));

    //Using deprecated fields. Begin.
    globalHandler->loglevel = loglevel;
    globalHandler->protocol = protocol;
    //Using deprecated fields. End.

    for (int i = 0; i < MAX_LINKS; ++i) {
        availableConnections[i].id = INVALID_LINK_ID;
        freeConnectionsIds[i] = FREE_CONNECTION_FLAG;
    }

    if (XLink_isOnDeviceSide()) {
        if (sem_init(&pingSem, 0, 0)) {
            mvLog(MVLOG_ERROR, "Can't create semaphore");
        }

        Connection *connection = getNextAvailableConnection();
        if (connection == NULL)
            return X_LINK_COMMUNICATION_NOT_OPEN;

        XLINK_RET_IF(Connection_Init(connection, getNextAvailableLinkUniqueId()));

        connection->status = XLINK_CONNECTION_UP;
        connection->deviceHandle.xLinkFD = NULL;
        connection->deviceHandle.protocol = X_LINK_ANY_PROTOCOL;

        Dispatcher_Start(&connection->dispatcher, &connection->deviceHandle, 0);

        sem_wait(&pingSem);
    }

    g_IsInitialized = 1;
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
    XLINK_RET_IF(devicesArraySize == 0);
    XLINK_RET_IF(out_foundDevicesCount == 0);

    xLinkPlatformErrorCode_t rc;
    rc = XLinkPlatformFindArrayOfDevicesNames(
        state, in_deviceRequirements,
        out_foundDevicesPtr, devicesArraySize, out_foundDevicesCount);

    return parsePlatformError(rc);
}

// Called only from app - per device
XLinkError_t XLinkConnect(XLinkHandler_t* handler)
{
    XLINK_RET_IF(handler == NULL);
    if (strnlen(handler->devicePath, MAX_PATH_LENGTH) < 2) {
        mvLog(MVLOG_ERROR, "Device path is incorrect");
        return X_LINK_ERROR;
    }

    Connection* connection = getNextAvailableConnection();
    XLINK_RET_IF(Connection_Init(connection, getNextAvailableLinkUniqueId()));
    mvLog(MVLOG_DEBUG,"device name=%s glHandler=%p protocol=%d\n", handler->devicePath, glHandler, handler->protocol);

    if (Connection_Connect(connection, handler) != X_LINK_SUCCESS) {
        releaseConnection(connection);
        return X_LINK_ERROR;
    }
    handler->linkId = Connection_GetId(connection);

    return X_LINK_SUCCESS;
}

XLinkError_t XLinkBoot(deviceDesc_t* deviceDesc, const char* binaryPath)
{
    if (!XLinkPlatformBootRemote(deviceDesc, binaryPath)) {
        return X_LINK_SUCCESS;
    }

    return X_LINK_COMMUNICATION_FAIL;
}

XLinkError_t XLinkBootFirmware(deviceDesc_t* deviceDesc, const char* firmware, unsigned long length)
{
    if (!XLinkPlatformBootFirmware(deviceDesc, firmware, length)) {
        return X_LINK_SUCCESS;
    }

    return X_LINK_COMMUNICATION_FAIL;
}

XLinkError_t XLinkResetRemote(linkId_t id)
{
    Connection* connection = getLinkById(id);
    ASSERT_XLINK(connection != NULL);
    xLinkConnectionStatus_t connectionStatus = Connection_GetStatus(connection);

    XLinkError_t rc;
    if (connectionStatus == XLINK_CONNECTION_UP) {
        rc = Connection_Reset(connection);
        if (rc != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Failed to reset remote");
        }
    } else {
        mvLog(MVLOG_WARN, "Link is down, close connection to device without reset");
        XLinkPlatformCloseRemote(&connection->deviceHandle);
        return X_LINK_COMMUNICATION_NOT_OPEN;
    }

    if (releaseConnection(connection) != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Failed to release connection");
    }

    return rc;
}

XLinkError_t XLinkResetAll()
{
#if defined(NO_BOOT)
    mvLog(MVLOG_INFO, "Devices will not be restarted for this configuration (NO_BOOT)");
#else
    for (int i = 0; i < MAX_LINKS; ++i) {
        if (availableConnections[i].id != INVALID_LINK_ID) {
            Connection* connection = &availableConnections[i];
            for (int streamId = 0; streamId < XLINK_CONTROL_STREAM_ID; streamId++) {
                Connection_CloseStream(connection, streamId);
            }
            if (XLinkResetRemote(Connection_GetId(&availableConnections[i])) != X_LINK_SUCCESS) {
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

linkId_t getNextAvailableLinkUniqueId()
{
    linkId_t start = nextUniqueLinkId;
    do {
        int i;
        for (i = 0; i < MAX_LINKS; i++) {
            if (availableConnections[i].id != INVALID_LINK_ID &&
                availableConnections[i].id == nextUniqueLinkId)
                break;
        }
        if (i >= MAX_LINKS) {
            return nextUniqueLinkId;
        }
        nextUniqueLinkId++;
        if (nextUniqueLinkId == INVALID_LINK_ID) {
            nextUniqueLinkId = 0;
        }
    } while (start != nextUniqueLinkId);
    mvLog(MVLOG_ERROR, "%s():- no next available link!", __func__);
    return INVALID_LINK_ID;
}

Connection* getNextAvailableConnection() {
    for (int i = 0; i < MAX_LINKS; ++i) {
        if (freeConnectionsIds[i] == FREE_CONNECTION_FLAG) {
            freeConnectionsIds[i] = BUSY_CONNECTION_FLAG;
            return &availableConnections[i];
        }
    }

    return NULL;
}

#ifdef __PC__

XLinkError_t releaseConnection(Connection* connection) {
    XLINK_RET_IF(connection == NULL);

    linkId_t id = Connection_GetId(connection);
    freeConnectionsIds[id] = FREE_CONNECTION_FLAG;

    Connection_Clean(connection);
    connection->id = INVALID_LINK_ID;

    return X_LINK_SUCCESS;
}

XLinkError_t parsePlatformError(xLinkPlatformErrorCode_t rc) {
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
