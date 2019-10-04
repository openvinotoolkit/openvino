// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string.h"
#include "stdlib.h"

#include "XLink.h"
#include "XLinkTool.h"
#include "XLinkPlatform.h"
#include "XLinkPublicDefines.h"
#include "XLinkPrivateFields.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "mvLog.h"
#include "mvStringUtils.h"

// ------------------------------------
// Deprecated API. Begin.
// ------------------------------------

XLinkError_t getDeviceName(int index, char* name, int nameSize, XLinkPlatform_t platform, XLinkDeviceState_t state)
{
    ASSERT_X_LINK(name != NULL);
    ASSERT_X_LINK(index >= 0);
    ASSERT_X_LINK(nameSize >= 0 && nameSize <= XLINK_MAX_NAME_SIZE);

    deviceDesc_t in_deviceRequirements = {};
    in_deviceRequirements.protocol = glHandler != NULL ? glHandler->protocol : USB_VSC;
    in_deviceRequirements.platform = platform;
    memset(name, 0, nameSize);

    if(index == 0)
    {
        deviceDesc_t deviceToBoot = {};
        XLinkError_t rc =
            XLinkFindFirstSuitableDevice(state, in_deviceRequirements, &deviceToBoot);
        if(rc != X_LINK_SUCCESS)
        {
            return rc;
        }

        return mv_strcpy(name, nameSize, deviceToBoot.name) == EOK ? X_LINK_SUCCESS : X_LINK_ERROR;
    }
    else
    {
        deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {};
        unsigned int numberOfDevices = 0;
        XLinkError_t rc =
            XLinkFindAllSuitableDevices(state, in_deviceRequirements,
                                        deviceDescArray, XLINK_MAX_DEVICES, &numberOfDevices);
        if(rc != X_LINK_SUCCESS)
        {
            return rc;
        }

        if((unsigned int)index >= numberOfDevices)
        {
            return X_LINK_DEVICE_NOT_FOUND;
        }

        return mv_strcpy(name, nameSize, deviceDescArray[index].name) == EOK ? X_LINK_SUCCESS : X_LINK_ERROR;
    }
}

XLinkError_t XLinkGetDeviceName(int index, char* name, int nameSize)
{
    return getDeviceName(index, name, nameSize, X_LINK_ANY_PLATFORM, X_LINK_ANY_STATE);
}

XLinkError_t XLinkGetDeviceNameExtended(int index, char* name, int nameSize, int pid)
{
    XLinkDeviceState_t state = XLinkPlatformPidToState(pid);
    XLinkPlatform_t platform = XLinkPlatformPidToPlatform(pid);

    return getDeviceName(index, name, nameSize, platform, state);
}

XLinkError_t XLinkBootRemote(const char* deviceName, const char* binaryPath)
{
    ASSERT_X_LINK(deviceName != NULL);
    ASSERT_X_LINK(binaryPath != NULL);

    deviceDesc_t deviceDesc = {};
    deviceDesc.protocol = glHandler != NULL ? glHandler->protocol : USB_VSC;
    mv_strcpy(deviceDesc.name, XLINK_MAX_NAME_SIZE, deviceName);

    return XLinkBoot(&deviceDesc, binaryPath);
}

XLinkError_t XLinkDisconnect(linkId_t id)
{
    xLinkDesc_t* link = getLinkById(id);
    ASSERT_X_LINK(link != NULL);

    link->hostClosedFD = 1;
    return XLinkPlatformCloseRemote(&link->deviceHandle);
}

XLinkError_t XLinkGetAvailableStreams(linkId_t id)
{
    (void)id;
    return X_LINK_NOT_IMPLEMENTED;
}

XLinkError_t XLinkWriteDataWithTimeout(streamId_t streamId, const uint8_t* buffer,
                                       int size, unsigned int timeout)
{
    (void)timeout;
    return XLinkWriteData(streamId, buffer, size);
}

XLinkError_t XLinkReadDataWithTimeOut(streamId_t streamId, streamPacketDesc_t** packet, unsigned int timeout)
{
    (void)timeout;
    return XLinkReadData(streamId, packet);
}

XLinkError_t XLinkAsyncWriteData()
{
    return X_LINK_NOT_IMPLEMENTED;
}

XLinkError_t XLinkSetDeviceOpenTimeOutMsec(unsigned int msec)  {
    (void)msec;
    return X_LINK_SUCCESS;
}

XLinkError_t XLinkSetCommonTimeOutMsec(unsigned int msec) {
    (void)msec;
    return X_LINK_SUCCESS;
}

// ------------------------------------
// Deprecated API. End.
// ------------------------------------

// ------------------------------------
// Public helpers. Begin.
// ------------------------------------

const char* XLinkErrorToStr(XLinkError_t rc) {
    switch (rc) {
        case X_LINK_SUCCESS:
            return "X_LINK_SUCCESS";
        case X_LINK_ALREADY_OPEN:
            return "X_LINK_ALREADY_OPEN";
        case X_LINK_COMMUNICATION_NOT_OPEN:
            return "X_LINK_COMMUNICATION_NOT_OPEN";
        case X_LINK_COMMUNICATION_FAIL:
            return "X_LINK_COMMUNICATION_FAIL";
        case X_LINK_COMMUNICATION_UNKNOWN_ERROR:
            return "X_LINK_COMMUNICATION_UNKNOWN_ERROR";
        case X_LINK_DEVICE_NOT_FOUND:
            return "X_LINK_DEVICE_NOT_FOUND";
        case X_LINK_TIMEOUT:
            return "X_LINK_TIMEOUT";
        case X_LINK_OUT_OF_MEMORY:
            return "X_LINK_OUT_OF_MEMORY";
        case X_LINK_ERROR:
        default:
            return "X_LINK_ERROR";
    }
}

// ------------------------------------
// Public helpers. End.
// ------------------------------------
