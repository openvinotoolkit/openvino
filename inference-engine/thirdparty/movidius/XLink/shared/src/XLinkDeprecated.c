// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "string.h"
#include "stdlib.h"

#include "XLink.h"
#include "XLinkErrorUtils.h"
#include "XLinkPlatform.h"
#include "XLinkPublicDefines.h"
#include "XLinkPrivateFields.h"

#ifdef MVLOG_UNIT_NAME
#undef MVLOG_UNIT_NAME
#define MVLOG_UNIT_NAME xLink
#endif
#include "XLinkLog.h"
#include "XLinkStringUtils.h"

#ifdef __PC__

// ------------------------------------
// Deprecated API. Begin.
// ------------------------------------

XLinkError_t getDeviceName(int index, char* name, int nameSize, XLinkPlatform_t platform, XLinkDeviceState_t state)
{
    XLINK_RET_IF(name == NULL);
    XLINK_RET_IF(index < 0);
    XLINK_RET_IF(nameSize <= 0);

    deviceDesc_t in_deviceRequirements = { 0 };
    in_deviceRequirements.protocol = glHandler != NULL ? glHandler->protocol : USB_VSC;
    in_deviceRequirements.platform = platform;
    memset(name, 0, nameSize);

    if(index == 0) {
        deviceDesc_t deviceToBoot = { 0 };
        XLINK_RET_IF_FAIL(
            XLinkFindFirstSuitableDevice(state,
                in_deviceRequirements, &deviceToBoot));

        XLINK_RET_IF(mv_strcpy(name, nameSize, deviceToBoot.name) != EOK);
        return X_LINK_SUCCESS;
    }

    deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = { 0 };
    unsigned int numberOfDevices = 0;

    XLINK_RET_IF_FAIL(
        XLinkFindAllSuitableDevices(state, in_deviceRequirements,
                                    deviceDescArray, XLINK_MAX_DEVICES, &numberOfDevices));

    XLINK_RET_ERR_IF(
        (unsigned int)index >= numberOfDevices,
        X_LINK_DEVICE_NOT_FOUND);

    XLINK_RET_IF(mv_strcpy(name, nameSize, deviceDescArray[index].name) != EOK);
    return X_LINK_SUCCESS;
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
    XLINK_RET_IF(deviceName == NULL);
    XLINK_RET_IF(binaryPath == NULL);

    deviceDesc_t deviceDesc = { 0 };
    deviceDesc.protocol = glHandler != NULL ? glHandler->protocol : USB_VSC;
    XLINK_RET_IF(mv_strcpy(deviceDesc.name, XLINK_MAX_NAME_SIZE, deviceName) != EOK);

    return XLinkBoot(&deviceDesc, binaryPath);
}

XLinkError_t XLinkDisconnect(linkId_t id)
{
    xLinkDesc_t* link = getLinkById(id);
    XLINK_RET_IF(link == NULL);

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

#endif // __PC__

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
