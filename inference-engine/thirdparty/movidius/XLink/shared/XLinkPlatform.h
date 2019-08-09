// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINK_LINKPLATFORM_H
#define _XLINK_LINKPLATFORM_H

#define _XLINK_ENABLE_PRIVATE_INCLUDE_
#include "XLinkPrivateDefines.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_POOLS_ALLOC 32
#define PACKET_LENGTH (64*1024)

int XLinkWrite(xLinkDeviceHandle_t* deviceHandle, void* data, int size, unsigned int timeout);
int XLinkRead(xLinkDeviceHandle_t* deviceHandle, void* data, int size, unsigned int timeout);
int XLinkPlatformConnect(const char* devPathRead, const char* devPathWrite,
    XLinkProtocol_t protocol, void** fd);
void XLinkPlatformInit();

/**
 * @brief Return Myriad device name on index
 */
int XLinkPlatformFindDeviceName(int index,
    XLinkDeviceState_t state,
    deviceDesc_t* in_deviceRequirements,
    deviceDesc_t* out_foundDevice);

int XLinkPlatformIsDescriptionValid(deviceDesc_t *in_deviceDesc);

int XLinkPlatformToPid(const XLinkPlatform_t platform);
XLinkPlatform_t XLinkPlatformPidToPlatform(const int pid);

int XLinkPlatformBootRemote(deviceDesc_t* deviceDesc,
                            const char* binaryPath);
int XLinkPlatformCloseRemote(xLinkDeviceHandle_t* deviceHandle);

void* allocateData(uint32_t size, uint32_t alignment);
void deallocateData(void* ptr,uint32_t size, uint32_t alignment);

typedef enum xLinkPlatformErrorCode {
    X_LINK_PLATFORM_SUCCESS = 0,
    X_LINK_PLATFORM_DEVICE_NOT_FOUND = -1,
    X_LINK_PLATFORM_ERROR = -2,
    X_LINK_PLATFORM_TIMEOUT = -3,
    X_LINK_PLATFORM_DRIVER_NOT_LOADED = -4
} xLinkPlatformErrorCode_t;

#ifdef __cplusplus
}
#endif

#endif

/* end of include file */
