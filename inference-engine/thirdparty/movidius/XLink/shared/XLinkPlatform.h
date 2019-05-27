/*
* Copyright 2018-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

#ifndef _XLINK_LINKPLATFORM_H
#define _XLINK_LINKPLATFORM_H
#include <stdint.h>
#include "XLinkPublicDefines.h"
#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_POOLS_ALLOC 32
#define PACKET_LENGTH (64*1024)

#define MAX_LINKS 32

int XLinkWrite(void* fd, void* data, int size, unsigned int timeout);
int XLinkRead(void* fd, void* data, int size, unsigned int timeout);
int XLinkPlatformConnect(const char* devPathRead,
                           const char* devPathWrite, void** fd);
int XLinkPlatformInit(XLinkProtocol_t protocol, int loglevel);

/**
 * @brief      Return Myriad device name on index
 * @param[in]  index Index of device in list of all Myriad devices
 * @param[out] name device name, which would be found
 */
int XLinkPlatformGetDeviceName(int index,
                                char* name,
                                int nameSize);

/**
 * @brief      Returning Myriad device suitable for the parameters
 * @param[in]  index Device index in list of suitable (matches pid argument) devices
 * @param[out] name device name, which would be found
 * @param[in] pid  0x2485 for MX, 0x2150 for M2, 0 for any, -1 for any not booted
 */
int XLinkPlatformGetDeviceNameExtended(int index,
                                char* name,
                                int nameSize,
                                int pid);

int XLinkPlatformBootRemote(const char* deviceName,
							const char* binaryPath);
int XLinkPlatformCloseRemote(void *fd);

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
