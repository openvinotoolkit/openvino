// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __NC_EXT_H_INCLUDED__
#define __NC_EXT_H_INCLUDED__
#include <mvnc.h>
#include "XLinkPlatform.h"
#ifdef __cplusplus
extern "C"
{
#endif

/**
 * @brief Boot device with firmware without creating handler for it
 * @param devicePlatform Platform to boot
 * @param customFirmwareDir Path to directory with firmware to load. If NULL, use default
 */
MVNC_EXPORT_API ncStatus_t ncDeviceLoadFirmware(const ncDevicePlatform_t devicePlatform, const char* customFirmwareDir);

/**
 * @brief Reset all devices
 */
MVNC_EXPORT_API ncStatus_t ncDeviceResetAll();

MVNC_EXPORT_API char* ncPlatformToStr(ncDevicePlatform_t devicePlatform);
MVNC_EXPORT_API char* ncProtocolToStr(ncDeviceProtocol_t deviceProtocol);

#ifdef __cplusplus
}
#endif

#endif  // __NC_EXT_H_INCLUDED__
