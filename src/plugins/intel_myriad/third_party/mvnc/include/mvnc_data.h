// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _MVNC_DATA_H
#define _MVNC_DATA_H

#include "mvnc.h"
#include "XLinkPlatform.h"
#include "ncPrivateTypes.h"
#ifdef __cplusplus
extern "C"
{
#endif

XLinkProtocol_t convertProtocolToXlink(const ncDeviceProtocol_t ncProtocol);
ncDeviceProtocol_t convertProtocolToNC(const XLinkProtocol_t xLinkProtocol);

XLinkPlatform_t convertPlatformToXlink();

int copyNcDeviceDescrToXLink(
    const struct ncDeviceDescr_t *in_ncDeviceDesc, deviceDesc_t *out_deviceDesc);
int copyXLinkDeviceDescrToNc(
    const deviceDesc_t *in_DeviceDesc, struct ncDeviceDescr_t *out_ncDeviceDesc);

ncStatus_t bootDevice(deviceDesc_t* deviceDescToBoot,
    const char* mv_cmd_file_path, const bootOptions_t bootOptions);

#ifdef __cplusplus
}
#endif

#endif //_MVNC_DATA_H
