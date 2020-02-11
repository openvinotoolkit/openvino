// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>
#include "mvnc_data.h"
#define MVLOG_UNIT_NAME ncTool
#include "XLinkLog.h"
#include "XLinkStringUtils.h"
#include "mvnc_tool.h"

XLinkProtocol_t convertProtocolToXlink(
    const ncDeviceProtocol_t ncProtocol) {
    switch (ncProtocol) {
        case NC_ANY_PROTOCOL: return X_LINK_ANY_PROTOCOL;
        case NC_USB:          return X_LINK_USB_VSC;
        case NC_PCIE:         return X_LINK_PCIE;
        default:              return X_LINK_ANY_PROTOCOL;
    }
}

ncDeviceProtocol_t convertProtocolToNC(
    const XLinkProtocol_t xLinkProtocol) {
    switch (xLinkProtocol) {
        case X_LINK_ANY_PROTOCOL:   return NC_ANY_PROTOCOL;
        case X_LINK_USB_VSC:        return NC_USB;
        case X_LINK_PCIE:           return NC_PCIE;
        default:
            mvLog(MVLOG_WARN, "This convertation not supported, set to ANY_PROTOCOL");
            return NC_ANY_PROTOCOL;
    }
}

XLinkPlatform_t convertPlatformToXlink(
    const ncDevicePlatform_t ncProtocol) {
    switch (ncProtocol) {
        case NC_ANY_PLATFORM: return X_LINK_ANY_PLATFORM;
        case NC_MYRIAD_2:     return X_LINK_MYRIAD_2;
        case NC_MYRIAD_X:     return X_LINK_MYRIAD_X;
        default:           return X_LINK_ANY_PLATFORM;
    }
}

ncDevicePlatform_t convertPlatformToNC(
    const XLinkPlatform_t xLinkProtocol) {
    switch (xLinkProtocol) {
        case X_LINK_ANY_PLATFORM:   return NC_ANY_PLATFORM;
        case X_LINK_MYRIAD_2:       return NC_MYRIAD_2;
        case X_LINK_MYRIAD_X:       return NC_MYRIAD_X;
        default:
            mvLog(MVLOG_WARN, "This convertation not supported, set to NC_ANY_PLATFORM");
            return NC_ANY_PLATFORM;
    }
}

int copyNcDeviceDescrToXLink(const struct ncDeviceDescr_t *in_ncDeviceDesc,
                                    deviceDesc_t *out_deviceDesc) {
    CHECK_HANDLE_CORRECT(in_ncDeviceDesc);
    CHECK_HANDLE_CORRECT(out_deviceDesc);

    out_deviceDesc->protocol = convertProtocolToXlink(in_ncDeviceDesc->protocol);
    out_deviceDesc->platform = convertPlatformToXlink(in_ncDeviceDesc->platform);
    mv_strncpy(out_deviceDesc->name, XLINK_MAX_NAME_SIZE, in_ncDeviceDesc->name, XLINK_MAX_NAME_SIZE - 1);

    return NC_OK;
}

int copyXLinkDeviceDescrToNc(const deviceDesc_t *in_DeviceDesc,
                                    struct ncDeviceDescr_t *out_ncDeviceDesc) {
    CHECK_HANDLE_CORRECT(in_DeviceDesc);
    CHECK_HANDLE_CORRECT(out_ncDeviceDesc);

    out_ncDeviceDesc->protocol = convertProtocolToNC(in_DeviceDesc->protocol);
    out_ncDeviceDesc->platform = convertPlatformToNC(in_DeviceDesc->platform);
    mv_strncpy(out_ncDeviceDesc->name, XLINK_MAX_NAME_SIZE, in_DeviceDesc->name, XLINK_MAX_NAME_SIZE - 1);

    return NC_OK;
}