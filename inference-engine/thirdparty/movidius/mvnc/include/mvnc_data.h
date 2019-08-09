#ifndef _MVNC_DATA_H
#define _MVNC_DATA_H

#include "mvnc.h"
#include "XLinkPlatform.h"
#ifdef __cplusplus
extern "C"
{
#endif

XLinkProtocol_t convertProtocolToXlink(const ncDeviceProtocol_t ncProtocol);
ncDeviceProtocol_t convertProtocolToNC(const XLinkProtocol_t xLinkProtocol);

XLinkPlatform_t convertPlatformToXlink(const ncDevicePlatform_t ncProtocol);
ncDevicePlatform_t convertPlatformToNC(const XLinkPlatform_t xLinkProtocol);

int copyNcDeviceDescrToXLink(
    const struct ncDeviceDescr_t *in_ncDeviceDesc, deviceDesc_t *out_deviceDesc);
int copyXLinkDeviceDescrToNc(
    const deviceDesc_t *in_DeviceDesc, struct ncDeviceDescr_t *out_ncDeviceDesc);

#ifdef __cplusplus
}
#endif

#endif //_MVNC_DATA_H
