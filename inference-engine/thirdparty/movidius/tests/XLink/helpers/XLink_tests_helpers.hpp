// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "gtest/gtest.h"
#include "XLink.h"

#define _XLINK_ENABLE_PRIVATE_INCLUDE_
#include <XLinkPrivateDefines.h>

//------------------------------------------------------------------------------
//      Defines
//------------------------------------------------------------------------------

#if (defined(_WIN32) || defined(_WIN64))
    static constexpr char PCIE_NAME_SUBSTR[] = "mxlink";
    static constexpr char FIRMWARE_SUBFOLDER[] = "./";
#else
    static constexpr char PCIE_NAME_SUBSTR[] = "mxlk";
    static constexpr char FIRMWARE_SUBFOLDER[] = "./lib/";
#endif

//------------------------------------------------------------------------------
//      Macros
//------------------------------------------------------------------------------

#define ASSERT_NO_ERROR(call)   ASSERT_EQ(call, 0)

//------------------------------------------------------------------------------
//      class XLinkTestsHelpersBoot
//------------------------------------------------------------------------------

class XLinkTestsHelpersBoot {
protected:
    // Boot
    void bootUSBDevice(deviceDesc_t& deviceDesc, deviceDesc_t& bootedDeviceDesc);

    // Connect
    void connectToBootedUSB(deviceDesc_t& bootedDeviceDesc, XLinkHandler_t* handler);

    // Close
    void closeUSBDevice(XLinkHandler_t* handler);
    void connectUSBAndClose(deviceDesc_t& bootedDeviceDesc);

    // Firmware
    std::string getMyriadStickFirmwarePath(const std::string& devAddr);
    std::string getMyriadPCIeFirmware();

    // Data helpers
    static void copyDeviceDescr(deviceDesc_t *destDeviceDescr,
                                const deviceDesc_t sourceDeviceDescr);

    // Search
    static XLinkError_t findDeviceOnIndex(const int index,
                                          const XLinkDeviceState_t deviceState,
                                          const deviceDesc_t in_deviceRequirements,
                                          deviceDesc_t *out_foundDevicesPtr);


    static int getAmountOfDevices(const XLinkDeviceState_t state = X_LINK_ANY_STATE,
                                  const XLinkProtocol_t deviceProtocol = X_LINK_ANY_PROTOCOL,
                                  const XLinkPlatform_t devicePlatform = X_LINK_ANY_PLATFORM);
};

//------------------------------------------------------------------------------
//      class XLinkTestsHelpersOneDeviceBoot
//------------------------------------------------------------------------------
/**
 * @brief For tests which require one isolated device, only handle available
 */
class XLinkTestsHelpersOneUSBDevice : public XLinkTestsHelpersBoot {
public:
    void bootUSBDevice();
    void closeUSBDevice();

    XLinkHandler_t* handler = nullptr;

private:
    deviceDesc_t _deviceDesc = {};
    deviceDesc_t _bootedDesc = {};

};
