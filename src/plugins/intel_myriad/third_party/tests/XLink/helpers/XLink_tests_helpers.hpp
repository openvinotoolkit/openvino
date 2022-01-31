// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "XLink.h"
#include "XLinkPrivateDefines.h"

#include <gtest/gtest.h>
#include <chrono>
#include <string>

//------------------------------------------------------------------------------
//      Defines
//------------------------------------------------------------------------------

using std_seconds = std::chrono::seconds;

#if (defined(_WIN32) || defined(_WIN64))
    static constexpr char PCIE_NAME_SUBSTR[] = "mxlink";
    static constexpr char FIRMWARE_SUBFOLDER[] = "./";
#else
    static constexpr char PCIE_NAME_SUBSTR[] = "mxlk";
    static constexpr char FIRMWARE_SUBFOLDER[] = "./lib/";
#endif

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------

constexpr std::chrono::seconds operator "" _sec(unsigned long long s)
{
    return std::chrono::seconds(s);
}

//------------------------------------------------------------------------------
//      class XLinkTestsHelper
//------------------------------------------------------------------------------

class XLinkTestsHelper {
public:
    const std_seconds kBootTimeoutSec = 2_sec;
    const std_seconds kResetTimeoutSec = 5_sec;

    const std::string kUSBMyriadX = "ma2480";
    const std::string kUSBMyriad2 = "ma2450";

    // Device management
    void bootDevice(const deviceDesc_t& in_deviceDesc, deviceDesc_t& out_bootedDeviceDesc);

    void connectToDevice(deviceDesc_t& in_bootedDeviceDesc, XLinkHandler_t* out_handler);
    void closeDevice(XLinkHandler_t* handler);

    void connectAndCloseDevice(deviceDesc_t& in_bootedDeviceDesc);

    // Firmware
    std::string getMyriadUSBFirmwarePath(const std::string& deviceName);
    std::string getMyriadFirmwarePath(const deviceDesc_t& in_deviceDesc);

    // Device searching
    XLinkError_t findDeviceOnIndex(const int index,
                                   const XLinkDeviceState_t deviceState,
                                   const deviceDesc_t in_deviceRequirements,
                                   deviceDesc_t *out_foundDevicesPtr);


    static int getCountSpecificDevices(const XLinkDeviceState_t state = X_LINK_ANY_STATE,
                                const XLinkProtocol_t deviceProtocol = X_LINK_ANY_PROTOCOL,
                                const XLinkPlatform_t devicePlatform = X_LINK_ANY_PLATFORM);
};
