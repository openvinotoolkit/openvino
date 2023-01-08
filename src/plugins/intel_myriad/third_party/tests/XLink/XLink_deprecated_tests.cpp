// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "XLink_specific_cases.hpp"
#include "usb_boot.h"
#include <thread>

//------------------------------------------------------------------------------
//      XLinkBootTests
//------------------------------------------------------------------------------

TEST_F(XLinkBootUSBTests, CanBootConnectAndResetDevice_deprecated) {
    if (getCountSpecificDevices(X_LINK_ANY_STATE, X_LINK_USB_VSC) == 0) {
        GTEST_SKIP();
    }

    std::string firmwarePath;
    char deviceName[XLINK_MAX_NAME_SIZE] = {0};
    // Find device
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkGetDeviceName(0, deviceName, XLINK_MAX_NAME_SIZE));
    ASSERT_NO_THROW(firmwarePath = getMyriadUSBFirmwarePath(deviceName));

    // Boot it
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBootRemote(deviceName, firmwarePath.c_str()));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(kBootTimeoutSec);

    // Find booted
    char bootedDeviceName[XLINK_MAX_NAME_SIZE] = {0};
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkGetDeviceNameExtended(0, bootedDeviceName, XLINK_MAX_NAME_SIZE, DEFAULT_OPENPID));

    // Connect to device
    XLinkHandler_t handler = {};
    handler.protocol = X_LINK_USB_VSC;
    handler.devicePath = bootedDeviceName;

    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(&handler));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler.linkId));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(kResetTimeoutSec);
}
