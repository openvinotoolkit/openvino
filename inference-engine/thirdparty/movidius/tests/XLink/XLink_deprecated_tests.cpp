// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cases/XLink_tests_case.hpp"
#include "usb_boot.h"
#include <thread>

//------------------------------------------------------------------------------
//      XLinkBootTests
//------------------------------------------------------------------------------

TEST_F(XLinkBootUSBTests, CanBootConnectAndResetDevice_deprecated) {
    if (getAmountOfDevices(X_LINK_ANY_STATE, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    std::string firmwarePath;
    char deviceName[XLINK_MAX_NAME_SIZE] = {0};
    // Find device
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkGetDeviceName(0, deviceName, XLINK_MAX_NAME_SIZE));
    ASSERT_NO_THROW(firmwarePath = getMyriadStickFirmwarePath(deviceName));

    // Boot it
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBootRemote(deviceName, firmwarePath.c_str()));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Find booted
    char bootedDeviceName[XLINK_MAX_NAME_SIZE] = {0};
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkGetDeviceNameExtended(0, bootedDeviceName, XLINK_MAX_NAME_SIZE, DEFAULT_OPENPID));

    // Connect to device
    XLinkHandler_t handler = {};
    handler.protocol = X_LINK_USB_VSC;
    handler.devicePath = bootedDeviceName;

    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(&handler));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler.linkId));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(10));
}
