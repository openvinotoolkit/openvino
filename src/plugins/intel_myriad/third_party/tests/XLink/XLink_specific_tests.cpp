// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cases/XLink_specific_cases.hpp"
#include <thread>

//------------------------------------------------------------------------------
//      XLinkFindFirstSuitableDeviceUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkFindFirstSuitableDeviceUSBTests, CanFindBootedDeviceByName) {
    if (getCountSpecificDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t deviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};

    bootDevice(deviceDesc, bootedDeviceDesc);

    deviceDesc_t foundDeviceDescr = {};
    EXPECT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, bootedDeviceDesc, &foundDeviceDescr));

    EXPECT_TRUE(strcmp(bootedDeviceDesc.name, foundDeviceDescr.name) == 0);

    connectAndCloseDevice(bootedDeviceDesc);
}

//------------------------------------------------------------------------------
//      XLinkBootUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkBootUSBTests, DeviceNameChangedAfterBoot) {
    if (getCountSpecificDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t unbootedDeviceDescr = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    // Get device name
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &unbootedDeviceDescr));
    std::string firmwarePath;
    ASSERT_NO_THROW(firmwarePath = getMyriadFirmwarePath(unbootedDeviceDescr));

    // Boot device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBoot(&unbootedDeviceDescr, firmwarePath.c_str()));
    std::this_thread::sleep_for(kBootTimeoutSec);

    // Booted device appear
    deviceDesc_t bootedDeviceDesc = {};
    EXPECT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &bootedDeviceDesc));

    // The device is not in unbooted and booted list at the same time
    deviceDesc_t foundDeviceDesc = {};
    EXPECT_EQ(X_LINK_DEVICE_NOT_FOUND,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, unbootedDeviceDescr, &foundDeviceDesc));

    connectAndCloseDevice(bootedDeviceDesc);
}


//------------------------------------------------------------------------------
//      XLinkFindPCIEDeviceTests
//------------------------------------------------------------------------------

TEST_F(XLinkPCIEDeviceTests, CannotFindSameDeviceTwice) {
    if (getCountSpecificDevices(X_LINK_ANY_STATE, X_LINK_PCIE) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceRequirements = {};
    deviceRequirements.protocol = X_LINK_PCIE;
    deviceRequirements.platform = X_LINK_ANY_PLATFORM;

    deviceDesc_t deviceDescFirst = {};
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, deviceRequirements, &deviceDescFirst));

    // On index 1
    deviceDesc_t deviceDescSecond = {};
    XLinkError_t rc = findDeviceOnIndex(
            1, X_LINK_ANY_STATE, deviceRequirements, &deviceDescSecond);

    if (rc != X_LINK_DEVICE_NOT_FOUND) {
        ASSERT_EQ(rc, X_LINK_SUCCESS);
        ASSERT_TRUE(strstr(deviceDescFirst.name, PCIE_NAME_SUBSTR) != nullptr);
        ASSERT_TRUE(strstr(deviceDescSecond.name, PCIE_NAME_SUBSTR) != nullptr);
        ASSERT_TRUE(strcmp(deviceDescFirst.name, deviceDescSecond.name) != 0);
    }
}

/**
 * This is real test for two multi-device case, require two PCIe cards
 * Boot second and expect that first will be unbooted, second booted
 */
TEST_F(XLinkPCIEDeviceTests, DISABLED_CanFindFirstDeviceAfterBootSecond) {
    if (getCountSpecificDevices(X_LINK_ANY_STATE, X_LINK_PCIE) == 0)
        GTEST_SKIP();

    // TODO Add check that there two devices
    deviceDesc_t deviceRequirements = {};
    deviceRequirements.protocol = X_LINK_PCIE;
    deviceRequirements.platform = X_LINK_ANY_PLATFORM;

    // Find first device
    deviceDesc_t firstDeviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS, findDeviceOnIndex(
            0, X_LINK_ANY_STATE, deviceRequirements, &firstDeviceDesc));

    // Find second device
    deviceDesc_t secondDeviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS, findDeviceOnIndex(
            1, X_LINK_ANY_STATE, deviceRequirements, &secondDeviceDesc));

    // Boot second device
    std::string firmwarePath;
    ASSERT_NO_THROW(firmwarePath = getMyriadFirmwarePath(deviceRequirements));

    EXPECT_EQ(X_LINK_SUCCESS, XLinkBoot(&secondDeviceDesc, firmwarePath.c_str()));
    std::this_thread::sleep_for(kBootTimeoutSec);

    // Check that first still in unbooted state
    deviceDesc_t firstDeviceDescAfter = {};
    firstDeviceDescAfter.protocol = X_LINK_PCIE;
    firstDeviceDescAfter.platform = X_LINK_ANY_PLATFORM;


    EXPECT_EQ(X_LINK_SUCCESS, findDeviceOnIndex(
            0, X_LINK_UNBOOTED, firstDeviceDesc, &firstDeviceDescAfter));

    // Check that second device now in booted state
    deviceDesc_t secondDeviceDescAfter = {};
    secondDeviceDescAfter.protocol = X_LINK_PCIE;
    secondDeviceDescAfter.platform = X_LINK_ANY_PLATFORM;

    EXPECT_EQ(X_LINK_SUCCESS, findDeviceOnIndex(
            0, X_LINK_BOOTED, secondDeviceDesc, &secondDeviceDescAfter));

    // TODO Move it to separate function
    // Close second device
    XLinkHandler_t handler = {0};
    handler.protocol = secondDeviceDesc.protocol;
    handler.devicePath = secondDeviceDesc.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(&handler));

    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler.linkId));
    std::this_thread::sleep_for(kResetTimeoutSec);
}
