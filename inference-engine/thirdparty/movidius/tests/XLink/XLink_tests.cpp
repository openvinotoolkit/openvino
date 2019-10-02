// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cases/XLink_tests_case.hpp"
#include <thread>

//------------------------------------------------------------------------------
//      XLinkBootUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkBootUSBTests, CanBootConnectAndResetDevice) {
    if (getAmountOfDevices(X_LINK_ANY_STATE, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    std::string firmwarePath;

    deviceDesc_t deviceDesc = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    // Find device
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &deviceDesc));
    ASSERT_NO_THROW(firmwarePath = getMyriadStickFirmwarePath(deviceDesc.name));

    // Boot it
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBoot(&deviceDesc, firmwarePath.c_str()));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Find booted
    deviceDesc_t bootedDeviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &bootedDeviceDesc));

    // Connect to device
    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    handler->protocol = bootedDeviceDesc.protocol;
    handler->devicePath = bootedDeviceDesc.name;

    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    free(handler);
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

// TODO Add boot test for PCIe

//------------------------------------------------------------------------------
//      XLinkOpenStreamUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkOpenStreamUSBTests, CanOpenAndCloseStream) {
    if (getAmountOfDevices(X_LINK_ANY_STATE, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    streamId_t stream = XLinkOpenStream(handler->linkId, "mySuperStream", 1024);
    ASSERT_NE(INVALID_STREAM_ID, stream);
    ASSERT_NE(INVALID_STREAM_ID_OUT_OF_MEMORY, stream);
    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream));
}

// CannotOpenStreamMoreThanMemoryOnDevice
TEST_F(XLinkOpenStreamUSBTests, CannotOpenStreamMoreThanMemoryOnDevice) {
    if (getAmountOfDevices(X_LINK_ANY_STATE, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    const int _512MB = 512 * 1024 * 1024;
    streamId_t stream = XLinkOpenStream(handler->linkId, "mySuperStream", _512MB);
    ASSERT_EQ(INVALID_STREAM_ID_OUT_OF_MEMORY, stream);
}

// FIXME: the test doesn't work
// TODO: is it correct behavior, should we accept the same names
TEST_F(XLinkOpenStreamUSBTests, DISABLED_CannotOpenTwoStreamsWithTheSameName) {
    if (getAmountOfDevices(X_LINK_ANY_STATE, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    const int _1KB = 1 * 1024;
    const char streamName[] = "mySuperStream";
    streamId_t stream0 = XLinkOpenStream(handler->linkId, streamName, _1KB);
    ASSERT_NE(INVALID_STREAM_ID, stream0);

    streamId_t stream1 = XLinkOpenStream(handler->linkId, streamName, _1KB);
    ASSERT_EQ(INVALID_STREAM_ID, stream1);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream0));
}

// FIXME: XLinkOpenStream doesn't allocate any memory on device
TEST_F(XLinkOpenStreamUSBTests, DISABLED_CannotOpenStreamsMoreThanMemoryOnDevice) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    const int _256MB = 256 * 1024 * 1024;
    streamId_t stream0 = XLinkOpenStream(handler->linkId, "mySuperStream0", _256MB);
    ASSERT_NE(INVALID_STREAM_ID, stream0);

    streamId_t stream1 = XLinkOpenStream(handler->linkId, "mySuperStream1", _256MB);
    ASSERT_EQ(INVALID_STREAM_ID, stream1);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream0));
    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream1));
}

//------------------------------------------------------------------------------
//      XLinkFindFirstSuitableDeviceUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkFindFirstSuitableDeviceUSBTests, ReturnAnyDeviceName) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceDesc = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, in_deviceDesc, &deviceDesc));
    ASSERT_TRUE(strlen(deviceDesc.name) > 2);
    ASSERT_EQ(deviceDesc.protocol, X_LINK_USB_VSC);
    ASSERT_NE(deviceDesc.platform, X_LINK_ANY_PLATFORM);
}

TEST_F(XLinkFindFirstSuitableDeviceUSBTests, ReturnCorrectM2DeviceName) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC, X_LINK_MYRIAD_2) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceDesc = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_MYRIAD_2;

    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, in_deviceDesc, &deviceDesc));
    EXPECT_TRUE(strstr(deviceDesc.name, "ma2450") != nullptr);
    EXPECT_EQ(deviceDesc.protocol, X_LINK_USB_VSC);
    EXPECT_EQ(deviceDesc.platform, X_LINK_MYRIAD_2);
}

TEST_F(XLinkFindFirstSuitableDeviceUSBTests, ReturnCorrectMXDeviceName) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC, X_LINK_MYRIAD_X) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceDesc = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_MYRIAD_X;

    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, in_deviceDesc, &deviceDesc));
    EXPECT_TRUE(strstr(deviceDesc.name, "ma2480") != nullptr);
    EXPECT_EQ(deviceDesc.protocol, X_LINK_USB_VSC);
    EXPECT_EQ(deviceDesc.platform, X_LINK_MYRIAD_X);
}

TEST_F(XLinkFindFirstSuitableDeviceUSBTests, ReturnCorrectBootedDeviceName) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceDesc = {};
    deviceDesc_t expectedBooted = {};
    bootUSBDevice(deviceDesc, expectedBooted);

    deviceDesc_t bootedDeviceDescr = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    EXPECT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &bootedDeviceDescr));
    EXPECT_TRUE(strstr(bootedDeviceDescr.name, "ma2480") == nullptr);
    EXPECT_TRUE(strstr(bootedDeviceDescr.name, "ma2450") == nullptr);
    EXPECT_TRUE(strcmp(expectedBooted.name, bootedDeviceDescr.name) == 0);
    EXPECT_EQ(bootedDeviceDescr.protocol, X_LINK_USB_VSC);

//      TODO #-20852
//    EXPECT_NE(bootedDeviceDescr.platform, X_LINK_ANY_PLATFORM);

    connectUSBAndClose(bootedDeviceDescr);
}

TEST_F(XLinkFindFirstSuitableDeviceUSBTests, CanFindBootedDeviceByName) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};

    bootUSBDevice(deviceDesc, bootedDeviceDesc);

    deviceDesc_t foundDeviceDescr = {};
    EXPECT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, bootedDeviceDesc, &foundDeviceDescr));

    EXPECT_TRUE(strcmp(bootedDeviceDesc.name, foundDeviceDescr.name) == 0);

    connectUSBAndClose(bootedDeviceDesc);
}

//------------------------------------------------------------------------------
//      XLinkFindAllSuitableDevicesTests
//------------------------------------------------------------------------------

TEST_F(XLinkFindAllSuitableDevicesTests, CanFindMoreThenTwoDeviceAnyState) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) < 2)
        GTEST_SKIP();

    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {{}};
    unsigned int numOfFoundDevices = 0;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkFindAllSuitableDevices(
                                X_LINK_ANY_STATE, in_deviceDesc,
                                deviceDescArray, XLINK_MAX_DEVICES, &numOfFoundDevices));

    ASSERT_EQ(numOfFoundDevices, getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC));
    ASSERT_EQ(numOfFoundDevices,
            getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC, X_LINK_MYRIAD_2) +
            getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC, X_LINK_MYRIAD_X));
}

TEST_F(XLinkFindAllSuitableDevicesTests, CanFindTwoDeviceDifferentState) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) < 2)
        GTEST_SKIP();

    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    // Find & boot one device
    deviceDesc_t firstDeviceDescr = {};
    deviceDesc_t bootedDeviceDescr = {};
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &firstDeviceDescr));
    bootUSBDevice(firstDeviceDescr, bootedDeviceDescr);

    deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {{}};
    unsigned int numOfFoundDevices = 0;
    EXPECT_EQ(X_LINK_SUCCESS, XLinkFindAllSuitableDevices(
            X_LINK_ANY_STATE, in_deviceDesc,
            deviceDescArray, XLINK_MAX_DEVICES, &numOfFoundDevices));

    bool foundBootedDevice = false;
    for (int i = 0; i < numOfFoundDevices; ++i) {
        if (deviceDescArray[i].platform == X_LINK_ANY_PLATFORM)
            foundBootedDevice = true;
    }

    EXPECT_GE(numOfFoundDevices, 2);
    EXPECT_TRUE(foundBootedDevice);
    EXPECT_EQ(numOfFoundDevices, getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) +
                                    getAmountOfDevices(X_LINK_BOOTED, X_LINK_USB_VSC));
    connectUSBAndClose(bootedDeviceDescr);
}

//------------------------------------------------------------------------------
//      XLinkResetAll
//------------------------------------------------------------------------------
/**
 * @brief XLinkResetAll function should reset all booted devices
 */
TEST_F(XLinkResetAllUSBTests, DISABLED_ResetBootedDevice) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};
    bootUSBDevice(deviceDesc, bootedDeviceDesc);

    // Try to reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetAll());
    std::this_thread::sleep_for(std::chrono::seconds(2));

    deviceDesc_t in_deviceDesc = {};
    // No one booted device should be found
    deviceDesc_t afterResetBootedDescr = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND,
              XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &afterResetBootedDescr));
}

//------------------------------------------------------------------------------
//      XLinkConnectUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkConnectUSBTests, ConnectToDevice) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    deviceDesc_t deviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};
    bootUSBDevice(deviceDesc, bootedDeviceDesc);

    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    handler->protocol = bootedDeviceDesc.protocol;
    handler->devicePath = bootedDeviceDesc.name;

    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
    // FIXME: need to find a way to avoid this sleep
    std::this_thread::sleep_for(std::chrono::seconds(1));

    closeUSBDevice(handler);
}

//------------------------------------------------------------------------------
//      XLinkBootUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkBootUSBTests, USBDeviceNameChangedAfterBoot) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    deviceDesc_t unbootedDeviceDescr = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    std::string firmwarePath;

    // Get device name
    ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &unbootedDeviceDescr));
    ASSERT_NO_THROW(firmwarePath = getMyriadStickFirmwarePath(unbootedDeviceDescr.name));

    // Boot device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBoot(&unbootedDeviceDescr, firmwarePath.c_str()));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Booted device appear
    deviceDesc_t bootedDeviceDesc = {};
    EXPECT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &bootedDeviceDesc));

    // The device is not in unbooted and booted list at the same time
    deviceDesc_t foundDeviceDesc = {};
    EXPECT_EQ(X_LINK_DEVICE_NOT_FOUND,
            XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, unbootedDeviceDescr, &foundDeviceDesc));

    connectUSBAndClose(bootedDeviceDesc);
}

//------------------------------------------------------------------------------
//      XLinkNullPtrTests
//------------------------------------------------------------------------------

TEST_F(XLinkNullPtrTests, XLinkInitialize) {
    ASSERT_EQ(XLinkInitialize(nullptr), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkConnect) {
    ASSERT_EQ(XLinkConnect(nullptr), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkOpenStream) {
    ASSERT_EQ(XLinkOpenStream(0, nullptr, 0), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkCloseStream) {
    ASSERT_EQ(XLinkCloseStream(0), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkFindFirstSuitableDevice) {
    ASSERT_EQ(XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, {}, nullptr), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkFindAllSuitableDevices) {
    ASSERT_EQ(XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, {}, nullptr, -1, nullptr),
            X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkWriteData) {
    ASSERT_EQ(XLinkWriteData(0, nullptr, 0), X_LINK_ERROR);
}

//------------------------------------------------------------------------------
//      XLinkFindPCIEDeviceTests
//------------------------------------------------------------------------------

TEST_F(XLinkFindPCIEDeviceTests, CanFindDevice) {
    if (available_devices == 0)
        GTEST_SKIP();

    deviceDesc_t deviceRequirements = getPCIeDeviceRequirements();

    deviceDesc_t deviceDesc = {};

    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, deviceRequirements, &deviceDesc));

    EXPECT_TRUE(strstr(deviceDesc.name, PCIE_NAME_SUBSTR) != nullptr);
    EXPECT_EQ(deviceDesc.protocol, X_LINK_PCIE);
    EXPECT_EQ(deviceDesc.platform, X_LINK_MYRIAD_X);
}

TEST_F(XLinkFindPCIEDeviceTests, CannotFindSameDeviceTwice) {
    if (available_devices == 0)
        GTEST_SKIP();

    deviceDesc_t deviceRequirements = getPCIeDeviceRequirements();

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

TEST_F(XLinkFindPCIEDeviceTests, CanFindDeviceByName) {
    if (available_devices == 0)
        GTEST_SKIP();

    deviceDesc_t deviceRequirements = getPCIeDeviceRequirements();

    deviceDesc_t deviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkFindFirstSuitableDevice(
                                    X_LINK_ANY_STATE, deviceRequirements, &deviceDesc));

    deviceDesc_t deviceRequirementsWithName = getPCIeDeviceRequirements();
    strcpy(deviceRequirementsWithName.name, deviceDesc.name);

    deviceDesc_t deviceDescSearchByName = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkFindFirstSuitableDevice(
            X_LINK_ANY_STATE, deviceRequirementsWithName, &deviceDescSearchByName));

    ASSERT_TRUE(strcmp(deviceDesc.name, deviceDescSearchByName.name) == 0);
}

TEST_F(XLinkFindPCIEDeviceTests, CanResetDevice) {
    deviceDesc_t deviceRequirements = getPCIeDeviceRequirements();
    deviceDesc_t deviceDesc = {};

    ASSERT_EQ(X_LINK_SUCCESS,
        XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, deviceRequirements, &deviceDesc));

    deviceDesc_t tempDeviceDesc = {};
    // Boot device, if it not booted already
    if (X_LINK_DEVICE_NOT_FOUND ==
        XLinkFindFirstSuitableDevice(X_LINK_BOOTED, deviceRequirements, &tempDeviceDesc)) {

        std::string firmwarePath;
        ASSERT_NO_THROW(firmwarePath = getMyriadPCIeFirmware());

        EXPECT_EQ(X_LINK_SUCCESS, XLinkBoot(&deviceDesc, firmwarePath.c_str()));
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    deviceDesc_t deviceDescBooted = {};
    XLinkError_t rc = XLinkFindFirstSuitableDevice(X_LINK_BOOTED, deviceRequirements, &deviceDesc);
    ASSERT_EQ(rc, X_LINK_SUCCESS);  // Booted device found

    int amountOfNotBootedBeforeReset = getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_PCIE);

    //// Close and reset
    XLinkHandler_t *handler = (XLinkHandler_t *) malloc(sizeof(XLinkHandler_t));
    //// Connect to device
    memset(handler, 0, sizeof(XLinkHandler_t));

    handler->protocol = deviceDesc.protocol;
    handler->devicePath = deviceDesc.name;
    EXPECT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));

    std::this_thread::sleep_for(std::chrono::seconds(1));
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    free(handler);
    std::this_thread::sleep_for(std::chrono::seconds(2));

    int amountOfNotBootedAfterReset = getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_PCIE);
    EXPECT_GT(amountOfNotBootedAfterReset, amountOfNotBootedBeforeReset);
}


//#-20037
TEST_F(XLinkFindPCIEDeviceTests, CanFindBootedDevice) {
    if (available_devices == 0)
        GTEST_SKIP();

    deviceDesc_t deviceRequirements = getPCIeDeviceRequirements();

    // Find device
    deviceDesc_t deviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkFindFirstSuitableDevice(
            X_LINK_ANY_STATE, deviceRequirements, &deviceDesc));

    // Boot device
    std::string firmwarePath;
    ASSERT_NO_THROW(firmwarePath = getMyriadPCIeFirmware());

    EXPECT_EQ(X_LINK_SUCCESS, XLinkBoot(&deviceDesc, firmwarePath.c_str()));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Find device
    deviceDesc_t bootedDeviceRequirements = getPCIeDeviceRequirements();
    XLinkDeviceState_t state = X_LINK_BOOTED;

    deviceDesc_t bootedDeviceDescr = {};

    EXPECT_EQ(X_LINK_SUCCESS, XLinkFindFirstSuitableDevice(
            state, bootedDeviceRequirements, &bootedDeviceDescr));

    // Close and reset
    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    // Connect to device
    memset(handler, 0, sizeof(XLinkHandler_t));

    handler->protocol = deviceDesc.protocol;
    handler->devicePath = deviceDesc.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    free(handler);
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

// #-17972
/**
 * This is temporary test.
 * For now it's not clear how to tests multiple device as for now we don't have bench like this
 */
TEST_F(XLinkFindPCIEDeviceTests, OnSecondIndexDeviceWillBeNotFound) {
    if (available_devices == 0)
        GTEST_SKIP();

    // TODO Add check that there is no more then one device
    // TODO Add check that one device is available
    deviceDesc_t deviceRequirements = getPCIeDeviceRequirements();

    const int index = 1;
    // Find device
    deviceDesc_t deviceDesc = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND, findDeviceOnIndex(
            index, X_LINK_ANY_STATE, deviceRequirements, &deviceDesc));
}

/**
 * This is real test for two multi-device case, require two PCIe cards
 * Boot second and expect that first will be unbooted, second booted
 */
TEST_F(XLinkFindPCIEDeviceTests, DISABLED_CanFindFirstDeviceAfterBootSecond) {
    if (available_devices == 0)
        GTEST_SKIP();

    // TODO Add check that there two devices
    deviceDesc_t deviceRequirements = getPCIeDeviceRequirements();

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
    ASSERT_NO_THROW(firmwarePath = getMyriadPCIeFirmware());

    EXPECT_EQ(X_LINK_SUCCESS, XLinkBoot(&secondDeviceDesc, firmwarePath.c_str()));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Check that first still in unbooted state
    deviceDesc_t firstDeviceDescAfter = getPCIeDeviceRequirements();

    EXPECT_EQ(X_LINK_SUCCESS, findDeviceOnIndex(
            0, X_LINK_UNBOOTED, firstDeviceDesc, &firstDeviceDescAfter));

    // Check that second device now in booted state
    deviceDesc_t secondDeviceDescAfter = getPCIeDeviceRequirements();

    EXPECT_EQ(X_LINK_SUCCESS, findDeviceOnIndex(
            0, X_LINK_BOOTED, secondDeviceDesc, &secondDeviceDescAfter));


    // TODO Move it to separate function
    // Close second device
    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    // Connect to device
    memset(handler, 0, sizeof(XLinkHandler_t));

    handler->protocol = secondDeviceDesc.protocol;
    handler->devicePath = secondDeviceDesc.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));
    std::this_thread::sleep_for(std::chrono::seconds(1));

    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    free(handler);
    std::this_thread::sleep_for(std::chrono::seconds(2));
}

//------------------------------------------------------------------------------
//     XLinkResetRemoteUSBTests
//------------------------------------------------------------------------------

TEST_F(XLinkResetRemoteUSBTests, CanResetRemoteUSBDevice) {
    if (getAmountOfDevices(X_LINK_UNBOOTED, X_LINK_USB_VSC) == 0)
        GTEST_SKIP();

    deviceDesc_t unbootedDeviceDescr = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_USB_VSC;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    std::string firmwarePath;

    // Get device name
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &unbootedDeviceDescr));
    ASSERT_NO_THROW(firmwarePath = getMyriadStickFirmwarePath(unbootedDeviceDescr.name));

    // Boot device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkBoot(&unbootedDeviceDescr, firmwarePath.c_str()));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Booted device appear
    deviceDesc_t bootedDeviceDesc = {};
    EXPECT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &bootedDeviceDesc));

    // The device is not in unbooted and booted list at the same time
    deviceDesc_t foundDeviceDesc = {};
    EXPECT_EQ(X_LINK_DEVICE_NOT_FOUND,
              XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, unbootedDeviceDescr, &foundDeviceDesc));

    // Connect to device
    XLinkHandler_t *handler = (XLinkHandler_t *)malloc(sizeof(XLinkHandler_t));
    memset(handler, 0, sizeof(XLinkHandler_t));

    handler->protocol = bootedDeviceDesc.protocol;
    handler->devicePath = bootedDeviceDesc.name;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(handler));

//    #-20854
//    std::this_thread::sleep_for(std::chrono::seconds(2));
    std::this_thread::sleep_for(std::chrono::seconds(1));

    // Reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler->linkId));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    // Make sure that device is really rebooted
    deviceDesc_t deviceDesc = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND,
            XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, bootedDeviceDesc, &deviceDesc));
}
