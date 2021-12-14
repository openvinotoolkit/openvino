// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "XLink_common_cases.hpp"

#include <thread>

//------------------------------------------------------------------------------
//      XLinkNullPtrTests
//------------------------------------------------------------------------------

TEST_F(XLinkNullPtrTests, XLinkInitialize) {
    ASSERT_EQ(XLinkInitialize(nullptr), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkConnect) {
    ASSERT_EQ(XLinkConnect(nullptr), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkOpenAndCloseStream) {
    ASSERT_EQ(XLinkOpenStream(0, nullptr, 0), INVALID_STREAM_ID);
    ASSERT_EQ(XLinkCloseStream(0), X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkFindDevice) {
    ASSERT_EQ(XLinkFindFirstSuitableDevice(X_LINK_ANY_STATE, {}, nullptr), X_LINK_ERROR);
    ASSERT_EQ(XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, {}, nullptr, -1, nullptr),
              X_LINK_ERROR);
}

TEST_F(XLinkNullPtrTests, XLinkWriteData) {
    ASSERT_EQ(XLinkWriteData(0, nullptr, 0), X_LINK_ERROR);
}

//------------------------------------------------------------------------------
//  XLinkBootTests
//------------------------------------------------------------------------------

TEST_P(XLinkBootTests, StressTestBootToOpenAndCloseDevice) {
    if (getCountSpecificDevices(X_LINK_ANY_STATE, _protocol) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t deviceDesc = {};
    deviceDesc_t in_deviceDesc = {};

    in_deviceDesc.protocol = _protocol;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    ASSERT_EQ(X_LINK_SUCCESS,
        XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &deviceDesc));

    std::string firmwarePath;
    ASSERT_NO_THROW(firmwarePath = getMyriadFirmwarePath(deviceDesc));

    for (int i = 0; i < 10; ++i) {
        printf("Boot device. Iteration: %d\n", i);
        ASSERT_EQ(X_LINK_SUCCESS, XLinkBoot(&deviceDesc, firmwarePath.c_str()));
        // FIXME: need to find a way to avoid this sleep
        std::this_thread::sleep_for(kBootTimeoutSec);

        // Find booted
        deviceDesc_t bootedDeviceDesc = {};
        ASSERT_EQ(X_LINK_SUCCESS,
            XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &bootedDeviceDesc));

        XLinkHandler_t handler = {0};

        connectToDevice(bootedDeviceDesc, &handler);
        closeDevice(&handler);
    }
}

//------------------------------------------------------------------------------
//      XLinkConnectUSBTests
//------------------------------------------------------------------------------

TEST_P(XLinkConnectTests, ConnectToDevice) {
    if (getCountSpecificDevices(X_LINK_UNBOOTED, _protocol) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t deviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};

    deviceDesc.protocol = _protocol;
    bootDevice(deviceDesc, bootedDeviceDesc);

    XLinkHandler_t handler = {0};
    handler.protocol = bootedDeviceDesc.protocol;
    handler.devicePath = bootedDeviceDesc.name;

    ASSERT_EQ(X_LINK_SUCCESS, XLinkConnect(&handler));
    std::this_thread::sleep_for(std::chrono::seconds(2));

    closeDevice(&handler);
}

//------------------------------------------------------------------------------
//      XLinkFindFirstSuitableDeviceTests
//------------------------------------------------------------------------------

TEST_P(XLinkFindFirstSuitableDevicePlatformTests, ReturnCorrectAvailableDeviceName) {
    if (getCountSpecificDevices(X_LINK_ANY_STATE, _protocol, _platform) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t deviceDesc = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = _protocol;
    in_deviceDesc.platform = _platform;

    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &deviceDesc));
    ASSERT_TRUE(strlen(deviceDesc.name) > 2);
    ASSERT_EQ(deviceDesc.protocol, in_deviceDesc.protocol);

    if(_platform == X_LINK_ANY_PLATFORM) {
        EXPECT_NE(deviceDesc.platform, X_LINK_ANY_PLATFORM);
    } else {
        EXPECT_EQ(deviceDesc.platform, _platform);
    }

    if(_protocol != X_LINK_USB_VSC) {
        std::string deviceName(deviceDesc.name);
        switch (_platform) {
            case X_LINK_MYRIAD_2: {
                EXPECT_TRUE(deviceName.find(kUSBMyriad2) != std::string::npos);
                break;
            }
            case X_LINK_MYRIAD_X: {
                EXPECT_TRUE(deviceName.find(kUSBMyriadX) != std::string::npos);
                break;
            }
            default:
                break;
        }
    }
}

TEST_P(XLinkFindFirstSuitableDeviceTests, CanFindDeviceByName) {
    if (getCountSpecificDevices(X_LINK_ANY_STATE, _protocol) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t deviceDesc = {};
    deviceDesc_t in_deviceDesc = {};

    in_deviceDesc.protocol = _protocol;
    ASSERT_EQ(X_LINK_SUCCESS, XLinkFindFirstSuitableDevice(
        X_LINK_ANY_STATE, in_deviceDesc, &deviceDesc));

    deviceDesc_t deviceRequirementsWithName = {};
    deviceRequirementsWithName.protocol = _protocol;
    strcpy(deviceRequirementsWithName.name, deviceDesc.name);

    deviceDesc_t deviceDescSearchByName = {};
    ASSERT_EQ(X_LINK_SUCCESS, XLinkFindFirstSuitableDevice(
        X_LINK_ANY_STATE, deviceRequirementsWithName, &deviceDescSearchByName));

    ASSERT_TRUE(strcmp(deviceDesc.name, deviceDescSearchByName.name) == 0);
}

/**
 * This is temporary test.
 * For now it's not clear how to tests multiple device as for now we don't have bench like this
 */
TEST_P(XLinkFindFirstSuitableDeviceTests, OnSecondIndexDeviceWillBeNotFound) {
    auto availableDevices = getCountSpecificDevices(X_LINK_ANY_STATE, _protocol);
    if (availableDevices != 1) {
        GTEST_SKIP();
    }

    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = _protocol;

    const int index = 1;
    // Find device
    deviceDesc_t deviceDesc = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND, findDeviceOnIndex(
        index, X_LINK_ANY_STATE, in_deviceDesc, &deviceDesc));
}

TEST_P(XLinkFindFirstSuitableDeviceTests, ReturnCorrectBootedDeviceName) {
    if (getCountSpecificDevices(X_LINK_ANY_STATE, _protocol) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t bootedDevice = {};
    deviceDesc_t deviceDesc = {};
    deviceDesc.protocol = _protocol;

    bootDevice(deviceDesc, bootedDevice);

    deviceDesc_t foundDeviceDesc = {};
    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = _protocol;
    in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

    EXPECT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_BOOTED, in_deviceDesc, &foundDeviceDesc));

    EXPECT_TRUE(strcmp(bootedDevice.name, foundDeviceDesc.name) == 0);
    EXPECT_EQ(foundDeviceDesc.protocol, _protocol);

    if(_protocol == X_LINK_USB_VSC) {
        std::string foundDeviceName(foundDeviceDesc.name);
        EXPECT_TRUE(foundDeviceName.find(kUSBMyriad2) == std::string::npos);
        EXPECT_TRUE(foundDeviceName.find(kUSBMyriadX) == std::string::npos);
    }

    connectAndCloseDevice(bootedDevice);
}

//------------------------------------------------------------------------------
//      XLinkFindAllSuitableDevicesTests
//------------------------------------------------------------------------------

TEST_F(XLinkFindAllSuitableDevicesTests, CanFindMoreThenTwoDeviceAnyState_USB_PCIE) {
    if (getCountSpecificDevices(X_LINK_UNBOOTED) < 2) {
        GTEST_SKIP();
    }

    deviceDesc_t in_deviceDesc = {};
    deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {{}};

    in_deviceDesc.protocol = X_LINK_ANY_PROTOCOL;
    unsigned int numOfFoundDevices = 0;
    ASSERT_EQ(X_LINK_SUCCESS,
        XLinkFindAllSuitableDevices(
            X_LINK_ANY_STATE, in_deviceDesc, deviceDescArray,
            XLINK_MAX_DEVICES, &numOfFoundDevices));

    ASSERT_EQ(numOfFoundDevices, getCountSpecificDevices(X_LINK_UNBOOTED));
    ASSERT_EQ(numOfFoundDevices,
              getCountSpecificDevices(X_LINK_UNBOOTED, X_LINK_ANY_PROTOCOL, X_LINK_MYRIAD_2) +
              getCountSpecificDevices(X_LINK_UNBOOTED, X_LINK_ANY_PROTOCOL, X_LINK_MYRIAD_X));
}

TEST_F(XLinkFindAllSuitableDevicesTests, CanFindTwoDeviceDifferentState_USB_PCIE) {
    if (getCountSpecificDevices(X_LINK_UNBOOTED) < 2) {
        GTEST_SKIP();
    }

    deviceDesc_t in_deviceDesc = {};
    in_deviceDesc.protocol = X_LINK_ANY_PROTOCOL;

    // Find & boot one device
    deviceDesc_t firstDeviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &firstDeviceDesc));
    bootDevice(firstDeviceDesc, bootedDeviceDesc);

    deviceDesc_t deviceDescArray[XLINK_MAX_DEVICES] = {{}};
    unsigned int numOfFoundDevices = 0;
    ASSERT_EQ(X_LINK_SUCCESS,
              XLinkFindAllSuitableDevices(
                  X_LINK_ANY_STATE, in_deviceDesc, deviceDescArray,
                  XLINK_MAX_DEVICES, &numOfFoundDevices));

    bool foundBootedDevice = false;
    for (int i = 0; i < numOfFoundDevices; ++i) {
        if (deviceDescArray[i].platform == X_LINK_ANY_PLATFORM)
            foundBootedDevice = true;
    }

    EXPECT_GE(numOfFoundDevices, 2);
    EXPECT_TRUE(foundBootedDevice);
    EXPECT_EQ(numOfFoundDevices, getCountSpecificDevices(X_LINK_UNBOOTED) +
        getCountSpecificDevices(X_LINK_BOOTED));

    connectAndCloseDevice(bootedDeviceDesc);
}

//------------------------------------------------------------------------------
//     XLinkResetRemoteTests
//------------------------------------------------------------------------------

TEST_P(XLinkResetRemoteTests, CanResetRemoteDevice) {
    if (getCountSpecificDevices(X_LINK_UNBOOTED, _protocol) == 0) {
        GTEST_SKIP();
    }

    XLinkHandler_t handler = {0};
    deviceDesc_t deviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};

    deviceDesc.protocol = _protocol;
    bootDevice(deviceDesc, bootedDeviceDesc);
    connectToDevice(bootedDeviceDesc, &handler);

    // Reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetRemote(handler.linkId));
    std::this_thread::sleep_for(kResetTimeoutSec);

    // Make sure that device is really rebooted
    deviceDesc_t foundDeviceDesc = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND,
              XLinkFindFirstSuitableDevice(X_LINK_BOOTED, deviceDesc, &foundDeviceDesc));
}

//------------------------------------------------------------------------------
//      XLinkResetAllTests
//------------------------------------------------------------------------------
TEST_P(XLinkResetAllTests, DISABLED_ResetBootedDevice) {
    if (getCountSpecificDevices(X_LINK_UNBOOTED, _protocol) == 0) {
        GTEST_SKIP();
    }

    deviceDesc_t deviceDesc = {};
    deviceDesc_t bootedDeviceDesc = {};

    deviceDesc.protocol = _protocol;
    bootDevice(deviceDesc, bootedDeviceDesc);

    // Try to reset device
    ASSERT_EQ(X_LINK_SUCCESS, XLinkResetAll());
    std::this_thread::sleep_for(kResetTimeoutSec);

    deviceDesc.protocol = X_LINK_ANY_PROTOCOL;
    deviceDesc_t afterResetBootedDescr = {};
    ASSERT_EQ(X_LINK_DEVICE_NOT_FOUND,
              XLinkFindFirstSuitableDevice(X_LINK_BOOTED, deviceDesc, &afterResetBootedDescr));
}

//------------------------------------------------------------------------------
//      XLinkOpenStreamTests
//------------------------------------------------------------------------------

TEST_P(XLinkOpenStreamTests, CanOpenAndCloseStream) {
    streamId_t stream = XLinkOpenStream(_handlerPtr.get()->linkId, "mySuperStream", 1024);
    ASSERT_NE(INVALID_STREAM_ID, stream);
    ASSERT_NE(INVALID_STREAM_ID_OUT_OF_MEMORY, stream);
    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream));
}

// CannotOpenStreamMoreThanMemoryOnDevice
TEST_P(XLinkOpenStreamTests, CannotOpenStreamMoreThanMemoryOnDevice) {
    const int _512MB = 512 * 1024 * 1024;
    streamId_t stream = XLinkOpenStream(_handlerPtr.get()->linkId, "mySuperStream", _512MB);
    ASSERT_EQ(INVALID_STREAM_ID_OUT_OF_MEMORY, stream);
}

// FIXME: the test doesn't work
// TODO: is it correct behavior, should we accept the same names
TEST_P(XLinkOpenStreamTests, DISABLED_CannotOpenTwoStreamsWithTheSameName) {
    const int _1KB = 1 * 1024;
    const char streamName[] = "mySuperStream";
    streamId_t stream0 = XLinkOpenStream(_handlerPtr.get()->linkId, streamName, _1KB);
    ASSERT_NE(INVALID_STREAM_ID, stream0);

    streamId_t stream1 = XLinkOpenStream(_handlerPtr.get()->linkId, streamName, _1KB);
    ASSERT_EQ(INVALID_STREAM_ID, stream1);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream0));
}

// FIXME: XLinkOpenStream doesn't allocate any memory on device
TEST_P(XLinkOpenStreamTests, DISABLED_CannotOpenStreamsMoreThanMemoryOnDevice) {
    const int _256MB = 256 * 1024 * 1024;
    streamId_t stream0 = XLinkOpenStream(_handlerPtr.get()->linkId, "mySuperStream0", _256MB);
    ASSERT_NE(INVALID_STREAM_ID, stream0);

    streamId_t stream1 = XLinkOpenStream(_handlerPtr.get()->linkId, "mySuperStream1", _256MB);
    ASSERT_EQ(INVALID_STREAM_ID, stream1);

    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream0));
    ASSERT_EQ(X_LINK_SUCCESS, XLinkCloseStream(stream1));
}

//------------------------------------------------------------------------------
// Initialization of XLinkCommonTests
//------------------------------------------------------------------------------

INSTANTIATE_TEST_SUITE_P(
    XLinkCommon,
    XLinkBootTests,
    Combine(Values(X_LINK_USB_VSC, X_LINK_PCIE),
            Values(X_LINK_ANY_PLATFORM)),
    XLinkBootTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    XLinkCommon,
    XLinkConnectTests,
    Combine(Values(X_LINK_USB_VSC, X_LINK_PCIE),
            Values(X_LINK_ANY_PLATFORM)),
    XLinkBootTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    XLinkCommon,
    XLinkFindFirstSuitableDevicePlatformTests,
    Combine(Values(X_LINK_USB_VSC),
            Values(X_LINK_MYRIAD_2, X_LINK_MYRIAD_X, X_LINK_ANY_PLATFORM)),
    XLinkBootTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    XLinkCommonPCIE,
    XLinkFindFirstSuitableDevicePlatformTests,
    Combine(Values(X_LINK_PCIE),
            Values(X_LINK_ANY_PLATFORM)),
    XLinkBootTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    XLinkCommon,
    XLinkFindFirstSuitableDeviceTests,
    Combine(Values(X_LINK_USB_VSC, X_LINK_PCIE),
            Values(X_LINK_ANY_PLATFORM)),
    XLinkBootTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    XLinkCommon,
    XLinkResetAllTests,
    Combine(Values(X_LINK_USB_VSC, X_LINK_PCIE),
            Values(X_LINK_ANY_PLATFORM)),
    XLinkBootTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    XLinkCommon,
    XLinkResetRemoteTests,
    Combine(Values(X_LINK_USB_VSC, X_LINK_PCIE),
            Values(X_LINK_ANY_PLATFORM)),
    XLinkBootTests::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(
    XLinkCommon,
    XLinkOpenStreamTests,
    Combine(Values(X_LINK_USB_VSC, X_LINK_PCIE),
            Values(X_LINK_ANY_PLATFORM)),
    XLinkOpenStreamTests::getTestCaseName);

