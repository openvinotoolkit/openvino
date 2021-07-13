// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <thread>

#include "mvnc.h"
#include "mvnc_test_helper.h"
#include "mvnc_usb_test_cases.h"
#include "ncPrivateTypes.h"

//------------------------------------------------------------------------------
//      MvncOpenUSBDevice Tests
//------------------------------------------------------------------------------
/**
* @brief Open any device with custom firmware path as ncDeviceOpen argument
*/

TEST_F(MvncOpenUSBDevice, ShouldOpenDeviceAfterChangeConnectTimeoutFromZero) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    ncDeviceHandle_t *deviceHandle = nullptr;
    std::string actDeviceName;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_ANY_PROTOCOL;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncSetDeviceConnectTimeout(0));
    ASSERT_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));
    std::this_thread::sleep_for(5_sec);
    ASSERT_NO_ERROR(ncDeviceResetAll());

    ASSERT_NO_ERROR(ncSetDeviceConnectTimeout(30));
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, m_ncDeviceOpenParams));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle, m_watchdogHndl));

    ASSERT_NO_ERROR(ncDeviceResetAll());
}


TEST_F(MvncOpenUSBDevice, WithCustomFirmware) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    // Use custom firmware dir path as parameter for ncDeviceOpen
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_, m_watchdogHndl));
}

/**
* @brief Open all available devices and close them
*/
TEST_F(MvncOpenUSBDevice, AllAvailableDevices) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    ncDeviceHandle_t * deviceHandles[MAX_DEVICES] = {nullptr};

    for (int index = 0; index < availableDevices_; ++index) {
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandles[index], deviceDesc_, m_ncDeviceOpenParams));
    }
    for (int index = 0; index < availableDevices_; ++index) {
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandles[index], m_watchdogHndl));
    }
}

/**
* @brief Open all available devices in parallel threads and close them
*/
TEST_F(MvncOpenUSBDevice, AllAvailableMultiThreads) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    std::thread requests[MAX_DEVICES];
    ncDeviceHandle_t * deviceHandle[MAX_DEVICES] = {nullptr};
    ncStatus_t rc[MAX_DEVICES];

    for (int i = 0; i < availableDevices_; ++i) {
        requests[i] = std::thread([i, &rc, &deviceHandle, this]() {
            rc[i] = ncDeviceOpen(&deviceHandle[i], deviceDesc_, m_ncDeviceOpenParams);
        });
    }

    for (int i = 0; i < availableDevices_; ++i) {
        requests[i].join();
        ASSERT_NO_ERROR(rc[i]);
    }

    for (int i = 0; i < availableDevices_; ++i) {
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle[i], m_watchdogHndl));
    }
}

/**
* @brief Open any device with invalid firmware path
*/
TEST_F(MvncOpenUSBDevice, WithInvalidFirmwarePath) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    const char invalidPath[MAX_PATH] = "./InvalidPath/";

    // Use custom firmware dir path as parameter for ncDeviceOpen
    m_ncDeviceOpenParams.customFirmwareDirectory = invalidPath;
    ASSERT_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));

    ASSERT_EQ(deviceHandle_, nullptr);
}

TEST_F(MvncOpenUSBDevice, OpenAvailableDeviceByName) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    char dev_addr_open[NC_MAX_NAME_SIZE];
    unsigned int data_length = NC_MAX_NAME_SIZE;

    auto availableDevices = getDevicesList();

    ASSERT_TRUE(availableDevices.size());
    strncpy(deviceDesc_.name, availableDevices[0].c_str(), NC_MAX_NAME_SIZE);

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle_, NC_RO_DEVICE_NAME,
                                      dev_addr_open, &data_length));

    ASSERT_TRUE(strncmp(dev_addr_open, deviceDesc_.name, NC_MAX_NAME_SIZE) == 0);
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_, m_watchdogHndl));
}

TEST_F(MvncOpenUSBDevice, ErrorWhenWrongDeviceName) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    char badName[] = "BadName";

    strncpy(deviceDesc_.name, badName, NC_MAX_NAME_SIZE);

    auto availableDevices = getDevicesList();
    ASSERT_TRUE(availableDevices.size());

    ASSERT_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));
}

TEST_F(MvncOpenUSBDevice, OpenTwiceSameHandlerByName) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    char dev_addr_first_open[MAX_DEV_NAME];
    unsigned int data_length_first = MAX_DEV_NAME;

    char dev_addr_second_open[MAX_DEV_NAME];
    unsigned int data_length_second = MAX_DEV_NAME;

    auto availableDevices = getDevicesList();

    ASSERT_TRUE(availableDevices.size());
    strncpy(deviceDesc_.name, availableDevices[0].c_str(), NC_MAX_NAME_SIZE);

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle_, NC_RO_DEVICE_NAME,
                                      dev_addr_first_open, &data_length_first));

    // Second open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle_, NC_RO_DEVICE_NAME,
                                      dev_addr_second_open, &data_length_second));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_, m_watchdogHndl));
    // Should be the same device
    ASSERT_STREQ(dev_addr_first_open, dev_addr_second_open);
}

TEST_F(MvncOpenUSBDevice, CheckErrorWhenPlatformConflictWithName) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    ncDevicePlatform_t wrongPlatform = NC_ANY_PLATFORM;
    auto availableDevices = getDevicesList();

    ASSERT_TRUE(availableDevices.size());

    if(isMyriadXUSBDevice(availableDevices[0])) {
        wrongPlatform = NC_MYRIAD_2;
    } else {
        wrongPlatform = NC_MYRIAD_X;
    }

    strncpy(deviceDesc_.name, availableDevices[0].c_str(), NC_MAX_NAME_SIZE);
    deviceDesc_.platform = wrongPlatform;

    ASSERT_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));
}

//------------------------------------------------------------------------------
//      MvncCloseUSBDevice Tests
//------------------------------------------------------------------------------
#if (!(defined(_WIN32) || defined(_WIN64)))
TEST_F(MvncCloseUSBDevice, USBDeviceWillBeAvailableRightAfterClosing) {
    if (availableDevices_ == 0)
        GTEST_SKIP();

    ASSERT_NO_ERROR(ncDeviceOpen(
            &deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));

    ASSERT_TRUE(deviceHandle_);

    deviceDesc_t toFindDeviceDescr = {
            .protocol = X_LINK_USB_VSC,
            .platform = X_LINK_ANY_PLATFORM
    };
    strcpy(deviceDesc_.name, deviceHandle_->private_data->dev_addr);

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_, m_watchdogHndl));

    deviceDesc_t foundDevice = {};
    XLinkError_t rc = XLinkFindFirstSuitableDevice(
            X_LINK_UNBOOTED, toFindDeviceDescr, &foundDevice);
    ASSERT_EQ(X_LINK_SUCCESS, rc);
}
#endif

//------------------------------------------------------------------------------
//      MvncDevicePlatform Tests
//------------------------------------------------------------------------------
/**
* @brief Open specified device and close it
*/
TEST_P(MvncDevicePlatform, OpenAndClose) {
    if (available_myriad2_ == 0 || available_myriadX_ == 0)
        GTEST_SKIP();

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle_, deviceDesc_, m_ncDeviceOpenParams));

    char deviceName[MAX_DEV_NAME];
    unsigned int size = MAX_DEV_NAME;
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle_, NC_RO_DEVICE_NAME, deviceName, &size));

    EXPECT_TRUE(isSamePlatformUSBDevice(deviceName, devicePlatform_));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle_, m_watchdogHndl));

}

INSTANTIATE_TEST_SUITE_P(MvncTestsPlatform,
                        MvncDevicePlatform,
                        ::testing::ValuesIn(myriadPlatforms),
                        PrintToStringParamName());
