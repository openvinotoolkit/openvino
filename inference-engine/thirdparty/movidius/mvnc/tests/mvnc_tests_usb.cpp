// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvnc.h"
#include "mvnc_tests_common.hpp"

//  ***********************************************  //
//              Open Device TESTS                    //

class MvncOpenUSBDevice : public MvncTestsCommon {
public:
    int available_devices = 0;
protected:
    ~MvncOpenUSBDevice() override = default;
    void SetUp() override {
        ncDeviceResetAll();
        MvncTestsCommon::SetUp();
        available_devices = getAmountOfNotBootedDevices(NC_USB);
        ASSERT_TRUE(available_devices > 0);
    }
};

/**
* @brief Open any device with custom firmware path as ncDeviceOpen argument
*/
TEST_F(MvncOpenUSBDevice, WithCustomFirmware) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    // Use custom firmware dir path as parameter for ncDeviceOpen
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
* @brief Open all available devices and close them
*/
TEST_F(MvncOpenUSBDevice, AllAvailableDevices) {
    ncDeviceHandle_t * deviceHandle[MAX_DEVICES] = {nullptr};
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    for (int index = 0; index < available_devices; ++index) {
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle[index], deviceDesc, watchdogInterval, firmwarePath));
    }
    for (int index = 0; index < available_devices; ++index) {
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle[index]));
    }
}

/**
* @brief Open all available devices in parallel threads and close them
*/
TEST_F(MvncOpenUSBDevice, AllAvailableMultiThreads) {
    std::thread requests[MAX_DEVICES];
    ncDeviceHandle_t * deviceHandle[MAX_DEVICES] = {nullptr};
    ncStatus_t rc[MAX_DEVICES];
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    for (int i = 0; i < available_devices; ++i) {
        requests[i] = std::thread([i, &rc, &deviceHandle, deviceDesc, this]() {
            rc[i] = ncDeviceOpen(&deviceHandle[i], deviceDesc, watchdogInterval, firmwarePath);
        });
    }

    for (int i = 0; i < available_devices; ++i) {
        requests[i].join();
        ASSERT_NO_ERROR(rc[i]);
    }

    for (int i = 0; i < available_devices; ++i) {
        ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle[i]));
    }
}

/**
* @brief Open any device with invalid firmware path
*/
TEST_F(MvncOpenUSBDevice, WithInvalidFirmwarePath) {
    const char invalidPath[MAX_PATH] = "./InvalidPath/";
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    // Use custom firmware dir path as parameter for ncDeviceOpen
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, invalidPath));

    ASSERT_EQ(deviceHandle, nullptr);
}

TEST_F(MvncOpenUSBDevice, OpenAvailableDeviceByName) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    char dev_addr_open[NC_MAX_NAME_SIZE];
    unsigned int data_lenght = NC_MAX_NAME_SIZE;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    auto availableDevices = getDevicesList();

    ASSERT_TRUE(availableDevices.size());
    strncpy(deviceDesc.name, availableDevices[0].c_str(), NC_MAX_NAME_SIZE);

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                                      dev_addr_open, &data_lenght));

    ASSERT_TRUE(strncmp(dev_addr_open, deviceDesc.name, NC_MAX_NAME_SIZE) == 0);
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

TEST_F(MvncOpenUSBDevice, ErrorWhenWrongDeviceName) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    char badName[] = "BadName";

    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;
    strncpy(deviceDesc.name, badName, NC_MAX_NAME_SIZE);

    auto availableDevices = getDevicesList();
    ASSERT_TRUE(availableDevices.size());

    ASSERT_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
}

TEST_F(MvncOpenUSBDevice, OpenTwiceSameHandlerByName) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    char dev_addr_first_open[MAX_DEV_NAME];
    unsigned int data_lenght_first = MAX_DEV_NAME;

    char dev_addr_second_open[MAX_DEV_NAME];
    unsigned int data_lenght_second = MAX_DEV_NAME;

    auto availableDevices = getDevicesList();

    ASSERT_TRUE(availableDevices.size());
    strncpy(deviceDesc.name, availableDevices[0].c_str(), NC_MAX_NAME_SIZE);

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                                      dev_addr_first_open, &data_lenght_first));

    // Second open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                                      dev_addr_second_open, &data_lenght_second));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
    // Should be the same device
    ASSERT_STREQ(dev_addr_first_open, dev_addr_second_open);
}

TEST_F(MvncOpenUSBDevice, CheckErrorWhenPlatformConflictWithName) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDevicePlatform_t wrongPlatform = NC_ANY_PLATFORM;
    auto availableDevices = getDevicesList();
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;

    ASSERT_TRUE(availableDevices.size());

    if(isMyriadXUSBDevice(availableDevices[0])) {
        wrongPlatform = NC_MYRIAD_2;
    } else {
        wrongPlatform = NC_MYRIAD_X;
    }

    strncpy(deviceDesc.name, availableDevices[0].c_str(), NC_MAX_NAME_SIZE);
    deviceDesc.platform = wrongPlatform;

    ASSERT_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
}

//  ***********************************************  //
//             Specific device TESTS                //

class MvncDevicePlatform : public MvncOpenUSBDevice,
                           public testing::WithParamInterface<ncDevicePlatform_t>{
public:
    long available_myriadX = 0;
    long available_myriad2 = 0;
    ncDevicePlatform_t devicePlatform;

    ~MvncDevicePlatform() override = default;

protected:
    void SetUp() override {
        MvncOpenUSBDevice::SetUp();

        available_myriadX = getAmountOfMyriadXDevices();
        available_myriad2 = getAmountOfMyriad2Devices();

        devicePlatform = GetParam();
    }
};

/**
* @brief Open specified device and close it
*/
TEST_P(MvncDevicePlatform, OpenAndClose) {
    if (available_myriad2 == 0 || available_myriadX == 0)
        GTEST_SKIP();

    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = devicePlatform;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));

    char deviceName[MAX_DEV_NAME];
    unsigned int size = MAX_DEV_NAME;
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME, deviceName, &size));

    EXPECT_TRUE(isSamePlatformUSBDevice(deviceName, devicePlatform));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));

}

INSTANTIATE_TEST_CASE_P(MvncTestsPlatform,
                        MvncDevicePlatform,
                        ::testing::ValuesIn(myriadPlatforms),
                        PrintToStringParamName());
