// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvnc.h"
#include "mvnc_no_boot_test_cases.h"

//------------------------------------------------------------------------------
//      MvncNoBootOpenDevice Tests
//------------------------------------------------------------------------------
/**
* @brief Open any device and close it
*/
TEST_F(MvncNoBootOpenDevice, OpenAndClose) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
* @brief Try to open device twice. DeviceHandle shouldn't be overwritten
*/
TEST_F(MvncNoBootOpenDevice, OpenTwiceSameHandler) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    char dev_addr_first_open[MAX_DEV_NAME];
    unsigned int data_length_first = MAX_DEV_NAME;

    char dev_addr_second_open[MAX_DEV_NAME];
    unsigned int data_length_second = MAX_DEV_NAME;

    // First open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                        dev_addr_first_open, &data_length_first));

    // Second open, get device name
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceGetOption(deviceHandle, NC_RO_DEVICE_NAME,
                        dev_addr_second_open, &data_length_second));

    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
    // Should be the same device
    ASSERT_STREQ(dev_addr_first_open, dev_addr_second_open);
}


/**
 * @brief Open device twice one run after another. It should check, that link to device closed correctly
 * @note Mostly this test important for PCIE and connect to booted option, as in that cases XLinkReset have another behavior
 */
TEST_F(MvncNoBootOpenDevice, OpenDeviceWithOneXLinkInitializion) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));

    // Second open
    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));

}

//------------------------------------------------------------------------------
//      MvncNoBootCloseDevice Tests
//------------------------------------------------------------------------------
/**
* @brief Correct closing if handle is empty
*/
TEST_F(MvncNoBootCloseDevice, EmptyDeviceHandler) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
* @brief Device, which was booted before open, shouldn't reboot after ncDeviceClose call
*/
TEST_F(MvncNoBootCloseDevice, AlreadyBootedDeviceWillNotReboot) {
    bootOneDevice();

    ASSERT_EQ(getAmountOfBootedDevices(), 1);

    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandle, deviceDesc, watchdogInterval, firmwarePath));
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));

    ASSERT_EQ(getAmountOfBootedDevices(), 1);
}
