// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvnc.h"
#include "mvnc_tests_common.hpp"

class MvncNoBootTests: public MvncTestsCommon {
public:
    void bootOneDevice() {
        // In case already booted device exist, do nothing
        if (getAmountOfBootedDevices() == 0) {
            MvncTestsCommon::bootOneDevice(NC_USB);
        }
    }
protected:
    ~MvncNoBootTests() override = default;
};

//  ***********************************************  //
//              Open Device TESTS                    //
class MvncOpenDevice : public MvncNoBootTests {
public:
    int available_devices = 0;
protected:
    ~MvncOpenDevice() override = default;
    void SetUp() override {
        MvncNoBootTests::SetUp();
        available_devices = getAmountOfDevices(NC_USB);
        ASSERT_TRUE(available_devices > 0);

        // With NO_BOOT option we should boot device with firmware before trying to open
        bootOneDevice();
    }
};

/**
* @brief Open any device and close it
*/
TEST_F(MvncOpenDevice, OpenAndClose) {
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
TEST_F(MvncOpenDevice, OpenTwiceSameHandler) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ncDeviceDescr_t deviceDesc = {};
    deviceDesc.protocol = NC_USB;
    deviceDesc.platform = NC_ANY_PLATFORM;

    char dev_addr_first_open[MAX_DEV_NAME];
    unsigned int data_lenght_first = MAX_DEV_NAME;

    char dev_addr_second_open[MAX_DEV_NAME];
    unsigned int data_lenght_second = MAX_DEV_NAME;

    // First open, get device name
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


/**
 * @brief Open device twice one run after another. It should check, that link to device closed correctly
 * @note Mostly this test important for PCIE and connect to booted option, as in that cases XLinkReset have another behavior
 */
TEST_F(MvncOpenDevice, OpenDeviceWithOneXLinkInitializion) {
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


//  ***********************************************  //
//               Close device tests                  //

class MvncCloseDevice : public MvncNoBootTests {
protected:
    ~MvncCloseDevice() override = default;
};

/**
* @brief Correct closing if handle is empty
*/
TEST_F(MvncCloseDevice, EmptyDeviceHandler) {
    ncDeviceHandle_t *deviceHandle = nullptr;
    ASSERT_NO_ERROR(ncDeviceClose(&deviceHandle));
}

/**
* @brief Device, which was booted before open, shouldn't reboot after ncDeviceClose call
*/
TEST_F(MvncCloseDevice, AlreadyBootedDeviceWillNotReboot) {
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
