// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvnc_no_boot_test_cases.h"

//------------------------------------------------------------------------------
//      Implementation of class MvncNoBootTests
//------------------------------------------------------------------------------
void MvncNoBootTests::bootOneDevice() {
    // In case already booted device exist, do nothing
    if (getAmountOfBootedDevices() == 0) {
        MvncTestsCommon::bootOneDevice(NC_USB);
    }
}

//------------------------------------------------------------------------------
//      Implementation of class MvncNoBootOpenDevice
//------------------------------------------------------------------------------
void MvncNoBootOpenDevice::SetUp() {
    MvncNoBootTests::SetUp();
    available_devices = getAmountOfDevices(NC_USB);
    ASSERT_TRUE(available_devices > 0);

    // With NO_BOOT option we should boot device with firmware before trying to open
    bootOneDevice();
}
