// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvnc_usb_test_cases.h"

//------------------------------------------------------------------------------
//      Implementation of class MvncOpenUSBDevice
//------------------------------------------------------------------------------
void MvncOpenUSBDevice::SetUp() {
    ncDeviceResetAll();
    MvncTestsCommon::SetUp();

    availableDevices_ = getAmountOfNotBootedDevices(NC_USB);

    deviceDesc_.protocol = NC_USB;
}

//------------------------------------------------------------------------------
//      Implementation of class MvncDevicePlatform
//------------------------------------------------------------------------------
void MvncDevicePlatform::SetUp() {
    MvncOpenUSBDevice::SetUp();

    available_myriadX_ = getAmountOfMyriadXDevices(NC_USB);
    available_myriad2_ = getAmountOfMyriad2Devices(NC_USB);

    devicePlatform_ = GetParam();
    deviceDesc_.platform = devicePlatform_;
}
