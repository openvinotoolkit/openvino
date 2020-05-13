// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvnc_common_test_cases.h"

//------------------------------------------------------------------------------
//      class MvncOpenUSBDevice
//------------------------------------------------------------------------------
class MvncOpenUSBDevice : public MvncTestsCommon {
public:
    ncDeviceHandle_t*   deviceHandle_       = nullptr;
    ncDeviceDescr_t     deviceDesc_         = {};

    ~MvncOpenUSBDevice() override = default;

protected:
    void SetUp() override;
};

//------------------------------------------------------------------------------
//      class MvncCloseUSBDevice
//------------------------------------------------------------------------------
class MvncCloseUSBDevice : public MvncOpenUSBDevice {
};

//------------------------------------------------------------------------------
//      class MvncDevicePlatform
//------------------------------------------------------------------------------
class MvncDevicePlatform : public MvncOpenUSBDevice,
                           public testing::WithParamInterface<ncDevicePlatform_t>{
public:
    long available_myriadX_ = 0;
    long available_myriad2_ = 0;
    ncDevicePlatform_t devicePlatform_;

    ~MvncDevicePlatform() override = default;

protected:
    void SetUp() override;
};
