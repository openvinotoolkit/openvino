// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvnc_common_test_cases.h"

//------------------------------------------------------------------------------
//      class MvncNoBootTests
//------------------------------------------------------------------------------
class MvncNoBootTests: public MvncTestsCommon {
public:
    void bootOneDevice();
protected:
    ~MvncNoBootTests() override = default;
};

//------------------------------------------------------------------------------
//      class MvncNoBootOpenDevice
//------------------------------------------------------------------------------
class MvncNoBootOpenDevice : public MvncNoBootTests {
public:
    int available_devices = 0;
protected:
    ~MvncNoBootOpenDevice() override = default;
    void SetUp() override;
};

//------------------------------------------------------------------------------
//      class MvncNoBootCloseDevice
//------------------------------------------------------------------------------
class MvncNoBootCloseDevice : public MvncNoBootTests {
protected:
    ~MvncNoBootCloseDevice() override = default;
};
