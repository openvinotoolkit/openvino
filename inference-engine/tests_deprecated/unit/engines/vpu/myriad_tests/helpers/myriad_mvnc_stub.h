// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include "gmock/gmock.h"
#include "myriad_mvnc_wrapper.h"

using namespace vpu::MyriadPlugin;

//------------------------------------------------------------------------------
// class MvncStub
//------------------------------------------------------------------------------

class MvncStub : public IMvnc {
public:
    //Operations
    MOCK_CONST_METHOD0(AvailableDevicesNames, std::vector<std::string>());
    MOCK_CONST_METHOD0(AvailableDevicesDesc, std::vector<ncDeviceDescr_t>());

    MOCK_METHOD0(watchdogHndl, WatchdogHndl_t*());

    ~MvncStub() = default;
};
