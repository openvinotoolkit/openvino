// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include "myriad_mvnc_wrapper.h"

using namespace vpu::MyriadPlugin;

//------------------------------------------------------------------------------
// class MvncStub
//------------------------------------------------------------------------------

class MvncStub : public IMvnc {
public:
    //Operations
    MOCK_METHOD(std::vector<std::string>, AvailableDevicesNames, (), (const));
    MOCK_METHOD(std::vector<ncDeviceDescr_t>, AvailableDevicesDesc, (), (const));

    MOCK_METHOD(WatchdogHndl_t*, watchdogHndl, ());

    ~MvncStub() = default;
};
