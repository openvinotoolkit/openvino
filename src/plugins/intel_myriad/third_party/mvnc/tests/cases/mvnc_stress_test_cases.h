// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "mvnc_common_test_cases.h"

//------------------------------------------------------------------------------
//      class MvncStressTests
//------------------------------------------------------------------------------
class MvncStressTests : public MvncTestsCommon,
                        public testing::WithParamInterface<ncDeviceProtocol_t>{
public:
    int available_devices = 0;

protected:
    ~MvncStressTests() override = default;
    void SetUp() override;

    ncDeviceProtocol_t _deviceProtocol = NC_ANY_PROTOCOL;
};
