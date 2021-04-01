// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvnc_stress_test_cases.h"

//------------------------------------------------------------------------------
//      Implementation of class MvncStressTests
//------------------------------------------------------------------------------
void MvncStressTests::SetUp() {
    MvncTestsCommon::SetUp();

    _deviceProtocol = GetParam();
    available_devices = getAmountOfDevices(_deviceProtocol);
    ASSERT_TRUE(available_devices > 0) << ncProtocolToStr(_deviceProtocol)
                                       << " devices not found";
    ASSERT_NO_ERROR(setLogLevel(MVLOG_WARN));

#ifdef NO_BOOT
    // In case already booted device exist, do nothing
        if (getAmountOfBootedDevices() == 0) {
            MvncTestsCommon::bootOneDevice(NC_USB);
        }
#endif
}
