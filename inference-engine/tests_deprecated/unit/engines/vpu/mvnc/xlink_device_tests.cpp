// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <gmock/gmock.h>
#include <tests_common.hpp>
#include <watchdog/xlink_device.h>
#include <mvnc/include/ncPrivateTypes.h>

using namespace ::testing;
using namespace InferenceEngine;

class XLinkDeviceTests: public TestsCommon {};
class XLinkDeviceTestsWithParam: public TestsCommon, public testing::WithParamInterface<int> {};

TEST_F(XLinkDeviceTests, shouldCreateXlinkDevice) {
    devicePrivate_t devicePrivate = {0};
    devicePrivate.wd_interval = 1;

    WdDeviceHndl_t* deviceHndl = nullptr;
    ASSERT_EQ(WD_ERRNO, xlink_device_create(&deviceHndl, &devicePrivate));

    xlink_device_destroy(deviceHndl);
}

TEST_P(XLinkDeviceTestsWithParam, shouldNotCreateXlinkDeviceWithInvalidInterval) {
    devicePrivate_t devicePrivate = {0};
    devicePrivate.wd_interval = GetParam();

    WdDeviceHndl_t* deviceHndl = nullptr;
    ASSERT_NE(WD_ERRNO, xlink_device_create(&deviceHndl, &devicePrivate));

    xlink_device_destroy(deviceHndl);
}

INSTANTIATE_TEST_SUITE_P(WatchdogDevice,
    XLinkDeviceTestsWithParam,
    testing::Values(0, -1, -WATCHDOG_MAX_PING_INTERVAL_MS));
