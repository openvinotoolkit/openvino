// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "ie_device.hpp"
#include "details/ie_exception.hpp"

using namespace InferenceEngine;

class DeviceTests : public ::testing::Test {
protected:
    virtual void TearDown() {
    }

    virtual void SetUp() {
    }

public:

};

TEST_F(DeviceTests, internalFindThrowsOnBadDevice) {
    FindPluginRequest request = { TargetDevice::eBalanced };
    ASSERT_THROW(findPlugin(request), InferenceEngine::details::InferenceEngineException);
}

TEST_F(DeviceTests, externalFindReturnsErrorStatus) {
    FindPluginRequest request = { TargetDevice::eBalanced };
    FindPluginResponse result;
    ResponseDesc desc;
    StatusCode status = findPlugin(request, result, &desc);
    ASSERT_EQ(status, GENERAL_ERROR);
}

#if defined(ENABLE_MKL_DNN)
TEST_F(DeviceTests, externalFindPopulatesResult) {
    FindPluginRequest request = { TargetDevice::eCPU };
    FindPluginResponse result;
    ResponseDesc desc;
    StatusCode status = findPlugin(request, result, &desc);
    ASSERT_EQ(status, OK);
    ASSERT_NE(result.names.size(), 0);
}
#endif

TEST_F(DeviceTests, returnsProperDeviceName) {
    ASSERT_STREQ(getDeviceName(TargetDevice::eDefault), "Default");
    ASSERT_STREQ(getDeviceName(TargetDevice::eBalanced), "Balanced");
    ASSERT_STREQ(getDeviceName(TargetDevice::eCPU), "CPU");
    ASSERT_STREQ(getDeviceName(TargetDevice::eGPU), "GPU");
    ASSERT_STREQ(getDeviceName(TargetDevice::eFPGA), "FPGA");
    ASSERT_STREQ(getDeviceName(TargetDevice::eMYRIAD), "MYRIAD");
    ASSERT_STREQ(getDeviceName(TargetDevice::eGNA), "GNA");
    ASSERT_STREQ(getDeviceName(TargetDevice::eHETERO), "HETERO");
    ASSERT_STREQ(getDeviceName(static_cast<TargetDevice>(-1)), "Unknown device");
    //off by one test - might not be enough
    ASSERT_STREQ(getDeviceName(static_cast<TargetDevice>((uint8_t)TargetDevice::eHETERO + 1)), "Unknown device");
}
