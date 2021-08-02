// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior_test_plugin.h>
#include <XLink.h>
#include <mvnc.h>
#include <mvnc_ext.h>
#include "vpu_test_data.hpp"
#include "helpers/myriad_devices.hpp"

namespace {
    #define ASSERT_NO_ERROR(call)   ASSERT_EQ(call, 0)
    #define ASSERT_ERROR            ASSERT_TRUE

    const int MAX_DEVICES   = 32;
    const int MAX_DEV_NAME  = 255;

    std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
        return obj.param.pluginName + "_" + obj.param.input_blob_precision.name()
               + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
    }
}

#if (defined(_WIN32) || defined(_WIN64) )
extern "C" void initialize_usb_boot();
#else
#define initialize_usb_boot()
#endif

class MYRIADBoot : public MyriadDevicesInfo,
                   public BehaviorPluginTest {
 public:
#if !(defined(_WIN32) || defined(_WIN64))
    const char  firmwareDir[255] = "./lib/";
#else
    const char* firmwareDir = nullptr;
#endif

    void SetUp() override {
        initialize_usb_boot();
    }

    /*
     * @brief Boot any free device
     */
    void bootOneDevice() {
        ASSERT_NO_ERROR(ncDeviceLoadFirmware(NC_ANY_PLATFORM, firmwareDir));
    }

};

/*
 * @brief Boot myriad device through XLink, and then try to connect to it with plugin
 */
#if !(defined(_WIN32) || defined(_WIN64))   // TODO Issue-15574
TEST_P(MYRIADBoot, ConnectToAlreadyBootedDevice) {
#else
TEST_P(MYRIADBoot, DISABLED_ConnectToAlreadyBootedDevice) {
#endif
    bootOneDevice();
    ASSERT_EQ(getAmountOfBootedDevices(), 1);
    {
        Core ie;
        CNNNetwork network = ie.ReadNetwork(GetParam().model_xml_str, Blob::CPtr());
        ExecutableNetwork net = ie.LoadNetwork(network, GetParam().device,
            { {KEY_LOG_LEVEL, LOG_DEBUG},
              {InferenceEngine::MYRIAD_WATCHDOG, NO} });

        ASSERT_EQ(getAmountOfBootedDevices(), 1);
    }
    ncDeviceResetAll();
}

/*
 * @brief Check that with NO option plugin would boot new device
 * @warn  Test required two or more Myriad devices
 */
TEST_P(MYRIADBoot, DISABLED_OpenNotBootedDevice) {
    ASSERT_GE(getAmountOfUnbootedDevices(), 2);
    bootOneDevice();
    ASSERT_EQ(getAmountOfBootedDevices(), 1);
    {
        Core ie;
        CNNNetwork network = ie.ReadNetwork(GetParam().model_xml_str, Blob::CPtr());
        ExecutableNetwork net = ie.LoadNetwork(network, GetParam().device,
            { {KEY_LOG_LEVEL, LOG_DEBUG},
              {InferenceEngine::MYRIAD_WATCHDOG, NO} });

        ASSERT_EQ(getAmountOfBootedDevices(), 2);
    }
    ncDeviceResetAll();
}

const BehTestParams vpuValues[] = {
        BEH_MYRIAD,
};

INSTANTIATE_TEST_SUITE_P(smoke_BehaviorTest, MYRIADBoot, ValuesIn(vpuValues), getTestCaseName);
