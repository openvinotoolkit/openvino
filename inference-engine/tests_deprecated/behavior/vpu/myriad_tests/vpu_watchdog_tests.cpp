// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <behavior_test_plugin.h>
#include <XLink.h>
#include <mvnc.h>
#include <mvnc/include/ncPrivateTypes.h>
#include <watchdog.h>
#include <watchdogPrivate.hpp>
#include <thread>
#include <file_utils.h>
#include "vpu_test_data.hpp"

#include "helpers/myriad_devices.hpp"
#include <details/ie_exception.hpp>

using namespace std;
using namespace ::testing;
using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace {
inline std::string getTestCaseName(testing::TestParamInfo<BehTestParams> obj) {
    return obj.param.device + "_" + obj.param.input_blob_precision.name()
        + (obj.param.config.size() ? "_" + obj.param.config.begin()->second : "");
}
}

#if (defined(_WIN32) || defined(_WIN64) )
extern "C" void initialize_usb_boot();
#else
#define initialize_usb_boot()
#endif


class MYRIADWatchdog :  public BehaviorPluginTest,
                        public MyriadDevicesInfo {
 public:
    WatchdogHndl_t* m_watchdogHndl = nullptr;
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;

    void SetUp() override {
        initialize_usb_boot();

        ASSERT_EQ(WD_ERRNO, watchdog_create(&m_watchdogHndl));
    }

    void TearDown() override {
        watchdog_destroy(m_watchdogHndl);
    }

    struct DevicesState {
        int booted = 0;
        int unbooted = 0;
        int total() const {return booted + unbooted;}
    };

    DevicesState queryDevices() {
        DevicesState devicesState;
        devicesState.booted = getAmountOfBootedDevices(NC_USB);
        devicesState.unbooted = getAmountOfUnbootedDevices(NC_USB);
        return devicesState;
    }

    ncDeviceHandle_t *device = nullptr;
    void resetOneDevice() {
        ncDeviceClose(&device, m_watchdogHndl);
        device = nullptr;
    }

    void bootOneDevice(int watchdogInterval, void* ptr_in_dll) {
        ncStatus_t statusOpen = NC_ERROR;
        std::cout << "Opening device" << std::endl;
#ifdef  _WIN32
        const char* pathToFw = nullptr;
#else
        std::string absPathToFw = getIELibraryPath();
        const char* pathToFw = absPathToFw.c_str();
#endif //  _WIN32

        ncDeviceDescr_t deviceDesc = {};
        deviceDesc.protocol = NC_ANY_PROTOCOL;
        deviceDesc.platform = NC_ANY_PLATFORM;

        ncDeviceOpenParams_t deviceOpenParams = {};
        deviceOpenParams.watchdogHndl = m_watchdogHndl;
        deviceOpenParams.watchdogInterval = watchdogInterval;
        deviceOpenParams.customFirmwareDirectory = pathToFw;

        statusOpen = ncDeviceOpen(&device, deviceDesc, deviceOpenParams);

        if (statusOpen != NC_OK) {
            ncDeviceClose(&device, m_watchdogHndl);
        }
    }
};


#define ASSERT_BOOTED_DEVICES_ONE_MORE() {\
    std::cout << "Time since boot:" << chrono::duration_cast<ms>(Time::now() - ctime).count() << std::endl;\
    auto q = queryDevices();\
    cout << "BOOTED=" << q.booted << "\n";\
    cout << "TOTAL=" << q.total() << "\n";\
    ASSERT_EQ(q.booted, startup_devices.booted + 1);\
    ASSERT_EQ(q.total(), startup_devices.total());\
}

#define ASSERT_BOOTED_DEVICES_SAME() {\
    std::cout << "Time since boot:" << chrono::duration_cast<ms>(Time::now() - ctime).count() << std::endl;\
    auto q = queryDevices();\
    cout << "BOOTED=" << q.booted << "\n";\
    cout << "TOTAL=" << q.total() << "\n";\
    ASSERT_EQ(q.booted, startup_devices.booted);\
    ASSERT_EQ(q.total(), startup_devices.total());\
}

TEST_P(MYRIADWatchdog, canDisableWatchdog) {

    auto startup_devices = queryDevices();
    ASSERT_GE(startup_devices.unbooted, 1);

    auto ctime = Time::now();
    SharedObjectLoader myriadPlg (make_plugin_name("myriadPlugin").c_str());
    void *p = myriadPlg.get_symbol(SOCreatorTrait<IInferencePlugin>::name);

    bootOneDevice(0,  p);

    ASSERT_BOOTED_DEVICES_ONE_MORE();

    // waiting while more that device side ping interval which is 12s
    for (int j = 0; j != 20; j++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "Time since boot:" << chrono::duration_cast<ms>(Time::now() - ctime).count() << std::endl;
        if (queryDevices().booted == startup_devices.booted) {
            SUCCEED() << "All devices gets reset";
            break;
        }
    }
    ASSERT_BOOTED_DEVICES_ONE_MORE();

    resetOneDevice();

    ASSERT_BOOTED_DEVICES_SAME();
}

TEST_P(MYRIADWatchdog, canDetectWhenHostSiteStalled) {
    auto startup_devices = queryDevices();
    ASSERT_GE(startup_devices.unbooted, 1);

    auto ctime = Time::now();

    SharedObjectLoader myriadPlg (make_plugin_name("myriadPlugin").c_str());
    void *p = myriadPlg.get_symbol(SOCreatorTrait<IInferencePlugin>::name);

    bootOneDevice(20000, p);

    //  due to increased ping interval device side of WD will abort execution
    ASSERT_BOOTED_DEVICES_ONE_MORE();

    // waiting while device understand that no ping request happened and reset itself
    for (int j = 0; j != 20; j++) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        std::cout << "Time since boot:" << chrono::duration_cast<ms>(Time::now() - ctime).count() << std::endl;
        if (queryDevices().booted == startup_devices.booted) {
            SUCCEED() << "All devices gets reset";
            break;
        }
    }
    // after watchdog reset device it requires some time to appear in system
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    ASSERT_BOOTED_DEVICES_SAME();

    resetOneDevice();
}

TEST_P(MYRIADWatchdog, watchDogIntervalDefault) {
    auto startup_devices = queryDevices();
    auto ctime = Time::now();
    {

        InferenceEngine::Core core;
        IE_SUPPRESS_DEPRECATED_START
        CNNNetReader reader;
        reader.ReadNetwork(GetParam().model_xml_str.data(), GetParam().model_xml_str.length());

        CNNNetwork network = reader.getNetwork();
        IE_SUPPRESS_DEPRECATED_END
        ASSERT_GE(startup_devices.unbooted, 1);

        ExecutableNetwork ret;
        ctime = Time::now();
        ASSERT_THROW(ret = core.LoadNetwork(network, GetParam().device, {
            {KEY_LOG_LEVEL, LOG_INFO}}),
            InferenceEngine::details::InferenceEngineException);

        ASSERT_BOOTED_DEVICES_ONE_MORE();

        // waiting while device understand that no ping request happened and reset itself
        for (int j = 0; j != 20; j++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::cout << "Time since boot:" << chrono::duration_cast<ms>(Time::now() - ctime).count() << std::endl;
            if (queryDevices().booted == startup_devices.booted) {
                SUCCEED() << "All devices gets reset";
                break;
            }
        }
        ASSERT_BOOTED_DEVICES_ONE_MORE();
    }
    // device willbe reset by unloaded plugin
    // after watchdog reset device it requires some time to appear in system
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    ASSERT_BOOTED_DEVICES_SAME();
}

TEST_P(MYRIADWatchdog, canTurnoffWatchDogViaConfig) {
    auto startup_devices = queryDevices();
    auto ctime = Time::now();
    {
        InferenceEngine::Core core;
        IE_SUPPRESS_DEPRECATED_START
        CNNNetReader reader;
        reader.ReadNetwork(GetParam().model_xml_str.data(), GetParam().model_xml_str.length());

        CNNNetwork network = reader.getNetwork();
        IE_SUPPRESS_DEPRECATED_END
        ASSERT_GE(startup_devices.unbooted, 1);

        ExecutableNetwork ret;
        ctime = Time::now();
        ASSERT_THROW(ret = core.LoadNetwork(network, GetParam().device, {
            {KEY_LOG_LEVEL, LOG_INFO},
            {KEY_VPU_MYRIAD_WATCHDOG, NO}}),
            InferenceEngine::details::InferenceEngineException);

        ASSERT_BOOTED_DEVICES_ONE_MORE();

        // waiting while device understand that no ping request happened and reset itself
        for (int j = 0; j != 20; j++) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            std::cout << "Time since boot:" << chrono::duration_cast<ms>(Time::now() - ctime).count() << std::endl;
            if (queryDevices().booted == startup_devices.booted) {
                SUCCEED() << "All devices gets reset";
                break;
            }
        }
        ASSERT_BOOTED_DEVICES_ONE_MORE();
    }
    // device will be reset by unloaded plugin
    // after watchdog reset device it requires some time to appear in system
    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    ASSERT_BOOTED_DEVICES_SAME();
}

const BehTestParams vpuValues[] = {
    BEH_MYRIAD,
};

INSTANTIATE_TEST_CASE_P(smoke_BehaviorTest, MYRIADWatchdog, ValuesIn(vpuValues), getTestCaseName);
