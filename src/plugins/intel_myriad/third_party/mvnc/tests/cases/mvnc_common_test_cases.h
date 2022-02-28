// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if (defined(_WIN32) || defined(_WIN64))
#include "windows.h"
#endif

#include <gtest/gtest.h>
#include <chrono>

#include "mvnc.h"
#include "mvnc_ext.h"
#include "XLinkLog.h"
#include "mvnc_test_helper.h"

//------------------------------------------------------------------------------
//      Helpers
//------------------------------------------------------------------------------
constexpr std::chrono::seconds operator "" _sec(unsigned long long s)
{
    return std::chrono::seconds(s);
}

//------------------------------------------------------------------------------
//      class MvncTestsCommon
//------------------------------------------------------------------------------
class MvncTestsCommon : public ::testing::Test {
public:
    char    firmwareDir[MAX_PATH]  = {};
    mvLog_t ncLogLevel              = MVLOG_INFO;
    int     watchdogInterval        = 1000;
    int     availableDevices_       = 0;
    WatchdogHndl_t* m_watchdogHndl = nullptr;
    ncDeviceOpenParams_t m_ncDeviceOpenParams = {};

    MvncTestsCommon();
    ~MvncTestsCommon() override = default;
protected:

    void SetUp() override;
    void TearDown() override;

public:
    int setLogLevel(const mvLog_t logLevel);

    /**
     * @brief Boot and open selected amount of device
     * @param[out] amountOfBooted Amount of device which was booted
     * @param[out] deviceHandlers Pre-allocated array for handlers
     */
    void openDevices(const int devicesToBoot, ncDeviceHandle_t** deviceHandlers,
                     int& amountOfBooted);

    /**
     * @brief Load firmware to device
     * @warning Only USB devices is supported
     */
    virtual void bootOneDevice(ncDeviceProtocol_t deviceProtocol= NC_USB);

    // Firmware
    std::string getMyriadUSBFirmwarePath(const std::string& deviceName);
    std::string getMyriadFirmwarePath(const deviceDesc_t& in_deviceDesc);
};

//------------------------------------------------------------------------------
//      class MvncOpenDevice
//------------------------------------------------------------------------------
class MvncOpenDevice :  public MvncTestsCommon,
                        public testing::WithParamInterface<ncDeviceProtocol_t> {
protected:
    ncDeviceProtocol_t _deviceProtocol = NC_ANY_PROTOCOL;

    ~MvncOpenDevice() override = default;
    void SetUp() override;

};

//------------------------------------------------------------------------------
//      class MvncLoggingTests
//------------------------------------------------------------------------------
class MvncLoggingTests :  public MvncOpenDevice {
public:
    char buff[BUFSIZ] = {};
protected:
    ncDeviceHandle_t * _deviceHandles[MAX_DEVICES] = {nullptr};
    ncDeviceDescr_t _deviceDesc = {};

    void SetUp() override;
    void TearDown() override;
    ~MvncLoggingTests() override = default;
};

//------------------------------------------------------------------------------
//      class MvncGraphAllocations
//------------------------------------------------------------------------------
/**
 * @brief Test transfer data from host to device
 * @detail Allocate 2 devices and test some graph allocate cases
 * @warning For correct testing should be used blob with size more than 30mb
 */
class MvncGraphAllocations: public MvncOpenDevice {
public:
    // Devices
    ncDeviceHandle_t * _deviceHandle[MAX_DEVICES] = {nullptr};
    int _bootedDevices = 0;

    // Graphs
    ncGraphHandle_t*  _graphHandle[MAX_DEVICES] = {nullptr};

    // Blob
    const std::string blobPath = "bvlc_googlenet_fp16.blob";
    std::vector<char> _blob;
    bool blobLoaded = false;

protected:
    void SetUp() override;
    void TearDown() override;
    ~MvncGraphAllocations() override = default;
};

//------------------------------------------------------------------------------
//      class MvncCloseDevice
//------------------------------------------------------------------------------
class MvncCloseDevice : public MvncTestsCommon {
protected:
    ~MvncCloseDevice() override = default;
};

//------------------------------------------------------------------------------
//      Parametric tests initialization
//------------------------------------------------------------------------------
static const std::vector<ncDeviceProtocol_t> myriadProtocols = {
        NC_USB,
        NC_PCIE
};

namespace {
    /**
     * @brief   Converter from enum to string
     */
    struct PrintToStringParamName {
        std::string operator()(
                const ::testing::TestParamInfo<ncDeviceProtocol_t> &info) const {
            return ncProtocolToStr(info.param);
        }

        std::string operator()() const {
            return std::string("USB_");
        }
    };
}
