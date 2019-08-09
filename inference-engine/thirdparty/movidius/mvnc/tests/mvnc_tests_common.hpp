// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#if (defined(_WIN32) || defined(_WIN64))
#include "windows.h"
#endif

#include <thread>
#include <gtest/gtest.h>
#include <fstream>

#include "XLink.h"
#include "mvnc.h"
#include "mvnc_ext.h"
#include "mvnc_data.h"
#include "mvLog.h"
#include "usb_boot.h"
#include "ncPrivateTypes.h"
#include "ncCommPrivate.h"


///// Macroses
#define ASSERT_NO_ERROR(call)   ASSERT_EQ(call, 0)
#define ASSERT_ERROR(call)      ASSERT_TRUE(call)


//// Defines
#define MYRIAD_X_NAME_STR "ma2480"
#define MYRIAD_2_NAME_STR "ma2450"

#if (defined(_WIN32) || defined(_WIN64))
#define PCIE_NAME_STR     "mxlink"
#else
#define PCIE_NAME_STR     "mxlk"
#endif

const int MAX_DEVICES = 32;
const int MAX_DEV_NAME = 20;

#ifndef MAX_PATH
const int MAX_PATH = 255;
#endif

//// Usb initialization
// Without this initialization find device on windows could not work
#if (defined(_WIN32) || defined(_WIN64) )
extern "C" void initialize_usb_boot();
#else
#define initialize_usb_boot()
#endif

class MvncTestsCommon : public ::testing::Test {
public:
    char        firmwarePath[MAX_PATH];
    mvLog_t     ncLogLevel;
    int         watchdogInterval;

    ~MvncTestsCommon() override = default;

    MvncTestsCommon() : ncLogLevel(MVLOG_INFO), watchdogInterval(1000) {
#if !(defined(_WIN32) || defined(_WIN64))
        // On linux we should use custom path to firmware due to another searching mechanism for library
        strcpy(firmwarePath, "./lib");
#else
        firmwarePath[0] = 0;
#endif
    }
protected:

    void SetUp() override {
        initialize_usb_boot();
        ASSERT_NO_ERROR(setLogLevel(ncLogLevel));
    }

    void TearDown() override {
        ncDeviceResetAll();
    }

public:
    int setLogLevel(const mvLog_t logLevel) {
        ncStatus_t status = ncGlobalSetOption(NC_RW_LOG_LEVEL, &logLevel,
                                              sizeof(logLevel));
        if (status != NC_OK) {
            fprintf(stderr,
                    "WARNING: failed to set log level: %d with error: %d\n",
                    ncLogLevel, status);
            return -1;
        }
        ncLogLevel = logLevel;
        return 0;
    }

    /**
     * @brief Get amount of all currently connected Myriad devices
     * @param[in] deviceProtocol Count only platform specific devices
     */
    static int getAmountOfDevices(const ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL) {
        int amount = 0;
        deviceDesc_t deviceDesc = {};
        deviceDesc_t in_deviceDesc = {};
        in_deviceDesc.protocol = convertProtocolToXlink(deviceProtocol);
        in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

        if(in_deviceDesc.protocol == X_LINK_USB_VSC) {
            for (; amount < MAX_DEVICES; ++amount) {
                if (XLinkFindDevice(amount, X_LINK_ANY_STATE, &in_deviceDesc, &deviceDesc))
                    break;
            }
            return amount;
        }

        if (XLinkFindDevice(amount, X_LINK_ANY_STATE, &in_deviceDesc, &deviceDesc) == X_LINK_SUCCESS) {
            return ++amount;
        }

        return amount;
    }

    /**
     * @brief Boot and open selected amount of device
     * @param[out] amountOfBooted Amount of device which was booted
     * @param[out] deviceHandlers Pre-allocated array for handlers
     */
    void openDevices(const int devicesToBoot, ncDeviceHandle_t** deviceHandlers,
            int& amountOfBooted,
            const ncDeviceProtocol_t protocol = NC_ANY_PROTOCOL)
    {
        ASSERT_TRUE(deviceHandlers != nullptr);
        const int availableDevices = getAmountOfDevices(NC_USB);
        if (availableDevices < devicesToBoot) {
            GTEST_SKIP_("Not enough devices");
        }

        amountOfBooted = 0;
        ncDeviceDescr_t ncDeviceDesc = {};
        ncDeviceDesc.protocol = NC_USB;
        ncDeviceDesc.platform = NC_ANY_PLATFORM;

        for (int index = 0; index < devicesToBoot; ++index) {
            ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandlers[index], ncDeviceDesc, watchdogInterval, firmwarePath));
            ASSERT_TRUE(deviceHandlers[index] != nullptr);
            ++amountOfBooted;
        }
        ASSERT_EQ(amountOfBooted, devicesToBoot) << "Not all devices was loaded";
    }

    /**
     * @brief Load firmware to device
     * @warning Only USB devices is supported
     */
    virtual void bootOneDevice(ncDeviceProtocol_t deviceProtocol= NC_USB) {
        if (deviceProtocol == NC_PCIE) {
            GTEST_FATAL_FAILURE_("Boot doesn't supported for PCIe protocol\n");
        }
        ASSERT_NO_ERROR(ncDeviceLoadFirmware(NC_ANY_PLATFORM, firmwarePath));
    }

    /**
     * @brief Get list of all currently connected Myriad devices
     */
    static std::vector<std::string> getDevicesList() {
        std::vector < std::string > devName;
        deviceDesc_t tempDeviceDesc = {};
        deviceDesc_t in_deviceDesc = {};
        in_deviceDesc.protocol = X_LINK_USB_VSC;
        in_deviceDesc.platform = X_LINK_ANY_PLATFORM;

        for (int i = 0; i < MAX_DEVICES; ++i) {
            if (XLinkFindDevice(i, X_LINK_ANY_STATE, &in_deviceDesc, &tempDeviceDesc))
                break;
            devName.emplace_back(tempDeviceDesc.name);
        }

        in_deviceDesc.protocol = X_LINK_PCIE;
        /// PCIe don't use indexes and always return same device
        if (XLinkFindDevice(0, X_LINK_ANY_STATE, &in_deviceDesc, &tempDeviceDesc) == 0)
            devName.emplace_back(tempDeviceDesc.name);

        return devName;
    }

    static bool isMyriadXUSBDevice(const std::string &deviceName) {
        return (deviceName.find(MYRIAD_X_NAME_STR) != std::string::npos);
    }

    static bool isMyriad2USBDevice(const std::string &deviceName) {
        return (deviceName.find(MYRIAD_2_NAME_STR) != std::string::npos);
    }

    static bool isMyriadPCIeDevice(const std::string& deviceName) {
        return deviceName.find(std::string(PCIE_NAME_STR)) != std::string::npos;
    }

    /**
     * @warning The booted USB device will also be counted here.
     */
    static bool isMyriadUSBDevice(const std::string& deviceName) {
        return (isMyriad2USBDevice(deviceName)
                    || isMyriadXUSBDevice(deviceName)
                    || isMyriadBootedUSBDevice(deviceName));
    }

    static bool isMyriadBootedUSBDevice(const std::string &deviceName) {
        return (!isMyriad2USBDevice(deviceName) &&
                    !isMyriadXUSBDevice(deviceName) &&
                    !isMyriadPCIeDevice(deviceName));
    }

    /**
     * @brief Check that device matches the specified protocol
     */
    static bool isSameProtocolDevice(const std::string &deviceName,
            const ncDeviceProtocol_t expectedProtocol) {
        switch (expectedProtocol) {
            case NC_USB:            return isMyriadUSBDevice(deviceName);
            case NC_PCIE:           return isMyriadPCIeDevice(deviceName);
            case NC_ANY_PROTOCOL:
                return isMyriadPCIeDevice(deviceName) || isMyriadUSBDevice(deviceName);
            default:
                std::cout << "Unknown device protocol" << std::endl;
                return false;
        }
    }

    /**
    * @brief Check that device matches the specified protocol
    */
    static bool isSamePlatformDevice(const std::string &deviceName,
                                     const ncDevicePlatform_t expectedPlatform) {
        switch (expectedPlatform) {
            case NC_MYRIAD_2:  return isMyriad2USBDevice(deviceName);
            case NC_MYRIAD_X:  return isMyriadXUSBDevice(deviceName);
            case NC_ANY_PLATFORM:
                return isMyriad2USBDevice(deviceName) || isMyriadXUSBDevice(deviceName);
            default:
                std::cout << "Unknown device platform" << std::endl;
                return false;
        }
    }

    static long getAmountOfMyriadXDevices() {
        auto devName = getDevicesList();
        return count_if(devName.begin(), devName.end(), isMyriadXUSBDevice);
    }

    static long getAmountOfMyriad2Devices() {
        auto devName = getDevicesList();
        return count_if(devName.begin(), devName.end(), isMyriad2USBDevice);
    }

    static long getAmountOfBootedDevices() {
        auto devName = getDevicesList();
        return count_if(devName.begin(), devName.end(), isMyriadBootedUSBDevice);
    }

    static long getAmountOfNotBootedDevices() {
        return (getAmountOfMyriadXDevices() + getAmountOfMyriad2Devices());
    }

    static long getAmountOfPCIeDevices() {
        return getAmountOfDevices(NC_PCIE);
    }

    static long getAmountOfUSBDevices() {
        return getAmountOfDevices(NC_USB);
    }


    /**
     * @brief   Read blob
     * @param   fileName Path to blob from bin directory
     * @return  True if blob is readed without problem
     */
    bool readBINFile(const std::string& fileName, std::vector<char>& buf) {
        std::ifstream file(fileName, std::ios_base::binary | std::ios_base::ate);
        if (!file.is_open()) {
            std::cout << "Can't open file!" << std::endl;
            return false;
        }
        buf.resize(static_cast<unsigned int>(file.tellg()));
        file.seekg(0);
        file.read(buf.data(), buf.size());
        return true;
    }
};

/// Parametric tests initialization

static const std::vector<ncDeviceProtocol_t> myriadProtocols = {
        NC_USB,
        NC_PCIE
};

static const std::vector<ncDevicePlatform_t> myriadPlatforms = {
        NC_MYRIAD_2,
        NC_MYRIAD_X
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

        std::string operator()(
                const ::testing::TestParamInfo<ncDevicePlatform_t> &info) const {
            return std::string("USB_") + ncPlatformToStr(info.param);
        }
    };
}