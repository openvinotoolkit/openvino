// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mvnc_common_test_cases.h"

//------------------------------------------------------------------------------
//      Implementation of class MvncTestsCommon
//------------------------------------------------------------------------------
MvncTestsCommon::MvncTestsCommon() {
#if !(defined(_WIN32) || defined(_WIN64))
    // On linux we should use custom path to firmware due to another searching mechanism for library
    strcpy(firmwareDir, "./lib/");
#else
    strcpy(firmwareDir, "./");
#endif
}

void MvncTestsCommon::SetUp() {
    initialize_usb_boot();
    ASSERT_NO_ERROR(setLogLevel(ncLogLevel));
    availableDevices_ = getAmountOfDevices();

    ASSERT_EQ(WD_ERRNO, watchdog_create(&m_watchdogHndl));

    m_ncDeviceOpenParams.watchdogInterval = watchdogInterval;
    m_ncDeviceOpenParams.customFirmwareDirectory = firmwareDir;
    m_ncDeviceOpenParams.watchdogHndl = m_watchdogHndl;
}

void MvncTestsCommon::TearDown() {
    ncDeviceResetAll();
    watchdog_destroy(m_watchdogHndl);
}

int MvncTestsCommon::setLogLevel(const mvLog_t logLevel) {
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

void MvncTestsCommon::openDevices(const int devicesToBoot, ncDeviceHandle_t **deviceHandlers,
                                  int &amountOfBooted) {
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
        ASSERT_NO_ERROR(ncDeviceOpen(&deviceHandlers[index], ncDeviceDesc, m_ncDeviceOpenParams));
        ASSERT_TRUE(deviceHandlers[index] != nullptr);
        ++amountOfBooted;
    }
    ASSERT_EQ(amountOfBooted, devicesToBoot) << "Not all devices was loaded";
}

void MvncTestsCommon::bootOneDevice(ncDeviceProtocol_t deviceProtocol) {
    if (deviceProtocol == NC_PCIE) {
        GTEST_FATAL_FAILURE_("Boot doesn't supported for PCIe protocol\n");
    }
    ASSERT_NO_ERROR(ncDeviceLoadFirmware(NC_ANY_PLATFORM, firmwareDir));
}


std::string MvncTestsCommon::getMyriadUSBFirmwarePath(const std::string& deviceName) {
    if (deviceName.find('-') == std::string::npos) {
        throw std::invalid_argument("Invalid device address");
    }

    std::string firmwareName = "usb-ma2450.mvcmd";
    if (deviceName.find("ma2480") != std::string::npos) {
        firmwareName = "usb-ma2x8x.mvcmd";
    }

    return firmwareDir + firmwareName;
}

std::string MvncTestsCommon::getMyriadFirmwarePath(const deviceDesc_t& in_deviceDesc) {
    if(in_deviceDesc.protocol != X_LINK_USB_VSC &&
       in_deviceDesc.protocol != X_LINK_PCIE) {
        throw std::invalid_argument("Device protocol must be specified");
    }

    if(in_deviceDesc.protocol == X_LINK_PCIE) {
#if defined(_WIN32)
        const std::string extension = "elf";
#else
        const std::string extension = "mvcmd";
#endif
        return firmwareDir + std::string("pcie-ma248x.") + extension;
    }

    return getMyriadUSBFirmwarePath(in_deviceDesc.name);
}

//------------------------------------------------------------------------------
//      Implementation of class MvncOpenDevice
//------------------------------------------------------------------------------
void MvncOpenDevice::SetUp() {
    MvncTestsCommon::SetUp();

    _deviceProtocol = GetParam();
    availableDevices_ = getAmountOfDevices(_deviceProtocol);
}

//------------------------------------------------------------------------------
//      Implementation of class MvncLoggingTests
//------------------------------------------------------------------------------
void MvncLoggingTests::SetUp() {
    MvncOpenDevice::SetUp();

    _deviceDesc.protocol = _deviceProtocol;
    _deviceDesc.platform = NC_ANY_PLATFORM;

    for (int index = 0; index < availableDevices_; ++index) {
        ASSERT_NO_ERROR(ncDeviceOpen(&_deviceHandles[index], _deviceDesc, m_ncDeviceOpenParams));
    }

    setbuf(stdout, buff);
    fprintf(stdout, "[workaround for getting full content from XLink]\n");
}

void MvncLoggingTests::TearDown() {
    setbuf(stdout, NULL);
    for (int index = 0; index < availableDevices_; ++index) {
        ASSERT_NO_ERROR(ncDeviceClose(&_deviceHandles[index], m_watchdogHndl));
    }
}

//------------------------------------------------------------------------------
//      Implementation of class MvncGraphAllocations
//------------------------------------------------------------------------------
void MvncGraphAllocations::SetUp() {
    MvncOpenDevice::SetUp();

    // Load blob
    blobLoaded = readBINFile(blobPath, _blob);
    if (!blobLoaded) {
        std::cout << blobPath << " blob for test not found\n";
    }
}

void MvncGraphAllocations::TearDown() {
    for (int index = 0; index < _bootedDevices; ++index) {
        ASSERT_NO_ERROR(ncDeviceClose(&_deviceHandle[index], m_watchdogHndl));
    }
    _bootedDevices = 0;
}
