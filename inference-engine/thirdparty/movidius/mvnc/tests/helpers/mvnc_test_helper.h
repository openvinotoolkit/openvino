// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <string>
#include <fstream>

#include "XLink.h"
#include "mvnc.h"

//------------------------------------------------------------------------------
//      Macroses
//------------------------------------------------------------------------------
#define ASSERT_NO_ERROR(call)   ASSERT_EQ(call, 0)
#define ASSERT_ERROR(call)      ASSERT_TRUE(call)


//------------------------------------------------------------------------------
//      Defines
//------------------------------------------------------------------------------
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

//------------------------------------------------------------------------------
//      Usb initialization
//------------------------------------------------------------------------------
// Without this initialization find device on windows could not work
#if (defined(_WIN32) || defined(_WIN64) )
extern "C" void initialize_usb_boot();
#else
#define initialize_usb_boot()
#endif


//------------------------------------------------------------------------------
//      Helpers - counters
//------------------------------------------------------------------------------
/**
     * @brief Get amount of all currently connected Myriad devices
     * @param[in] deviceProtocol Count only platform specific devices
     */
int getAmountOfDevices(const ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL,
                              const ncDevicePlatform_t devicePlatform = NC_ANY_PLATFORM,
                              const XLinkDeviceState_t state = X_LINK_ANY_STATE);

long getAmountOfMyriadXDevices(ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL);

long getAmountOfMyriad2Devices(ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL);

long getAmountOfBootedDevices(ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL);

long getAmountOfNotBootedDevices(ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL);

long getAmountOfPCIeDevices();

long getAmountOfUSBDevices();

//------------------------------------------------------------------------------
//      Helpers - get devices
//------------------------------------------------------------------------------
/**
 * @brief Get list of all currently connected Myriad devices
 */
std::vector<std::string> getDevicesList(
        const ncDeviceProtocol_t deviceProtocol = NC_ANY_PROTOCOL,
        const ncDevicePlatform_t devicePlatform = NC_ANY_PLATFORM,
        const XLinkDeviceState_t state = X_LINK_ANY_STATE);

//------------------------------------------------------------------------------
//      Helpers - comparators
//------------------------------------------------------------------------------
bool isMyriadXUSBDevice(const std::string &deviceName);

bool isMyriad2USBDevice(const std::string &deviceName);

bool isMyriadPCIeDevice(const std::string& deviceName);

/**
     * @warning The booted USB device will also be counted here.
     */
bool isMyriadUSBDevice(const std::string& deviceName);

bool isMyriadBootedUSBDevice(const std::string &deviceName);

/**
 * @brief Check that device matches the specified protocol
 */
bool isSameProtocolDevice(const std::string &deviceName,
                                 const ncDeviceProtocol_t expectedProtocol);

/**
* @brief Check that device matches the specified platform for USB
*/
bool isSamePlatformUSBDevice(const std::string &deviceName,
                                    const ncDevicePlatform_t expectedPlatform);

//------------------------------------------------------------------------------
//      Helpers - file loader
//------------------------------------------------------------------------------
/**
 * @brief   Read blob
 * @param   fileName Path to blob from bin directory
 * @return  True if blob is readed without problem
 */
bool readBINFile(const std::string& fileName, std::vector<char>& buf);
