// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <string.h>

#include "XLinkPlatform.h"
#include "XLinkPlatformTool.h"
#include "usb_boot.h"
#include "pcie_host.h"
#include "mvStringUtils.h"

#define MVLOG_UNIT_NAME PlatformDeviceSearch
#include "mvLog.h"

// ------------------------------------
// Helpers declaration. Begin.
// ------------------------------------

static int platformToPid(const XLinkPlatform_t platform, const XLinkDeviceState_t state);
static pciePlatformState_t xlinkDeviceStateToPciePlatformState(const XLinkDeviceState_t state);

static xLinkPlatformErrorCode_t parseUsbBootError(usbBootError_t rc);
static xLinkPlatformErrorCode_t parsePCIeHostError(pcieHostError_t rc);

static xLinkPlatformErrorCode_t getUSBDeviceName(int index,
                                                 XLinkDeviceState_t state,
                                                 const deviceDesc_t in_deviceRequirements,
                                                 deviceDesc_t* out_foundDevice);
static xLinkPlatformErrorCode_t getPCIeDeviceName(int index,
                                                  XLinkDeviceState_t state,
                                                  const deviceDesc_t in_deviceRequirements,
                                                  deviceDesc_t* out_foundDevice);

// ------------------------------------
// Helpers declaration. End.
// ------------------------------------


// ------------------------------------
// XLinkPlatform API implementation. Begin.
// ------------------------------------

xLinkPlatformErrorCode_t XLinkPlatformFindDeviceName(XLinkDeviceState_t state,
                                                     const deviceDesc_t in_deviceRequirements,
                                                     deviceDesc_t* out_foundDevice) {
    memset(out_foundDevice, 0, sizeof(deviceDesc_t));
    xLinkPlatformErrorCode_t USB_rc;
    xLinkPlatformErrorCode_t PCIe_rc;

    switch (in_deviceRequirements.protocol){
        case X_LINK_USB_CDC:
        case X_LINK_USB_VSC:
            return getUSBDeviceName(0, state, in_deviceRequirements, out_foundDevice);

        case X_LINK_PCIE:
            return getPCIeDeviceName(0, state, in_deviceRequirements, out_foundDevice);

        case X_LINK_ANY_PROTOCOL:
            USB_rc = getUSBDeviceName(0, state, in_deviceRequirements, out_foundDevice);
            if (USB_rc == X_LINK_PLATFORM_SUCCESS) {      // Found USB device, return it
                return X_LINK_PLATFORM_SUCCESS;
            }
            if (USB_rc != X_LINK_PLATFORM_DEVICE_NOT_FOUND) {   // Issue happen, log it
                mvLog(MVLOG_DEBUG, "USB find device failed with rc: %s",
                      XLinkPlatformErrorToStr(USB_rc));
            }

            // Try to find PCIe device
            memset(out_foundDevice, 0, sizeof(deviceDesc_t));
            PCIe_rc = getPCIeDeviceName(0, state, in_deviceRequirements, out_foundDevice);
            if (PCIe_rc == X_LINK_PLATFORM_SUCCESS) {     // Found PCIe device, return it
                return X_LINK_PLATFORM_SUCCESS;
            }
            if (PCIe_rc != X_LINK_PLATFORM_DEVICE_NOT_FOUND) {   // Issue happen, log it
                mvLog(MVLOG_DEBUG, "PCIe find device failed with rc: %s",
                      XLinkPlatformErrorToStr(PCIe_rc));
            }
            return X_LINK_PLATFORM_DEVICE_NOT_FOUND;

        default:
            mvLog(MVLOG_WARN, "Unknown protocol");
            return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
    }
}

xLinkPlatformErrorCode_t XLinkPlatformFindArrayOfDevicesNames(
    XLinkDeviceState_t state,
    const deviceDesc_t in_deviceRequirements,
    deviceDesc_t* out_foundDevice,
    const unsigned int devicesArraySize,
    unsigned int *out_amountOfFoundDevices) {

    memset(out_foundDevice, 0, sizeof(deviceDesc_t) * devicesArraySize);

    unsigned int usb_index = 0;
    unsigned int pcie_index = 0;
    unsigned int both_protocol_index = 0;

    // TODO Handle possible errors
    switch (in_deviceRequirements.protocol){
        case X_LINK_USB_CDC:
        case X_LINK_USB_VSC:
            while(getUSBDeviceName(
                usb_index, state, in_deviceRequirements, &out_foundDevice[usb_index]) ==
                  X_LINK_PLATFORM_SUCCESS) {
                ++usb_index;
            }

            *out_amountOfFoundDevices = usb_index;
            return X_LINK_PLATFORM_SUCCESS;

        case X_LINK_PCIE:
            while(getPCIeDeviceName(
                pcie_index, state, in_deviceRequirements, &out_foundDevice[pcie_index]) ==
                  X_LINK_PLATFORM_SUCCESS) {
                ++pcie_index;
            }

            *out_amountOfFoundDevices = pcie_index;
            return X_LINK_PLATFORM_SUCCESS;

        case X_LINK_ANY_PROTOCOL:
            while(getUSBDeviceName(
                usb_index, state, in_deviceRequirements,
                &out_foundDevice[both_protocol_index]) ==
                  X_LINK_PLATFORM_SUCCESS) {
                ++usb_index;
                ++both_protocol_index;
            }
            while(getPCIeDeviceName(
                pcie_index, state, in_deviceRequirements,
                &out_foundDevice[both_protocol_index]) ==
                  X_LINK_PLATFORM_SUCCESS) {
                ++pcie_index;
                ++both_protocol_index;
            }
            *out_amountOfFoundDevices = both_protocol_index;
            return X_LINK_PLATFORM_SUCCESS;

        default:
            mvLog(MVLOG_WARN, "Unknown protocol");
            return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
    }
}

char* XLinkPlatformErrorToStr(const xLinkPlatformErrorCode_t errorCode) {
    switch (errorCode) {
        case X_LINK_PLATFORM_SUCCESS: return "X_LINK_PLATFORM_SUCCESS";
        case X_LINK_PLATFORM_DEVICE_NOT_FOUND: return "X_LINK_PLATFORM_DEVICE_NOT_FOUND";
        case X_LINK_PLATFORM_ERROR: return "X_LINK_PLATFORM_ERROR";
        case X_LINK_PLATFORM_TIMEOUT: return "X_LINK_PLATFORM_TIMEOUT";
        case X_LINK_PLATFORM_DRIVER_NOT_LOADED: return "X_LINK_PLATFORM_DRIVER_NOT_LOADED";
        default: return "";
    }
}

XLinkPlatform_t XLinkPlatformPidToPlatform(const int pid) {
    switch (pid) {
        case DEFAULT_UNBOOTPID_2150: return X_LINK_MYRIAD_2;
        case DEFAULT_UNBOOTPID_2485: return X_LINK_MYRIAD_X;
        default:       return X_LINK_ANY_PLATFORM;
    }
}

XLinkDeviceState_t XLinkPlatformPidToState(const int pid) {
    switch (pid) {
        case DEFAULT_OPENPID: return X_LINK_BOOTED;
        case AUTO_PID: return X_LINK_ANY_STATE;
        default:       return X_LINK_UNBOOTED;
    }
}

// ------------------------------------
// XLinkPlatform API implementation. End.
// ------------------------------------



// ------------------------------------
// Helpers implementation. Begin.
// ------------------------------------

int platformToPid(const XLinkPlatform_t platform, const XLinkDeviceState_t state) {
    if (state == X_LINK_UNBOOTED) {
        switch (platform) {
            case X_LINK_MYRIAD_2:  return DEFAULT_UNBOOTPID_2150;
            case X_LINK_MYRIAD_X:  return DEFAULT_UNBOOTPID_2485;
            default:               return AUTO_UNBOOTED_PID;
        }
    } else if (state == X_LINK_BOOTED) {
        return DEFAULT_OPENPID;
    } else if (state == X_LINK_ANY_STATE) {
        switch (platform) {
            case X_LINK_MYRIAD_2:  return DEFAULT_UNBOOTPID_2150;
            case X_LINK_MYRIAD_X:  return DEFAULT_UNBOOTPID_2485;
            default:               return AUTO_PID;
        }
    }

    return AUTO_PID;
}

pciePlatformState_t xlinkDeviceStateToPciePlatformState(const XLinkDeviceState_t state) {
    switch (state) {
        case X_LINK_ANY_STATE:  return PCIE_PLATFORM_ANY_STATE;
        case X_LINK_BOOTED:     return PCIE_PLATFORM_BOOTED;
        case X_LINK_UNBOOTED:   return PCIE_PLATFORM_UNBOOTED;
        default:
            return PCIE_PLATFORM_ANY_STATE;
    }
}

xLinkPlatformErrorCode_t parseUsbBootError(usbBootError_t rc) {
    switch (rc) {
        case USB_BOOT_SUCCESS:
            return X_LINK_PLATFORM_SUCCESS;
        case USB_BOOT_DEVICE_NOT_FOUND:
            return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
        case USB_BOOT_TIMEOUT:
            return X_LINK_PLATFORM_TIMEOUT;
        default:
            return X_LINK_PLATFORM_ERROR;
    }
}

xLinkPlatformErrorCode_t parsePCIeHostError(pcieHostError_t rc) {
    switch (rc) {
        case PCIE_HOST_SUCCESS:
            return X_LINK_PLATFORM_SUCCESS;
        case PCIE_HOST_DEVICE_NOT_FOUND:
            return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
        case PCIE_HOST_ERROR:
            return X_LINK_PLATFORM_ERROR;
        case PCIE_HOST_TIMEOUT:
            return X_LINK_PLATFORM_TIMEOUT;
        case PCIE_HOST_DRIVER_NOT_LOADED:
            return X_LINK_PLATFORM_DRIVER_NOT_LOADED;
        default:
            return X_LINK_PLATFORM_ERROR;
    }
}

xLinkPlatformErrorCode_t getUSBDeviceName(int index,
                                                 XLinkDeviceState_t state,
                                                 const deviceDesc_t in_deviceRequirements,
                                                 deviceDesc_t* out_foundDevice) {
    ASSERT_X_LINK_PLATFORM(index >= 0);
    ASSERT_X_LINK_PLATFORM(out_foundDevice);

    int vid = AUTO_VID;
    int pid = AUTO_PID;

    char name[XLINK_MAX_NAME_SIZE] = { 0 };

    int searchByName = 0;
    if (strlen(in_deviceRequirements.name) > 0) {
        searchByName = 1;
        mv_strcpy(name, XLINK_MAX_NAME_SIZE, in_deviceRequirements.name);
    }

    // Set PID
    if (state == X_LINK_BOOTED) {
        if (in_deviceRequirements.platform != X_LINK_ANY_PLATFORM) {
            mvLog(MVLOG_WARN, "Search specific platform for booted device unavailable");
            return X_LINK_PLATFORM_ERROR;
        }
        pid = DEFAULT_OPENPID;
    } else {
        if (searchByName) {
            pid = get_pid_by_name(in_deviceRequirements.name);
        } else {
            pid = platformToPid(in_deviceRequirements.platform, state);
        }
    }

#if (!defined(_WIN32) && !defined(_WIN64))
    uint16_t  bcdusb = -1;
    usbBootError_t rc = usb_find_device_with_bcd(
        index, name, XLINK_MAX_NAME_SIZE, 0, vid, pid, &bcdusb);
#else
    usbBootError_t rc = usb_find_device(
                index, name, XLINK_MAX_NAME_SIZE, 0, vid, pid);
#endif
    xLinkPlatformErrorCode_t xLinkRc = parseUsbBootError(rc);
    if(xLinkRc == X_LINK_PLATFORM_SUCCESS)
    {
        mv_strcpy(out_foundDevice->name, XLINK_MAX_NAME_SIZE, name);
        out_foundDevice->protocol = X_LINK_USB_VSC;
        out_foundDevice->platform = XLinkPlatformPidToPlatform(get_pid_by_name(name));
    }
    return xLinkRc;
}

xLinkPlatformErrorCode_t getPCIeDeviceName(int index,
                                                  XLinkDeviceState_t state,
                                                  const deviceDesc_t in_deviceRequirements,
                                                  deviceDesc_t* out_foundDevice) {
    ASSERT_X_LINK_PLATFORM(index >= 0);
    ASSERT_X_LINK_PLATFORM(out_foundDevice);
    ASSERT_X_LINK_PLATFORM(in_deviceRequirements.platform != X_LINK_MYRIAD_2);

    char name[XLINK_MAX_NAME_SIZE] = { 0 };

    if (strlen(in_deviceRequirements.name) > 0) {
        mv_strcpy(name, XLINK_MAX_NAME_SIZE, in_deviceRequirements.name);
    }

    pcieHostError_t rc = pcie_find_device_port(
        index, name, XLINK_MAX_NAME_SIZE, xlinkDeviceStateToPciePlatformState(state));

    xLinkPlatformErrorCode_t xLinkRc = parsePCIeHostError(rc);
    if(xLinkRc == X_LINK_PLATFORM_SUCCESS)
    {
        mv_strcpy(out_foundDevice->name, XLINK_MAX_NAME_SIZE, name);
        out_foundDevice->protocol = X_LINK_PCIE;
        out_foundDevice->platform = X_LINK_MYRIAD_X;
    }
    return xLinkRc;
}

// ------------------------------------
// Helpers implementation. End.
// ------------------------------------
