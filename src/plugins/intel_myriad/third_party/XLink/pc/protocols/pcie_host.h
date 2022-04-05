// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PCIE_HOST_H
#define PCIE_HOST_H

#include "XLinkPlatform.h"

// ------------------------------------
//          PCIe Enums
// ------------------------------------
typedef enum {
    /* PCIE_PLATFORM_ANY_STATE intended for use in the device requirement,
    /  but also means an unknown state if we cannot get the device status */
    PCIE_PLATFORM_ANY_STATE = 0,
    PCIE_PLATFORM_BOOTED = 1,
    PCIE_PLATFORM_UNBOOTED = 2,
} pciePlatformState_t;

typedef enum {
    PCIE_HOST_SUCCESS = 0,
    PCIE_HOST_DEVICE_NOT_FOUND = -1,
    PCIE_HOST_ERROR = -2,
    PCIE_HOST_TIMEOUT = -3,
    PCIE_HOST_DRIVER_NOT_LOADED = -4,
    PCIE_INVALID_PARAMETERS = -5
} pcieHostError_t;

// ------------------------------------
//          PCIe functions
// ------------------------------------

/**
 * @brief       Open device on specified slot
 * @param[in]   slot - device address
 * @param[out]  fd   - Opened filedescriptor
 */
pcieHostError_t pcie_init(const char *slot, void **fd);

pcieHostError_t pcie_close(void *fd);

#if (!defined(_WIN32))
pcieHostError_t pcie_boot_device(int fd, const char  *buffer, size_t length);
pcieHostError_t pcie_reset_device(int fd);

#else // Windows
pcieHostError_t pcie_boot_device(HANDLE fd, const char  *buffer, size_t length);
pcieHostError_t pcie_reset_device(HANDLE fd);
#endif

int pcie_write(void *fd, void * buf, size_t bufSize);

int pcie_read(void *fd, void *buf, size_t bufSize);


/**
 *  @brief Get device name on index
 *  @param port_name   Port on which device is located.
 *                      If not empty, function will search for device with this name
 */
pcieHostError_t pcie_find_device_port(
    int index, char* port_name, int name_length, pciePlatformState_t requiredState);

// ------------------------------------
//       PCIE Driver specific calls
// ------------------------------------
/**
 * @brief Get state for PCIe device on specified port
 */
pcieHostError_t pcie_get_device_state(
        const char *port_name, pciePlatformState_t *platformState);

#endif  // PCIE_HOST_H
