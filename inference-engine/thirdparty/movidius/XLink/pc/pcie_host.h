// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef PCIE_HOST_H
#define PCIE_HOST_H

#include "XLinkPlatform.h"
#include "XLinkPlatform_tool.h"

typedef enum {
    /* PCIE_PLATFORM_ANY_STATE intended for use in the device requirement,
    /  but also means an unknown state if we cannot get the device status */
    PCIE_PLATFORM_ANY_STATE = 0,
    PCIE_PLATFORM_BOOTED = 1,
    PCIE_PLATFORM_UNBOOTED = 2,
} pciePlatformState_t;

typedef enum {
    PCIE_HOST_SUCCESS = 0,
    PCIE_HOST_DEVICE_NOT_FOUND,
    PCIE_HOST_ERROR,
    PCIE_HOST_TIMEOUT,
    PCIE_HOST_DRIVER_NOT_LOADED
} pcieHostError_t;

int pcie_init(const char *slot, void **fd);
pcieHostError_t pcie_write(void *fd, void * buf, size_t bufSize, unsigned int timeout_ms);
pcieHostError_t pcie_read(void *fd, void *buf, size_t bufSize, unsigned int timeout_ms);
int pcie_close(void *fd);

/**
 *  @brief Get device name on index
 *  @param port_name   Port on which device is located.
 *                      If not empty, function will search for device with this name
 */
pcieHostError_t pcie_find_device_port(
    int index, char* port_name, int name_length, pciePlatformState_t requiredState);

/**
 * @brief Get state for pcie device on specified port
 */
pcieHostError_t pcie_get_device_state(
    const char * port_name, pciePlatformState_t* platformState);


#if (!defined(_WIN32) && !defined(_WIN64))
int pcie_reset_device(int fd);
int pcie_boot_device(int fd, void *buffer, size_t length);
#else
int pcie_reset_device(HANDLE fd);
int pcie_boot_device(HANDLE fd);
#endif
#endif  // PCIE_HOST_H
