// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "XLinkPlatform.h"

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <fcntl.h>
#include <ctype.h>

#if (defined(_WIN32) || defined(_WIN64))
#include <windows.h>
#include "gettime.h"
#include <setupapi.h>
#include <strsafe.h>
#include <cfgmgr32.h>
#include <tchar.h>
#else
#include <sys/types.h>
#include <unistd.h>
#include <dirent.h>
#endif

#define MVLOG_UNIT_NAME PCIe
#include "mvLog.h"

#define PCIE_DEVICE_ID 0x6200
#define PCIE_VENDOR_ID 0x8086

#if (defined(_WIN32) || defined(_WIN64))
int pcie_write(HANDLE fd, void * buf, size_t bufSize, int timeout)
{
    int bytesWritten;
    HANDLE dev = fd;

    BOOL ret = WriteFile(dev, buf, bufSize, &bytesWritten, 0);

    if (ret == FALSE)
        return -errno;

    return bytesWritten;
}
#else
int pcie_write(void *fd, void * buf, size_t bufSize, int timeout)
{
    int ret = write(*((int*)fd), buf, bufSize);

    if (ret < 0) {
        return -errno;
    }
    return ret;
}
#endif  // (defined(_WIN32) || defined(_WIN64))

#if (defined(_WIN32) || defined(_WIN64))
int pcie_read(HANDLE fd, void * buf, size_t bufSize, int timeout)
{
    int bytesRead;
    HANDLE dev = fd;
    BOOL ret = ReadFile(dev, buf, bufSize, &bytesRead, 0);

    if (ret == FALSE) {
        return -errno;
    }

   return bytesRead;
}
#else
int pcie_read(void *fd, void *buf, size_t bufSize, int timeout)
{
    int ret = read(*((int*)fd), buf, bufSize);

    if (ret < 0) {
        return -errno;
    }
    return ret;
}
#endif


#if (defined(_WIN32) || defined(_WIN64))
int pcie_init(const char *slot, HANDLE *fd)
{
    HANDLE hDevice = CreateFile(slot,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        0,
        NULL);

    if (hDevice == INVALID_HANDLE_VALUE) {
        mvLog(MVLOG_ERROR, "Failed to open device. Error %d", GetLastError());
        return -1;
    }

    *fd = hDevice;

    return 0;
}
#else
int pcie_init(const char *slot, void **fd)
{
    int mx_fd = open(slot, O_RDWR);

    if (mx_fd == -1) {
        return -1;
    } else {
        if (!(*fd)) {
            *fd = (int *) malloc(sizeof(int));
        }

        if (!(*fd)) {
            mvLog(MVLOG_ERROR, "Memory allocation failed");
            close(mx_fd);
            return -1;
        }
        *((int*)*fd) = mx_fd;
    }

    return 0;
}
#endif

int pcie_close(void *fd)
{
#if (defined(_WIN32) || defined(_WIN64))
    return 0;
#else
    if (!fd) {
        mvLog(MVLOG_ERROR, "Incorrect device filedescriptor");
        return -1;
    }
    int mx_fd = *((int*) fd);
    close(mx_fd);
    free(fd);
    
    return 0;
#endif
}

#if (defined(_WIN32) || defined(_WIN64))
int pci_count_devices(uint16_t vid, uint16_t pid)
{
    int i;
    int deviceCnt = 0;

    HDEVINFO hDevInfo;
    SP_DEVINFO_DATA DeviceInfoData;
    char hwid_buff[256];
    DeviceInfoData.cbSize = sizeof(DeviceInfoData);

    // List all connected PCI devices
    hDevInfo = SetupDiGetClassDevs(NULL, TEXT("PCI"), NULL, DIGCF_PRESENT | DIGCF_ALLCLASSES);
    if (hDevInfo == INVALID_HANDLE_VALUE)
        return -1;


    for (i = 0; SetupDiEnumDeviceInfo(hDevInfo, i, &DeviceInfoData); i++)
    {
        DeviceInfoData.cbSize = sizeof(DeviceInfoData);
        if (!SetupDiEnumDeviceInfo(hDevInfo, i, &DeviceInfoData))
            break;

        if (!SetupDiGetDeviceRegistryPropertyA(hDevInfo, &DeviceInfoData, SPDRP_HARDWAREID, NULL, (PBYTE)hwid_buff, sizeof(hwid_buff), NULL)) {
            continue;
        }

        uint16_t venid, devid;
        if (sscanf_s(hwid_buff, "PCI\\VEN_%hx&DEV_%hx", (int16_t *)&venid, (int16_t *)&devid) != 2) {
            continue;
        }
        if (venid == vid && devid == pid)
        {
            deviceCnt++;
        }
    }
    return deviceCnt;
}
#endif  // (defined(_WIN32) || defined(_WIN64))

xLinkPlatformErrorCode_t pcie_find_device_port(int index, char* port_name, int size) {
#if (defined(_WIN32) || defined(_WIN64))
    snprintf(port_name, size, "%s%d", "\\\\.\\mxlink", index);

    if (pci_count_devices(PCIE_VENDOR_ID, PCIE_DEVICE_ID) == 0) {
        mvLog(MVLOG_WARN, "No PCIe device(s) with Vendor ID: 0x%hX and Device ID: 0x%hX found",
                PCIE_VENDOR_ID, PCIE_DEVICE_ID);
        return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
    }

    if (index > pci_count_devices(PCIE_VENDOR_ID, PCIE_DEVICE_ID)) {
        return X_LINK_PLATFORM_DEVICE_NOT_FOUND;
    }

    return X_LINK_PLATFORM_SUCCESS;

#else
    xLinkPlatformErrorCode_t rc = X_LINK_PLATFORM_DEVICE_NOT_FOUND;
    struct dirent *entry;
    DIR *dp;
    if (port_name == NULL)
        return X_LINK_PLATFORM_ERROR;

    dp = opendir("/sys/class/mxlk/");
    if (dp == NULL)
    {
        mvLog(MVLOG_ERROR, "Unable to find a PCIe device. Make sure the driver is installed correctly.");
        return X_LINK_PLATFORM_DRIVER_NOT_LOADED;
    }

    // All entries in this (virtual) directory are generated when the driver
    // is loaded, and correspond 1:1 to entries in /dev/
    int device_cnt = 0;
    while((entry = readdir(dp))) {
        // Compare the beginning of the name to make sure it is a device name
        if (strncmp(entry->d_name, "mxlk", 4) == 0)
        {
            if (device_cnt == index)
            {
                snprintf(port_name, size, "/dev/%s", entry->d_name);
                rc = X_LINK_PLATFORM_SUCCESS;
                break;
            }
            device_cnt++;
        }
    }
    closedir(dp);

    return rc;
#endif  // (!defined(_WIN32) && !defined(_WIN64))
}
