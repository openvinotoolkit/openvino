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
#include <sys/ioctl.h>
#include <sys/time.h>
#include <sys/select.h>
#include <unistd.h>
#include <dirent.h>
#endif

#define MVLOG_UNIT_NAME PCIe
#include "mvLog.h"
#include "mvStringUtils.h"

#define PCIE_DEVICE_ID 0x6200
#define PCIE_VENDOR_ID 0x8086

#if (defined(_WIN32) || defined(_WIN64))
static HANDLE global_pcie_lock_fd = NULL;
static OVERLAPPED global_pcie_lock_overlap = { 0 };
#define GLOBAL_PCIE_LOCK() LockFileEx(global_pcie_lock_fd, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD, &global_pcie_lock_overlap)
#define GLOBAL_PCIE_UNLOCK() UnlockFileEx(global_pcie_lock_fd, 0, MAXDWORD, MAXDWORD, &global_pcie_lock_overlap)
#endif

#if (!defined(_WIN32) || !defined(_WIN64))
/**         MXLK data           */
/* IOCTL commands IDs. */
#define IOC_MAGIC 'Z'
#define MXLK_RESET_DEV _IO(IOC_MAGIC, 0x80)
#define MXLK_BOOT_DEV  _IOW(IOC_MAGIC, 0x81, struct mxlk_boot_param)

struct mxlk_boot_param {
    /* Buffer containing the MX application image (MVCMD format) */
    const char *buffer;
    /* Size of the image in bytes. */
    size_t length;
};
/**         MXLK data end       */
#endif

#if !(defined(_WIN32) || defined(_WIN64))
static inline void timeout_to_timeval(unsigned int timeout_ms,
                                      struct timeval *timeval)
{
    timeval->tv_sec = timeout_ms / 1000;
    timeval->tv_usec = (timeout_ms - (timeval->tv_sec * 1000)) * 1000;
}
#endif


#if (defined(_WIN32) || defined(_WIN64))
int pcie_write(HANDLE fd, void * buf, size_t bufSize, unsigned int timeout)
{
    int bytesWritten;
    HANDLE dev = fd;

    BOOL ret = WriteFile(dev, buf, bufSize, &bytesWritten, 0);

    if (ret == FALSE)
        return -errno;

    return bytesWritten;
}
#else
int pcie_write(void *fd, void * buf, size_t bufSize, unsigned int timeout_ms)
{
    fd_set wrfds;
    struct timeval timeval;
    struct timeval *select_timeout;
    int ret;

    FD_ZERO(&wrfds);
    FD_SET(*((int*)fd), &wrfds);

    if (timeout_ms)
    {
        timeout_to_timeval(timeout_ms, &timeval);
        select_timeout = &timeval;
    }
    else
    {
        select_timeout = NULL;
    }

    ret = select(*((int*)fd) + 1, NULL, &wrfds, NULL, select_timeout);
    if (ret < 0)
    {
        return X_LINK_PLATFORM_ERROR;
    }
    if (!FD_ISSET(*((int*)fd), &wrfds))
    {
        return X_LINK_PLATFORM_TIMEOUT;
    }

    ret = write(*((int*)fd), buf, bufSize);
    if (ret < 0)
    {
        return X_LINK_PLATFORM_ERROR;
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
int pcie_read(void *fd, void *buf, size_t bufSize, int timeout_ms)
{
    fd_set rdfds;
    struct timeval timeval;
    struct timeval *select_timeout;
    int ret;

    FD_ZERO(&rdfds);
    FD_SET(*((int*)fd), &rdfds);

    if (timeout_ms) {
        timeout_to_timeval(timeout_ms, &timeval);
        select_timeout = &timeval;
    } else {
        select_timeout = NULL;
    }

    ret = select(*((int*)fd) + 1, &rdfds, NULL, NULL, select_timeout);
    if (ret < 0) {
        return X_LINK_PLATFORM_ERROR;
    }
    if (!FD_ISSET(*((int*)fd), &rdfds)) {
        return X_LINK_PLATFORM_TIMEOUT;
    }

    ret = read(*((int*)fd), buf, bufSize);
    if (ret < 0) {
        return X_LINK_PLATFORM_ERROR;
    }

    return ret;
}
#endif

#if (defined(_WIN32) || defined(_WIN64))
int pcie_init(const char *slot, HANDLE *fd)
{
    const char* tempPath = getenv("TEMP");
    const char pcieMutexName[] = "\\pcie.mutex";
    if (tempPath) {
        size_t pathSize = strlen(tempPath) + sizeof(pcieMutexName);
        char *path = malloc(pathSize);
        if (!path) {
            return -1;
        }
        mv_strcpy(path, pathSize, tempPath);
        strcat_s(path, pathSize, pcieMutexName);
        global_pcie_lock_fd = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
        free(path);
    }

    if (!global_pcie_lock_fd) {
        mvLog(MVLOG_ERROR, "Global pcie mutex initialization failed.");
        exit(1);
    }

    if (!GLOBAL_PCIE_LOCK()) {
        mvLog(MVLOG_ERROR, "Only one device supported.");
        return -1;
    }

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
    if (!fd)
     return -1;

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
    GLOBAL_PCIE_UNLOCK();

    HANDLE hDevice = (HANDLE)fd;
    if (hDevice == INVALID_HANDLE_VALUE) {
        mvLog(MVLOG_ERROR, "Invalid device handle");
        return -1;
    }
    CloseHandle(hDevice);

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
        mvLog(MVLOG_DEBUG, "No PCIe device(s) with Vendor ID: 0x%hX and Device ID: 0x%hX found",
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
    if (dp == NULL) {
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

int pcie_reset_device(int fd)
{
#if (!defined(_WIN32) || !defined(_WIN64))
    return ioctl(fd, MXLK_RESET_DEV);
#else
    return -1;
#endif
}

int pcie_boot_device(int fd, void *buffer, size_t length)
{
#if (!defined(_WIN32) || !defined(_WIN64))
    struct mxlk_boot_param boot_param;

    boot_param.buffer = buffer;
    boot_param.length = length;
    return ioctl(fd, MXLK_BOOT_DEV, &boot_param);
#else
    return -1;
#endif
}
