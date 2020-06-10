// Copyright (C) 2018-2020 Intel Corporation
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
#include "win_time.h"
#include <windows.h>
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
#include "XLinkLog.h"
#include "XLinkStringUtils.h"
#include "pcie_host.h"
#include "XLinkPlatformErrorUtils.h"

#ifndef XLINK_PCIE_DATA_TIMEOUT
#define XLINK_PCIE_DATA_TIMEOUT 0
#endif

#define PCIE_DEVICE_ID 0x6200
#define PCIE_VENDOR_ID 0x8086

#if (defined(_WIN32) || defined(_WIN64))
/* IOCTL commands IDs. for Windows*/
#define MXLK_DEVICE_TYPE 40001

#define MXLK_STATUS_DEV         CTL_CODE(MXLK_DEVICE_TYPE, 0xA08, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define MXLK_RESET_DEV          CTL_CODE(MXLK_DEVICE_TYPE, 0xA09, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define MXLK_BOOT_DEV           CTL_CODE(MXLK_DEVICE_TYPE, 0xA0A, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define MXLK_BOOTLOADER_DEV     CTL_CODE(MXLK_DEVICE_TYPE, 0xA0B, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

#endif

#if (!defined(_WIN32) && !defined(_WIN64))
/**         MXLK data           */
/* IOCTL commands IDs. */
#define IOC_MAGIC 'Z'
#define MXLK_RESET_DEV    _IO(IOC_MAGIC,  0x80)
#define MXLK_BOOT_DEV     _IOW(IOC_MAGIC, 0x81, struct mxlk_boot_param)
#define MXLK_STATUS_DEV   _IOR(IOC_MAGIC, 0x82,  enum mx_fw_status)
#endif

struct mxlk_boot_param {
    /* Buffer containing the MX application image (MVCMD format) */
    const char *buffer;
    /* Size of the image in bytes. */
    size_t length;
};

/* State of Myriad X device. */
enum mx_fw_status {
    /* MX waiting for FW to be loaded from host */
    MX_FW_STATE_BOOTLOADER,
    /* MX running FW loaded from host. */
    MX_FW_STATUS_USER_APP,
    /* MX context is not restored or device is lost*/
    MX_FW_STATUS_UNKNOWN_STATE,
};
/**         MXLK data end       */

#if !(defined(_WIN32) || defined(_WIN64))
static inline void timeout_to_timeval(unsigned int timeout_ms,
                                      struct timeval *timeval)
{
    timeval->tv_sec = timeout_ms / 1000;
    timeval->tv_usec = (timeout_ms - (timeval->tv_sec * 1000)) * 1000;
}
#endif

static inline void sleepForSeconds(const unsigned int seconds) {
#if (!defined(_WIN32) && !defined(_WIN64))
    sleep(seconds);
#else
    Sleep(seconds * 1000); // Sleep using miliseconds as input
#endif
}

#if !defined(_WIN32)
static pcieHostError_t getDeviceFwStatusIOCTL(const int fd, enum mx_fw_status *fwStatus) {
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(fwStatus, PCIE_INVALID_PARAMETERS);

    int ret = ioctl(fd, MXLK_STATUS_DEV, fwStatus);

    if (ret) {
        mvLog(MVLOG_INFO, "Get device status ioctl failed with error: %d", ret);
        *fwStatus = MX_FW_STATUS_UNKNOWN_STATE;
        return PCIE_HOST_ERROR;
    }
    return PCIE_HOST_SUCCESS;
}
#else
static pcieHostError_t getDeviceFwStatusIOCTL(const HANDLE deviceHandle, enum mx_fw_status *fwStatus) {
    ASSERT_XLINK_PLATFORM_R(deviceHandle, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(fwStatus, PCIE_INVALID_PARAMETERS);

    DWORD junk = 0;

    BOOL successRc = DeviceIoControl(deviceHandle,          // device to be queried
                                   MXLK_STATUS_DEV,             // operation to perform
                                   NULL, 0,                     // no input buffer
                                   fwStatus, sizeof(*fwStatus), // output buffer
                                   &junk,                       // # bytes returned
                                   (LPOVERLAPPED) NULL);        // synchronous I/O

     if (!successRc) {
         mvLog(MVLOG_ERROR, "Get PCIe device status ioctl failed with error: 0x%x", GetLastError());
         *fwStatus = MX_FW_STATUS_UNKNOWN_STATE;
         return PCIE_HOST_ERROR;
     }
     return PCIE_HOST_SUCCESS;
}
#endif

#if (defined(_WIN32) || defined(_WIN64))
int pcie_write(HANDLE fd, void * buf, size_t bufSize)
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(buf, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(bufSize >= 0, PCIE_INVALID_PARAMETERS);

    HANDLE dev =  fd;
    OVERLAPPED Overlapped;
    HANDLE Event = NULL;
    bool OutputCode = FALSE;
    ULONG  bytesWrite;

    ZeroMemory(&Overlapped, sizeof(Overlapped));
    Event = CreateEvent(NULL, TRUE, FALSE, NULL);

    if (Event == NULL) {
        mvLog(MVLOG_INFO, "Error creating I/O event for pcie_write - 0x%x\n", GetLastError());
        return PCIE_HOST_ERROR;
    }

    Overlapped.hEvent = Event;
    ResetEvent(Overlapped.hEvent);
    OutputCode = WriteFile(dev, buf, bufSize, NULL, &Overlapped);

    if (OutputCode == FALSE) {
        if (GetLastError() == ERROR_IO_PENDING) {
            if (GetOverlappedResult(dev, &Overlapped, &bytesWrite, TRUE)) {
                CloseHandle(Overlapped.hEvent);
                return bytesWrite;
            } else {
                mvLog(MVLOG_DEBUG, "WriteFile GetOverlappedResult failed");
            }
        } else {
            mvLog(MVLOG_DEBUG, "WriteFile failed with error code = 0x%x \n", GetLastError());
        }
    } else {
        mvLog(MVLOG_DEBUG, "fOverlapped - operation complete immediately");
    }

    CloseHandle(Overlapped.hEvent);

    return PCIE_HOST_ERROR;
}
#else
int pcie_write(void *fd, void * buf, size_t bufSize)
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(buf, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(bufSize >= 0, PCIE_INVALID_PARAMETERS);

    fd_set wrfds;
    int ret;

    FD_ZERO(&wrfds);
    FD_SET(*((int*)fd), &wrfds);

    ret = select(*((int*)fd) + 1, NULL, &wrfds, NULL, NULL);
    if (ret < 0)
    {
        return PCIE_HOST_ERROR;
    }
    if (!FD_ISSET(*((int*)fd), &wrfds))
    {
        return PCIE_HOST_TIMEOUT;
    }

    ret = write(*((int*)fd), buf, bufSize);

    if (ret < 0)
    {
        return PCIE_HOST_ERROR;
    }
    return ret;
}
#endif  // (defined(_WIN32) || defined(_WIN64))

#if (defined(_WIN32) || defined(_WIN64))
int pcie_read(HANDLE fd, void * buf, size_t bufSize)
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(buf, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(bufSize >= 0, PCIE_INVALID_PARAMETERS);

    HANDLE dev =  fd;
    OVERLAPPED Overlapped;
    HANDLE Event = NULL;
    bool OutputCode = FALSE;
    ULONG  bytesRead;

    ZeroMemory(&Overlapped, sizeof(Overlapped));
    Event = CreateEvent(NULL, TRUE, FALSE, NULL);

    if (Event == NULL) {
        mvLog(MVLOG_ERROR, "Error creating I/O event for pcie_read - 0x%x\n", GetLastError());
        return PCIE_HOST_ERROR;
    }

    Overlapped.hEvent = Event;
    ResetEvent(Overlapped.hEvent);
    OutputCode = ReadFile(dev, buf, bufSize, NULL, &Overlapped);

    if (OutputCode == FALSE) {
       if (GetLastError() == ERROR_IO_PENDING) {
           if (GetOverlappedResult(dev, &Overlapped, &bytesRead, TRUE))
           {
               CloseHandle(Overlapped.hEvent);
               return bytesRead;
            } else{
               mvLog(MVLOG_DEBUG, "ReadFile GetOverlappedResult failed" );
            }
        } else {
            mvLog(MVLOG_DEBUG, "ReadFile failed with error code = 0x%x \n", GetLastError());
        }
    } else {
         mvLog(MVLOG_DEBUG, "fOverlapped - operation complete immediately");
    }
    CloseHandle(Overlapped.hEvent);

    return PCIE_HOST_ERROR;
}
#else
int pcie_read(void *fd, void *buf, size_t bufSize)
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(buf, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(bufSize >= 0, PCIE_INVALID_PARAMETERS);

    fd_set rdfds;
    int ret;

    FD_ZERO(&rdfds);
    FD_SET(*((int*)fd), &rdfds);

    ret = select(*((int*)fd) + 1, &rdfds, NULL, NULL, NULL);
    if (ret < 0) {
        return PCIE_HOST_ERROR;
    }
    if (!FD_ISSET(*((int*)fd), &rdfds)) {
        return PCIE_HOST_TIMEOUT;
    }

    ret = read(*((int*)fd), buf, bufSize);

    if (ret < 0) {
        return PCIE_HOST_ERROR;
    }
    return ret;
}
#endif

#if (defined(_WIN32) || defined(_WIN64))
pcieHostError_t pcie_init(const char *slot, HANDLE *fd)
{
    ASSERT_XLINK_PLATFORM_R(slot, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);

    HANDLE hDevice = CreateFile(slot,
        GENERIC_READ | GENERIC_WRITE,
        FILE_SHARE_READ | FILE_SHARE_WRITE,
        NULL,
        OPEN_EXISTING,
        FILE_FLAG_OVERLAPPED,
        NULL);

    if (hDevice == INVALID_HANDLE_VALUE) {
        mvLog(MVLOG_ERROR, "Failed to open device. Error %d", GetLastError());
        return PCIE_HOST_ERROR;
    }

    *fd = hDevice;

    mvLog(MVLOG_DEBUG, "PCIe init completed, new fd %d", *fd);
    return PCIE_HOST_SUCCESS;
}
#else
pcieHostError_t pcie_init(const char *slot, void **fd)
{
    ASSERT_XLINK_PLATFORM_R(slot, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);

    int mx_fd = open(slot, O_RDWR);

    if (mx_fd == -1) {
        return PCIE_HOST_DEVICE_NOT_FOUND;
    } else {
        if (!(*fd)) {
            *fd = (int *) malloc(sizeof(int));
        }

        if (!(*fd)) {
            mvLog(MVLOG_ERROR, "Memory allocation failed");
            close(mx_fd);
            return PCIE_HOST_ERROR;
        }
        *((int*)*fd) = mx_fd;
    }

    return PCIE_HOST_SUCCESS;
}
#endif

pcieHostError_t pcie_close(void *fd)
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
#if (defined(_WIN32) || defined(_WIN64))
    HANDLE hDevice = (HANDLE)fd;
    if (hDevice == INVALID_HANDLE_VALUE) {
        mvLog(MVLOG_ERROR, "Invalid device handle");
        return PCIE_HOST_ERROR;
    }
    CloseHandle(hDevice);

    return PCIE_HOST_SUCCESS;
#else

    int mx_fd = *((int*) fd);
    close(mx_fd);
    free(fd);

    return PCIE_HOST_SUCCESS;
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
        return PCIE_HOST_ERROR;

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

pcieHostError_t pcie_find_device_port(
    int index, char* port_name, int name_length, const pciePlatformState_t requiredState) {
    ASSERT_XLINK_PLATFORM_R(port_name, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(index >= 0, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(name_length > 0, PCIE_INVALID_PARAMETERS);

    pcieHostError_t rc = PCIE_HOST_DEVICE_NOT_FOUND;

    char found_device[XLINK_MAX_NAME_SIZE] = { 0 };
    pciePlatformState_t platformState;

#if (defined(_WIN32) || defined(_WIN64))
    int amoutOfMyriadPCIeDevices = pci_count_devices(PCIE_VENDOR_ID, PCIE_DEVICE_ID);
    if (amoutOfMyriadPCIeDevices <= 0)
        return PCIE_HOST_DEVICE_NOT_FOUND;

    int amountOfSuitableDevices = 0;
    int deviceCount = 0;

    while (deviceCount < amoutOfMyriadPCIeDevices) {
        snprintf(found_device, XLINK_MAX_NAME_SIZE, "%s%d", "\\\\.\\mxlink", deviceCount);

        // Get state of device
        if (pcie_get_device_state(found_device, &platformState) != 0) {
            return PCIE_HOST_ERROR;   // Get device state step failed
        }

        // Found device suits requested state
        if (platformState == requiredState || requiredState == PCIE_PLATFORM_ANY_STATE) {
            // If port_name is specified, we search for specific device
            if (strnlen(port_name, name_length) > 1 &&
                strncmp(port_name, found_device, name_length) == 0) {
                rc = PCIE_HOST_SUCCESS;
                break;
                // Trying to find device which suits requirements and index
            }
            else if (amountOfSuitableDevices == index) {
                mv_strncpy(port_name, name_length,
                    found_device, XLINK_MAX_NAME_SIZE - 1);
                rc = PCIE_HOST_SUCCESS;
                break;
            }
            ++amountOfSuitableDevices;
        }
        ++deviceCount;
    }

    return rc;
#else
    struct dirent *entry;
    DIR *dp;

    dp = opendir("/sys/class/mxlk/");
    if (dp == NULL) {
        return PCIE_HOST_DRIVER_NOT_LOADED;
    }

    // All entries in this (virtual) directory are generated when the driver
    // is loaded, and correspond 1:1 to entries in /dev/
    int device_cnt = 0;
    while((entry = readdir(dp))) {
        // Compare the beginning of the name to make sure it is a device name
        if (strncmp(entry->d_name, "mxlk", 4) == 0)
        {
            // Save name
            snprintf(found_device, name_length, "/dev/%s", entry->d_name);
            // Get state of device
            if (pcie_get_device_state(found_device, &platformState) != 0) {
                closedir(dp);
                return PCIE_HOST_ERROR;   // Get device state step failed
            }

            // Found device suits requested state
            if (platformState == requiredState || requiredState == PCIE_PLATFORM_ANY_STATE) {
                // If port_name is specified, we search for specific device
                if (strnlen(port_name, name_length) > 1 &&
                    strncmp(port_name, found_device, name_length) == 0) {
                    rc = PCIE_HOST_SUCCESS;
                    break;
                    // Trying to find device which suits requirements and index
                } else if (device_cnt == index){
                    mv_strncpy(port_name, name_length,
                               found_device, XLINK_MAX_NAME_SIZE - 1);
                    rc = PCIE_HOST_SUCCESS;
                    break;
                }
                ++device_cnt;
            }
        }
    }
    closedir(dp);

    return rc;
#endif  // (!defined(_WIN32) && !defined(_WIN64))
}

#if (!defined(_WIN32) && !defined(_WIN64))
pcieHostError_t pcie_reset_device(int fd)
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
    int ret = ioctl(fd, MXLK_RESET_DEV);

    if (ret) {
        mvLog(MVLOG_INFO, "Reset ioctl failed with error: %d", ret);
        return PCIE_HOST_ERROR;
    }
    return PCIE_HOST_SUCCESS;
}
#else
pcieHostError_t pcie_reset_device(HANDLE fd)
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);

    BOOL bResult   = FALSE;
    DWORD junk     = 0;                     // discard results
    int output_buffer;

    mvLog(MVLOG_DEBUG, "calling Windows RESET DeviceIoControl fd %d", fd);

    bResult = DeviceIoControl(fd,                    // device to be queried
                              MXLK_RESET_DEV,                // operation to perform
                              NULL, 0,                       // no input buffer
                              &output_buffer, sizeof(output_buffer), // output buffer
                              &junk,                         // # bytes returned
                              (LPOVERLAPPED) NULL);          // synchronous I/O

    if (!bResult) {
        mvLog(MVLOG_ERROR, "RESET failed(status = %d).", GetLastError());
        return PCIE_HOST_ERROR;
    }
    return PCIE_HOST_SUCCESS;
}
#endif

#if !defined(_WIN32)
pcieHostError_t pcie_boot_device(int fd, const char *buffer, size_t length)
#else
pcieHostError_t pcie_boot_device(HANDLE fd, const char  *buffer, size_t length)
#endif
{
    ASSERT_XLINK_PLATFORM_R(fd, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(buffer, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(length >= 0, PCIE_INVALID_PARAMETERS);

    // Get device context
    enum mx_fw_status fw_status = MX_FW_STATUS_UNKNOWN_STATE;
    int deviceStatusRC = getDeviceFwStatusIOCTL(fd, &fw_status);

    if (deviceStatusRC) {
        return PCIE_HOST_ERROR;
    }

    // After reset PCIe device will lose context, so we need to restore in by calling reset
    if (fw_status == MX_FW_STATUS_UNKNOWN_STATE) {
        // Device may be not ready if reset context just after reset
        sleepForSeconds(3);

        int resetDeviceRC = pcie_reset_device(fd);
        if (resetDeviceRC) {
            mvLog(MVLOG_ERROR, "Device resetting failed with error: %d\n", resetDeviceRC);
            return resetDeviceRC;
        }
    }
#if !defined(_WIN32)
    struct mxlk_boot_param boot_param;

    boot_param.buffer = buffer;
    boot_param.length = length;

    int ret = ioctl(fd, MXLK_BOOT_DEV, &boot_param);

    if (ret) {
        mvLog(MVLOG_INFO, "Boot ioctl failed with error: %d", ret);
        return PCIE_HOST_ERROR;
    }
    return PCIE_HOST_SUCCESS;
#else
    BOOL bResult   = FALSE;
    DWORD junk     = 0;                     // discard results
    int output_buffer;

    mvLog(MVLOG_DEBUG, "calling Windows BOOT DeviceIoControl %d, buffer = %p, size = %d",fd, buffer, length);

    bResult = DeviceIoControl(fd,                    // device to be queried
                              MXLK_BOOT_DEV,                 // operation to perform
                              (void*)buffer, length,
                              &output_buffer, sizeof(output_buffer), // output buffer
                              &junk,                         // # bytes returned
                              (LPOVERLAPPED) NULL);          // synchronous I/O
    if (!bResult) {
        mvLog(MVLOG_ERROR, "BOOT failed(status = %d)", GetLastError());
        return PCIE_HOST_ERROR;
    }
    return PCIE_HOST_SUCCESS;
#endif
}

pcieHostError_t pcie_get_device_state(const char *port_name, pciePlatformState_t *platformState) {
    ASSERT_XLINK_PLATFORM_R(port_name, PCIE_INVALID_PARAMETERS);
    ASSERT_XLINK_PLATFORM_R(platformState, PCIE_INVALID_PARAMETERS);
    pcieHostError_t retCode = PCIE_HOST_SUCCESS;

#if (!defined(_WIN32) && !defined(_WIN64))       // Linux implementation
    int mx_fd = open(port_name, O_RDONLY);

    if (mx_fd == -1) {
        // driver returns EACCESS in case it instance already used.
        *platformState = PCIE_PLATFORM_BOOTED;
    } else {
        enum mx_fw_status fw_status = MX_FW_STATUS_UNKNOWN_STATE;
        int deviceStatusRC = getDeviceFwStatusIOCTL(mx_fd, &fw_status);

        if(deviceStatusRC){
            *platformState = PCIE_PLATFORM_ANY_STATE;
            retCode = PCIE_HOST_DEVICE_NOT_FOUND;
        } else if(fw_status == MX_FW_STATUS_USER_APP) {
            *platformState = PCIE_PLATFORM_BOOTED;
        } else {
            *platformState = PCIE_PLATFORM_UNBOOTED;
        }

        close(mx_fd);
    }
#else                                           // Windows implementation
    HANDLE hDevice = INVALID_HANDLE_VALUE;  // handle to the drive to be examined
    DWORD junk     = 0;                     // discard results

    hDevice = CreateFile(port_name,         // drive to open
                         0,                 // no access to the drive
                         FILE_SHARE_READ |  // share mode
                         FILE_SHARE_WRITE,
                         NULL,              // default security attributes
                         OPEN_EXISTING,     // disposition
                         0,                 // file attributes
                         NULL);             // do not copy file attributes

    if (hDevice == INVALID_HANDLE_VALUE){   // cannot open the drive
        mvLog(MVLOG_DEBUG, "No PCIE device found: %s. Error %d", port_name, GetLastError());
        *platformState = PCIE_PLATFORM_ANY_STATE;
        return PCIE_HOST_DEVICE_NOT_FOUND;
    }
    enum mx_fw_status fw_status = MX_FW_STATUS_USER_APP;
    int deviceStatusRC = getDeviceFwStatusIOCTL(hDevice, &fw_status);

    if (deviceStatusRC) {
        *platformState = PCIE_PLATFORM_ANY_STATE;
        retCode = PCIE_HOST_DEVICE_NOT_FOUND;
        mvLog(MVLOG_DEBUG, "PCIE_PLATFORM_ANY_STATE");
    } else if (fw_status == MX_FW_STATUS_USER_APP) {
        *platformState = PCIE_PLATFORM_BOOTED;
        mvLog(MVLOG_DEBUG, "PCIE_PLATFORM_BOOTED");
    } else {
        *platformState = PCIE_PLATFORM_UNBOOTED;
        mvLog(MVLOG_DEBUG, "PCIE_PLATFORM_UNBOOTED");
    }

    CloseHandle(hDevice);
#endif
    return retCode;
}
