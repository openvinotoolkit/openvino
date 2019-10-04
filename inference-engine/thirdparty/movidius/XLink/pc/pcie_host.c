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
#include "pcie_host.h"


#define PCIE_DEVICE_ID 0x6200
#define PCIE_VENDOR_ID 0x8086

#if (defined(_WIN32) || defined(_WIN64))
static HANDLE global_pcie_lock_fd = NULL;
static OVERLAPPED global_pcie_lock_overlap = { 0 };
#define GLOBAL_PCIE_LOCK() LockFileEx(global_pcie_lock_fd, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, MAXDWORD, MAXDWORD, &global_pcie_lock_overlap)
#define GLOBAL_PCIE_UNLOCK() UnlockFileEx(global_pcie_lock_fd, 0, MAXDWORD, MAXDWORD, &global_pcie_lock_overlap)
/* IOCTL commands IDs. for Windows*/
#define MXLK_DEVICE_TYPE 40001

#define MXLK_STATUS_DEV   CTL_CODE(MXLK_DEVICE_TYPE, 0xA08, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define MXLK_RESET_DEV    CTL_CODE(MXLK_DEVICE_TYPE, 0xA09, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define MXLK_BOOT_DEV     CTL_CODE(MXLK_DEVICE_TYPE, 0xA0A, METHOD_IN_DIRECT, FILE_ANY_ACCESS)

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


#if (defined(_WIN32) || defined(_WIN64))
int pcie_write(HANDLE fd, void * buf, size_t bufSize, unsigned int timeout)
{
    int bytesWritten;
    HANDLE dev = fd;

    BOOL ret = WriteFile(dev, buf, bufSize, &bytesWritten, 0);
    mvLog(MVLOG_DEBUG, "pcie_write windows return  fd %d buff %p bytesWritten %d  errno %d", dev,buf, bytesWritten, errno);
    if (ret == FALSE)
        return -errno;

    return bytesWritten;
}
#else
pcieHostError_t pcie_write(void *fd, void * buf, size_t bufSize, unsigned int timeout_ms)
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
int pcie_read(HANDLE fd, void * buf, size_t bufSize, unsigned int timeout)
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
pcieHostError_t pcie_read(void *fd, void *buf, size_t bufSize, unsigned int timeout_ms)
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
int pcie_init(const char *slot, HANDLE *fd)
{

// Commented out to re-run when the execution is aborted
/*
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
*/
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

    mvLog(MVLOG_DEBUG, "pcie_init windows new fd %d", *fd);
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
    // Commented out to re-run when the execution is aborted
    //GLOBAL_PCIE_UNLOCK();

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

pcieHostError_t pcie_find_device_port(
    int index, char* port_name, int name_length, const pciePlatformState_t requiredState) {
    ASSERT_X_LINK_PLATFORM(port_name);
    ASSERT_X_LINK_PLATFORM(index >= 0);
    ASSERT_X_LINK_PLATFORM(name_length > 0);

    pcieHostError_t rc = PCIE_HOST_DEVICE_NOT_FOUND;

    char found_device[XLINK_MAX_NAME_SIZE] = { 0 };
    pciePlatformState_t platformState;

#if (defined(_WIN32) || defined(_WIN64))
    int amoutOfMyriadPCIeDevices = pci_count_devices(PCIE_VENDOR_ID, PCIE_DEVICE_ID);
    if (amoutOfMyriadPCIeDevices == 0)
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
int pcie_reset_device(int fd)
{
    return ioctl(fd, MXLK_RESET_DEV);
}
#else
int pcie_reset_device(HANDLE fd)
{
    BOOL bResult   = FALSE;
    DWORD junk     = 0;                     // discard results
    int output_buffer;

    mvLog(MVLOG_DEBUG, "calling Windows RESET DeviceIoControl fd %d", fd);
    if (fd == 0) {
        return PCIE_HOST_ERROR;
    }

    bResult = DeviceIoControl(fd,                    // device to be queried
                              MXLK_RESET_DEV,                // operation to perform
                              NULL, 0,                       // no input buffer
                              &output_buffer, sizeof(output_buffer), // output buffer
                              &junk,                         // # bytes returned
                              (LPOVERLAPPED) NULL);          // synchronous I/O

    if (!bResult) {
        mvLog(MVLOG_ERROR, "RESET failed(status = %d).", GetLastError());
        return PCIE_HOST_ERROR;
    } else {
        return PCIE_HOST_SUCCESS;
    }
}
#endif

#if (!defined(_WIN32) && !defined(_WIN64))
int pcie_boot_device(int fd, void *buffer, size_t length)
{
    int rc = pcie_reset_device(fd);
    if (rc) {
        mvLog(MVLOG_INFO, "Device resetting failed with error: %d\n", rc);
        return rc;
    }
    struct mxlk_boot_param boot_param;

    boot_param.buffer = buffer;
    boot_param.length = length;
    return ioctl(fd, MXLK_BOOT_DEV, &boot_param);
}
#else
 int pcie_boot_device(HANDLE fd)
 {
    int rc = pcie_reset_device(fd);
    if (rc) {
        mvLog(MVLOG_INFO, "Device resetting failed with error: %d\n", rc);
        return rc;
    }

    BOOL bResult   = FALSE;
    DWORD junk     = 0;                     // discard results
    int output_buffer;
    struct mxlk_boot_param boot_param;

    mvLog(MVLOG_DEBUG, "calling Windows BOOT DeviceIoControl %d",fd);
    if (fd == 0) {
        return PCIE_HOST_ERROR;
    }
    bResult = DeviceIoControl(fd,                    // device to be queried
                              MXLK_BOOT_DEV,                 // operation to perform
                              NULL, 0,                      // no input buffer
                              &output_buffer, sizeof(output_buffer), // output buffer
                              &junk,                         // # bytes returned
                              (LPOVERLAPPED) NULL);          // synchronous I/O
    if (!bResult) {
        mvLog(MVLOG_ERROR, "BOOT failed(status = %d)", GetLastError());
        return PCIE_HOST_ERROR;
    } else {
        return PCIE_HOST_SUCCESS;
    }
}
#endif


pcieHostError_t pcie_get_device_state(const char *port_name, pciePlatformState_t *platformState) {
    ASSERT_X_LINK_PLATFORM(port_name);
    ASSERT_X_LINK_PLATFORM(platformState);
    pcieHostError_t retCode = PCIE_HOST_SUCCESS;

#if (!defined(_WIN32) && !defined(_WIN64))       // Linux implementation
    int mx_fd = open(port_name, O_RDONLY);

    if (mx_fd == -1) {
        // driver returns EACCESS in case it instance already used.
        *platformState = PCIE_PLATFORM_BOOTED;
    } else {
        enum mx_fw_status fw_status= MX_FW_STATUS_UNKNOWN_STATE;
        int ret = ioctl(mx_fd, MXLK_STATUS_DEV, &fw_status);
        if(ret){
            *platformState = PCIE_PLATFORM_ANY_STATE;
            mvLog(MVLOG_WARN, "Failed to get device status: %d. Errno %d", ret, errno);
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
    BOOL bResult   = FALSE;                 // results flag
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
        mvLog(MVLOG_ERROR, "Failed to open device: %s. Error %d", port_name, GetLastError());
        *platformState = PCIE_PLATFORM_ANY_STATE;
        return PCIE_HOST_DEVICE_NOT_FOUND;
    }
    enum mx_fw_status fw_status = MX_FW_STATUS_USER_APP;

    bResult = DeviceIoControl(hDevice,                       // device to be queried
                              MXLK_STATUS_DEV, // operation to perform
                              NULL, 0,                       // no input buffer
                              &fw_status, sizeof(fw_status), // output buffer
                              &junk,                         // # bytes returned
                              (LPOVERLAPPED) NULL);          // synchronous I/O

    if (!bResult) {
        mvLog(MVLOG_ERROR, "Failed to get device status. Error %d", GetLastError());
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
