// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <time.h>
#include <sys/types.h>
#if (defined(_WIN32) || defined(_WIN64))
#include "win_time.h"
#include <windows.h>    // for Sleep()
#include <io.h>
#else
#include <dlfcn.h>      // For dladdr
#include <unistd.h>
#include <dirent.h>
#include <sys/file.h>
#include <errno.h>
#endif

#include "mvnc.h"
#include "mvnc_data.h"

#include "XLink.h"
#include "ncCommPrivate.h"
#include "ncPrivateTypes.h"
#include "XLinkPlatform.h"

#define MVLOG_UNIT_NAME ncAPI
#include "XLinkLog.h"
#include "mvnc_tool.h"
#include "XLinkMacros.h"
#include "XLinkStringUtils.h"
#include "watchdog.h"
#include "xlink_device.h"

#define THERMAL_BUFFER_SIZE 100
#define THERMAL_THROTTLING_BUFFER_SIZE (THERMAL_BUFFER_SIZE + sizeof(int))
#define DEBUG_BUFFER_SIZE     120

#define MAX_TENSORS_TO_LOAD (2)
#define BLOB_STREAM_SIZE 4096
#define TENSOR_STREAM_SIZE 320*1024   * MAX_TENSORS_TO_LOAD
#define OUTPUT_STREAM_SIZE 8 //read only from PC

#define CONFIG_STREAM_SIZE 2000

#define MAX_PATH_LENGTH         255
#define MAX_RELATED_PATH_LENGTH   100

//      Firmware
#define FIRMWARE_DIR_LENGTH         (190)
#define FIRMWARE_NAME_LENGTH        (60)
#define FIRMWARE_PROTOCOL_LENGTH    (15)
#define FIRWMARE_DEVICE_LENGTH      (30)
#define FIRMWARE_FORMAT_LENGTH      (15)

//      Timeouts
#define DEVICE_APPEAR_TIMEOUT_ON_OPEN       (5)
#define DEVICE_APPEAR_TIMEOUT_ON_CLOSE      (10)

#define SLEEP_MS        250
#define MAX_ITERATIONS  20

#define FP16_DATA_SIZE 2

static int initialized = 0;
static int reset_all = 1;

static int g_deviceConnectTimeoutSec = 15;

pthread_mutex_t deviceOpenMutex = PTHREAD_MUTEX_INITIALIZER;

#if (defined(_WIN32) || defined(_WIN64))
static HANDLE global_lock_fd = NULL;
static OVERLAPPED global_lock_overlap = { 0 };
#define GLOBAL_LOCK() LockFileEx(global_lock_fd, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &global_lock_overlap)
#define GLOBAL_UNLOCK() UnlockFileEx(global_lock_fd, 0, MAXDWORD, MAXDWORD, &global_lock_overlap)
#else
static int global_lock_fd = -1;
#define GLOBAL_LOCK()                                                                               \
    do {                                                                                            \
        CHECK_MUTEX_SUCCESS_RC(flock(global_lock_fd, LOCK_EX), NC_ERROR);                           \
        if (pthread_mutex_lock(&deviceOpenMutex) != 0) {                                            \
            CHECK_MUTEX_SUCCESS(flock(global_lock_fd, LOCK_UN));                                    \
            return NC_ERROR;                                                                        \
        }                                                                                           \
    } while (0)

#define GLOBAL_UNLOCK()                                                                             \
    do {                                                                                            \
        if (flock(global_lock_fd, LOCK_UN) != 0) {                                                  \
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));                            \
            return NC_ERROR;                                                                        \
        }                                                                                           \
        CHECK_MUTEX_SUCCESS_RC(pthread_mutex_unlock(&deviceOpenMutex), NC_ERROR);                   \
    } while (0)
#endif

#define STRINGIFY(_text) #_text
#define CASE(entry) case entry: return STRINGIFY(entry);


// To suppress warning in the macro below
#if defined __GNUC__ || defined __clang__
#pragma GCC diagnostic ignored "-Wformat-extra-args"
#endif

/**
 * @brief The macro checks a stream id passed to it
 * @param id Stream id to check
 * @param callReleasingResources if it is needed to release resource in case of error, put your code of releasing
 *        to { you code here }. If no need to release resource pass {} to the parameter
 * @param errorMsg Message to be written in case of error. It is a format string
 */
#ifndef CHECK_STREAM_ID
#define CHECK_STREAM_ID(id, callReleasingResources, errorMsg) {                                                     \
    char errorMsgWithReason[255];                                                                                   \
    if (id == INVALID_STREAM_ID_OUT_OF_MEMORY) {                                                                    \
        snprintf(errorMsgWithReason, 255, "%s %s", errorMsg, "due to not enough memory on device");                 \
        mvLog(MVLOG_ERROR, errorMsgWithReason);                                                                     \
        callReleasingResources;                                                                                     \
        return NC_OUT_OF_MEMORY;                                                                                    \
    } else if (id == INVALID_STREAM_ID) {                                                                           \
         snprintf(errorMsgWithReason, 255, "%s %s", errorMsg, "due to unknown error");                              \
         mvLog(MVLOG_ERROR, errorMsgWithReason);                                                                    \
         callReleasingResources;                                                                                    \
         return NC_ERROR;                                                                                           \
    }                                                                                                               \
    mvLog(MVLOG_DEBUG, "Stream opened");                                                                            \
}
#endif // CHECK_STREAM_ID

static XLinkGlobalHandler_t ghandler;

devicePrivate_t *devices;

/////////////////////////// Converters /////////////////////////////

char* ncProtocolToStr(const ncDeviceProtocol_t deviceProtocol) {
    switch (deviceProtocol) {
        case NC_USB:            return "USB";
        case NC_PCIE:           return "PCIE";
        case NC_ANY_PROTOCOL:   return "ANY_PROTOCOL";
        default:                return "Unknown protocol name";
    }
}

char* ncPlatformToStr(const ncDevicePlatform_t platform) {
    switch(platform) {
        case NC_MYRIAD_2:              return "NC_MYRIAD_2";
        case NC_MYRIAD_X:              return "NC_MYRIAD_X";
        default:                    return "NC_ANY_PLATFORM";
    }
}

int mvnc_memcpy(void* dest, size_t destsz, void const* src, size_t count) {
    size_t i;
    if (!src || count > destsz ||
        count > (dest > src ? ((uintptr_t)dest - (uintptr_t)src)
                            : ((uintptr_t)src - (uintptr_t)dest))) {
        // zero out dest if error detected
        memset(dest, 0, destsz);
        return -1;
    }

    for (i = 0; i < count; ++i) ((uint8_t*)dest)[i] = ((const uint8_t*)src)[i];
    return 0;
}

static
char* mvnc_strdup(const char* s) {
#ifdef _MSC_VER
    return _strdup(s);
#else
    return strdup(s);
#endif
}

static ncStatus_t parseXLinkError(XLinkError_t rc) {
    switch (rc) {
    case X_LINK_SUCCESS:
        return NC_OK;
    case X_LINK_DEVICE_NOT_FOUND:
        return NC_DEVICE_NOT_FOUND;
    case X_LINK_TIMEOUT:
        return NC_TIMEOUT;
    default:
        return NC_ERROR;
    }
}

static char* ncStatusToStr(const ncStatus_t status) {
    switch(status) {
        CASE(NC_OK)
        CASE(NC_BUSY)
        CASE(NC_OUT_OF_MEMORY)
        CASE(NC_DEVICE_NOT_FOUND)
        CASE(NC_INVALID_PARAMETERS)
        CASE(NC_TIMEOUT)
        CASE(NC_MVCMD_NOT_FOUND)
        CASE(NC_NOT_ALLOCATED)
        CASE(NC_UNAUTHORIZED)
        CASE(NC_UNSUPPORTED_GRAPH_FILE)
        CASE(NC_UNSUPPORTED_CONFIGURATION_FILE)
        CASE(NC_UNSUPPORTED_FEATURE)
        CASE(NC_MYRIAD_ERROR)
        CASE(NC_INVALID_DATA_LENGTH)
        CASE(NC_INVALID_HANDLE)
        default: return STRINGIFY(NC_ERROR);
    }
}

static char* ncMvNCIErrorCodeToStr(const ncMvNCIErrorCode_t code) {
    switch(code) {
        CASE(MVNCI_SUCCESS)
        CASE(MVNCI_NULL_PARAM)
        CASE(MVNCI_MASK_NOTCONTINUOUS)
        CASE(MVNCI_UNSUPPORTED_NETWORK_ELEMENT)
        CASE(MVNCI_INVALID_HANDLE)
        CASE(MVNCI_OUT_OF_RESOURCES)
        CASE(MVNCI_NOT_IMPLEMENTED)
        CASE(MVNCI_SHAVES_SLICES_MISMATCH)
        CASE(MVNCI_TIMEOUT)
        CASE(MVNCI_OUT_OF_MEMORY)
        default: return STRINGIFY(MVNCI_INTERNAL_ERROR);
    }
}

static double timeInSeconds()
{
    static double s;
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    if (!s)
        s = ts.tv_sec + ts.tv_nsec * 1e-9;
    return ts.tv_sec + ts.tv_nsec * 1e-9 - s;
}

static void sleepForSeconds(const unsigned int seconds) {
#if (!defined(_WIN32) && !defined(_WIN64))
    sleep(seconds);
#else
    Sleep(seconds * 1000); // Sleep using miliseconds as input
#endif
}

static ncOptionAccess_t getOptionAccess(int option, int base)
{
    return (int) ((option - base) / OPTION_CLASS_SIZE);
}

#if (defined(_WIN32) || defined(_WIN64) )
#define MAX_2(a,b)		((a) > (b) ? (a) : (b))
#define MAX_3(a,b,c)	((a) > (b) ? MAX_2((a), (c)) : MAX_2((b), (c)))
#ifdef MAX
#undef MAX
#define MAX MAX_2
#endif
#else
#define MAX_3(a,b,c)                            \
    ({ __typeof__ (a) _a = (a);                 \
        __typeof__ (b) _b = (b);                \
        __typeof__ (c) _c = (c);                \
        (_a > _b && _a > _c) ? _a : ((_b > _c && _b > _a) ? _b : _c); })
#endif

static void resetAll()
{
#if defined(NO_BOOT)
    mvLog(MVLOG_INFO, "Devices will not be restarted for this configuration (NO_BOOT)");
#else
    // Reset only USB devices
    deviceDesc_t in_deviceDesc = {
        .protocol = X_LINK_ANY_PROTOCOL,
        .platform = X_LINK_ANY_PLATFORM
    };

    unsigned int stalled_count = 0;
    deviceDesc_t stalledDevices[NC_MAX_DEVICES] = { { 0 } };

    unsigned int stalled_count_after_reboot = 0;


    double waittm = timeInSeconds() + DEVICE_APPEAR_TIMEOUT_ON_OPEN;
    do {
        // Find stalled devices
        stalled_count = 0;
        XLinkFindAllSuitableDevices(
                X_LINK_BOOTED, in_deviceDesc, stalledDevices, NC_MAX_DEVICES, &stalled_count);

        if (stalled_count) {
            mvLog(MVLOG_INFO, "%d stalled devices found, Resetting...", stalled_count);
        } else {
            mvLog(MVLOG_DEBUG, "Stalled devices not found");
            return;
        }

        // Try to reboot them
        int i;
        for (i = 0; i < (int)stalled_count; ++i) {
            mvLog(MVLOG_DEBUG, "Found stalled device %s", stalledDevices[i].name);

            XLinkHandler_t* handler = calloc(1, sizeof(XLinkHandler_t));
            if (!handler){
                mvLog(MVLOG_ERROR, "Memory allocation failed");
                return;
            }

            handler->protocol = stalledDevices[i].protocol;
            handler->devicePath = (char*)stalledDevices[i].name;
            XLinkError_t rc = XLinkConnect(handler);
            if (rc) {
                mvLog(MVLOG_ERROR," Failed to connect to stalled device, rc: %s", XLinkErrorToStr(rc));
            } else {

            }
            free(handler);
        }

        // This command will reset all previously connected devices
        XLinkError_t rc = XLinkResetAll();
        if (rc) {
            mvLog(MVLOG_WARN,"Failed to reset all device, rc: %s", XLinkErrorToStr(rc));
        }

        // Check that all devices are rebooted
        stalled_count_after_reboot = 0;
        deviceDesc_t stalledDevicesAfterReboot[NC_MAX_DEVICES] = { { 0 } };
        XLinkFindAllSuitableDevices(
                X_LINK_BOOTED, in_deviceDesc,
                stalledDevicesAfterReboot, NC_MAX_DEVICES, &stalled_count_after_reboot);

        mvLog(MVLOG_INFO,"...");
        usleep(SLEEP_MS*1000);

    } while (stalled_count_after_reboot > 0 && timeInSeconds() < waittm);
#endif
}

static ncStatus_t initializeXLink();

ncStatus_t ncDeviceResetAll() {
#if defined(NO_BOOT)
    mvLog(MVLOG_INFO, "Devices will not be restarted for this configuration (NO_BOOT)");
#else
    if (!initialized) {
        ncStatus_t sc;
        if ((sc = initializeXLink()) != 0) {
            return sc;
        }
    }
    resetAll();
#endif
    return NC_OK;
}

static ncStatus_t initializeXLink()
{
    XLinkSetCommonTimeOutMsec(60 * 1000);
    // We sanitize the situation by trying to reset the devices that have been left open
    initialized = 1;
    devices = NULL;

    int sc = XLinkInitialize(&ghandler);
    if (sc != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR," Initialization failed, rc = %s\n", XLinkErrorToStr(sc));
        return NC_ERROR;
    }

#if !(defined(NO_BOOT))
    if (reset_all) {
        resetAll();
    }
#endif  // NO_BOOT
    return NC_OK;
}

/**
 * @brief Check is path exists (directory or file)
 */
static int isPathExists(const char* filePath) {
#if (defined(_WIN32) || defined(_WIN64))
    return ( _access( filePath, 0 ) != -1 ) ? 1 : 0;
#else
    return (  access( filePath, 0 ) != -1 ) ? 1 : 0;
#endif
}

static char getPathSeparator() {
#ifdef _WIN32
    return '\\';
#else
    return '/';
#endif
}

/**
 * @brief Add / or \\ at the end of the path, if doesn't have it
 */

static void addEndPathSeparator(char* buffer, const int buffer_length) {
    const int filePathLen = (int)strnlen(buffer, buffer_length);
    if ((filePathLen > 1) && (filePathLen < buffer_length - 1) &&
            buffer[filePathLen - 1] != getPathSeparator()) {
        buffer[filePathLen] = getPathSeparator();
        buffer[filePathLen + 1] = 0;
    }
}

static char* getProtocolName(XLinkProtocol_t protocol) {
    if (protocol == X_LINK_PCIE) {
        return "pcie";
    } else if (protocol == X_LINK_USB_VSC) {
        return "usb";
    }
    return "";
}

static ncStatus_t getDeviceFwProtocolPrefix(const deviceDesc_t deviceDesc,
                                            char *fw_protocol_prefix,
                                            const int fw_protocol_prefix_length) {
    if (deviceDesc.protocol != X_LINK_USB_VSC && deviceDesc.protocol != X_LINK_PCIE) {
        return NC_INVALID_PARAMETERS;
    }

    int rc = mv_strcpy(fw_protocol_prefix, fw_protocol_prefix_length,
                                getProtocolName(deviceDesc.protocol));
    if (rc != 0) {
        return NC_ERROR;
    }
    return NC_OK;
}

static char* getDevicePlatform(deviceDesc_t deviceDesc, int useUniversalFirmware) {
    if (deviceDesc.platform == X_LINK_MYRIAD_X) {
        if (useUniversalFirmware) {
            return "ma2x8x";
        } else {
            return "ma248x";
        }
    } else if (deviceDesc.platform == X_LINK_MYRIAD_2) {
        return "ma2450";
    }
    return "";
}

static ncStatus_t getDeviceFwNameBody(const deviceDesc_t deviceDesc,
                                      char *fw_device_name,
                                      const int fw_device_name_length,
                                      const int useUniversalFirmware) {
    if (deviceDesc.platform != X_LINK_MYRIAD_2 && deviceDesc.platform != X_LINK_MYRIAD_X) {
        return NC_INVALID_PARAMETERS;
    }

    int rc = mv_strcpy(fw_device_name, fw_device_name_length,
                                getDevicePlatform(deviceDesc, useUniversalFirmware));
    if (rc != 0) {
        return NC_ERROR;
    }
    return NC_OK;
}

static ncStatus_t getDeviceFwFormat(const deviceDesc_t deviceDesc,
                                    char *fw_format, const int fw_formate_length) {
    // On Windows unified bootloader .elf file required instead of mvcmd
    if (deviceDesc.protocol == X_LINK_PCIE) {
#if defined(_WIN32)
        mv_strcpy(fw_format, fw_formate_length, ".elf");
#else
        mv_strcpy(fw_format, fw_formate_length, ".mvcmd");
#endif //defined(_WIN32)
    } else {
        mv_strcpy(fw_format, fw_formate_length, ".mvcmd");
    }
    return NC_OK;
}

static ncStatus_t getLibDirectory(char *firmware_directory, const int firwmare_directory_length) {
    // If firmware_directory contain path, use it.
    // It's case when firmware_directory was set by ncDeviceOpen custom path argument
    if (!strnlen(firmware_directory, firwmare_directory_length)) {
        int rc = 0;
        char path_to_lib_file[MAX_PATH_LENGTH] = {0};
        // Get dll full path
#if (defined(_WIN32) || defined(_WIN64))
        HMODULE hm = NULL;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR) "ncDeviceOpen", &hm)) {
            int ret = GetLastError();
            fprintf(stderr, "GetModuleHandle returned %d", ret);
        }
        GetModuleFileNameA(hm, path_to_lib_file, MAX_PATH_LENGTH - 1);
#else
        Dl_info info;
        dladdr(ncDeviceOpen, &info);
        rc = mv_strncpy(path_to_lib_file, MAX_PATH_LENGTH, info.dli_fname, MAX_PATH_LENGTH - 1);
        if (rc != 0) {
            return NC_ERROR;
        }
#endif
        // Path can contains library name. Use path before last '/'
        char* pointerToSeparator = NULL;
        size_t lib_dir_path_len = 0;

        pointerToSeparator = strrchr(path_to_lib_file, getPathSeparator());
        if(pointerToSeparator) {
            *pointerToSeparator = 0;
            lib_dir_path_len = pointerToSeparator - path_to_lib_file + 1;
        }

        rc = mv_strncpy(firmware_directory, firwmare_directory_length, path_to_lib_file, lib_dir_path_len);
        if (rc != 0) {
            return NC_ERROR;
        }
    }
    addEndPathSeparator(firmware_directory, firwmare_directory_length);
    return NC_OK;
}

static int isDeviceDescriptionCorrect(deviceDesc_t deviceDesc) {
    if (strnlen(deviceDesc.name, XLINK_MAX_NAME_SIZE) == 0) {
        mvLog(MVLOG_INFO, "Device name is empty");
        return 0;
    }

    //
    if (strstr(deviceDesc.name, "ma") && deviceDesc.protocol != X_LINK_USB_VSC) {
        mvLog(MVLOG_INFO, "Mismatch device name and protocol. Device name: %s. Protocol: %s",
              deviceDesc.name, ncProtocolToStr(convertProtocolToNC(deviceDesc.protocol)));
        return 0;
    }

    // PCIe devices should be named mxlk/mxlink
    if (strstr(deviceDesc.name, "mxl") && deviceDesc.protocol != X_LINK_PCIE) {
        mvLog(MVLOG_INFO, "Mismatch device name and protocol. Device name: %s. Protocol: %s",
                deviceDesc.name, ncProtocolToStr(convertProtocolToNC(deviceDesc.protocol)));
        return 0;
    }

    // Myriad 2 PCIe is not exists
    if ((strstr(deviceDesc.name, "2150") || deviceDesc.platform == X_LINK_MYRIAD_2)
                            && deviceDesc.protocol == X_LINK_PCIE) {
        mvLog(MVLOG_INFO, "Incorrect platform for PCIe device");
        return 0;
    }

    if (deviceDesc.protocol != X_LINK_ANY_PROTOCOL &&
        deviceDesc.protocol != X_LINK_USB_VSC &&
        deviceDesc.protocol != X_LINK_PCIE) {
        mvLog(MVLOG_INFO, "Protocol %s not supported",
                ncProtocolToStr(convertProtocolToNC(deviceDesc.protocol)));
        return 0;
    }
    return 1;
}

/**
 * Search the mvnc executable in the same directory of this library
 * in the future there will ideally be one FW file for all, for now they are separate
 */
ncStatus_t getFirmwarePath(char *firmware_file_path, const int firmware_file_length,
                           const deviceDesc_t deviceDesc) {

    if (!firmware_file_path || !isDeviceDescriptionCorrect(deviceDesc) ||
        deviceDesc.protocol == X_LINK_ANY_PROTOCOL) {
        return NC_INVALID_PARAMETERS;
    }

    ncStatus_t ncStatus;
    int rc;

    int useUniversalFirmware = 1;       // Try to use universal first

    char full_path_to_firmware[MAX_PATH_LENGTH]           = {0};

    char firmware_dir[FIRMWARE_DIR_LENGTH]                = {0};

    char firmware_full_name[FIRMWARE_NAME_LENGTH]         = {0};
    char fw_protocol_prefix[FIRMWARE_PROTOCOL_LENGTH]     = {0};
    char fw_device_name[FIRWMARE_DEVICE_LENGTH]           = {0};
    char fw_format[FIRMWARE_FORMAT_LENGTH]                = {0};

    // User can provide custom directory with firmware
    rc = mv_strncpy(firmware_dir, FIRMWARE_DIR_LENGTH, firmware_file_path,
                    strnlen(firmware_file_path, firmware_file_length));
    firmware_file_path[0] = 0;  // Clean input path

    if (rc != 0) {
        return NC_ERROR;
    }

    /// Construct file name
    ncStatus = getDeviceFwProtocolPrefix(deviceDesc, fw_protocol_prefix, FIRMWARE_PROTOCOL_LENGTH);
    if (ncStatus != NC_OK) {
        return ncStatus;
    }

    ncStatus = getDeviceFwNameBody(deviceDesc, fw_device_name, FIRWMARE_DEVICE_LENGTH,
                                   useUniversalFirmware);
    if (ncStatus != NC_OK) {
        return ncStatus;
    }

    ncStatus = getDeviceFwFormat(deviceDesc, fw_format, FIRMWARE_FORMAT_LENGTH);
    if (ncStatus != NC_OK) {
        return ncStatus;
    }

    rc = snprintf(firmware_full_name, FIRMWARE_NAME_LENGTH,
             "%s-%s%s", fw_protocol_prefix, fw_device_name, fw_format);
    if (rc < 0) {
        return NC_ERROR;
    }
    mvLog(MVLOG_DEBUG, "Firmware name %s", firmware_full_name);

    ///     Get file location
    ncStatus = getLibDirectory(firmware_dir, FIRMWARE_DIR_LENGTH);
    if (ncStatus != NC_OK) {
        return ncStatus;
    }
    mvLog(MVLOG_DEBUG, "Firmware dir %s", firmware_full_name);

    rc = snprintf(full_path_to_firmware, MAX_PATH_LENGTH, "%s%s", firmware_dir, firmware_full_name);
    if (rc < 0) {
        return NC_ERROR;
    }

    // If there is no universal firmware available, use a special one
    if (deviceDesc.protocol == X_LINK_USB_VSC && deviceDesc.platform == X_LINK_MYRIAD_X
                                                && !isPathExists(full_path_to_firmware)) {
        mvLog(MVLOG_INFO, "Cannot find universal firmware for ma2x8x. Try to find special one.");

        useUniversalFirmware = 0;
        ncStatus = getDeviceFwNameBody(deviceDesc, fw_device_name, FIRWMARE_DEVICE_LENGTH,
                                       useUniversalFirmware);
        if (ncStatus != NC_OK) {
            return ncStatus;
        }
        rc = snprintf(full_path_to_firmware, MAX_PATH_LENGTH,
                 "%s%s-%s%s", firmware_dir, fw_protocol_prefix, fw_device_name, fw_format);
        if (rc < 0) {
            return NC_ERROR;
        }
    }

    if (!isPathExists(full_path_to_firmware)) {
        mvLog(MVLOG_ERROR, "Firmware not found in: %s", full_path_to_firmware);
        return NC_ERROR;
    }
    rc = mv_strcpy(firmware_file_path, MAX_PATH_LENGTH, full_path_to_firmware);
    if (rc != 0) {
        return NC_ERROR;
    }

    mvLog(MVLOG_DEBUG, "File path %s", firmware_file_path);
    return 0;
}

static ncStatus_t getDevAttributes(struct _devicePrivate_t *d);
static void printfOverXLinkOpen(struct _devicePrivate_t *d);
static void printfOverXLinkClose(struct _devicePrivate_t *d);
static ncStatus_t destroyDeviceHandle(struct ncDeviceHandle_t **deviceHandlePtr);

ncStatus_t ncSetDeviceConnectTimeout(int deviceConnectTimeoutSec) {
    if(deviceConnectTimeoutSec < 0) {
        return NC_INVALID_PARAMETERS;
    }

    g_deviceConnectTimeoutSec = deviceConnectTimeoutSec;
    return NC_OK;
}

ncStatus_t ncDeviceOpen(struct ncDeviceHandle_t **deviceHandlePtr,
    struct ncDeviceDescr_t in_ncDeviceDesc, ncDeviceOpenParams_t deviceOpenParams) {

    //----------------------------------------------------------
    //      Check input

    deviceDesc_t deviceDescToBoot = {0};
    deviceDesc_t in_deviceDesc = {0};
    copyNcDeviceDescrToXLink(&in_ncDeviceDesc, &in_deviceDesc);

    int watchdogInterval = deviceOpenParams.watchdogInterval;
    const char* customFirmwareDirectory = deviceOpenParams.customFirmwareDirectory;

    CHECK_HANDLE_CORRECT_RC(deviceHandlePtr, NC_INVALID_PARAMETERS);
    CHECK_HANDLE_CORRECT_RC(deviceOpenParams.watchdogHndl, NC_INVALID_PARAMETERS);
    if (watchdogInterval < 0) {
        mvLog(MVLOG_ERROR, "Invalid watchdogInterval");
        return NC_INVALID_PARAMETERS;
    }

    bootOptions_t bootOptions = {0};
    bootOptions.memType = deviceOpenParams.memoryType;
    bootOptions.wdEnable = watchdogInterval > 0;

#ifdef NO_BOOT
    XLinkDeviceState_t state = X_LINK_BOOTED;
    if (watchdogInterval > 0) {
        mvLog(MVLOG_INFO, "Watchdog for already booted device would be disabled");
        watchdogInterval = 0;
    }

    // If trying open already booted device, we should not reset_all device on
    mvLog(MVLOG_INFO, "Connect to already booted device");
    reset_all = 0;
#else
    XLinkDeviceState_t state = X_LINK_UNBOOTED;
#endif

    if(!XLinkPlatformIsDescriptionValid(&in_deviceDesc, state)) {
        mvLog(MVLOG_ERROR, "Invalid in_ncDeviceDesc");
        return NC_INVALID_PARAMETERS;
    }

    if (*deviceHandlePtr && (*deviceHandlePtr)->private_data->state == NC_DEVICE_OPENED) {
        mvLog(MVLOG_WARN, "Device was already opened");
        return NC_OK;
    }

    //--------------------------------------------------------
    //      Initialize global mutex and mutex for deviceOpen

    if (!initialized) {
#if (defined(_WIN32) || defined(_WIN64))
        char* tempPath = getenv("TEMP");
        if (tempPath) {
            size_t pathSize = strlen(tempPath) + 15;
            char *path = malloc(pathSize);
            if (!path) {
                return NC_OUT_OF_MEMORY;
            }
            mv_strcpy(path, pathSize, tempPath);
            strcat_s(path, pathSize, "\\mvnc.mutex");
            global_lock_fd = CreateFile(path, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
            free(path);
        }

        if (!global_lock_fd) {
            mvLog(MVLOG_ERROR, "global mutex initialization failed");
            exit(1);
        }
#else
        global_lock_fd = open(
#if defined(ANDROID)
            "/data/local/tmp/mvnc.mutex",
#else
            "/tmp/mvnc.mutex",
#endif
            O_CREAT, 0660);

        if (global_lock_fd == -1) {
            mvLog(MVLOG_ERROR, "global mutex initialization failed");
            exit(1);
        }
#endif
    }

    GLOBAL_LOCK();

    if (!initialized) {
        ncStatus_t sc;
        if ((sc = initializeXLink()) != 0) {
            GLOBAL_UNLOCK();
            return sc;
        }
    }

    //--------------------------------------------------------
    //      Search for device

    XLinkError_t rc = X_LINK_ERROR;
    double waittm = timeInSeconds() + DEVICE_APPEAR_TIMEOUT_ON_OPEN;
    while ((rc != X_LINK_SUCCESS) && (timeInSeconds() < waittm)) {
        rc = XLinkFindFirstSuitableDevice(state, in_deviceDesc, &deviceDescToBoot);
    }

    if (rc != X_LINK_SUCCESS) {
        GLOBAL_UNLOCK();
        return parseXLinkError(NC_ERROR);
    }

    //--------------------------------------------------------
    //      Allocate device handler

    struct ncDeviceHandle_t *dH = calloc(1, sizeof(*dH));
    struct _devicePrivate_t *d = calloc(1, sizeof(*d));

    if (dH && d) {
        dH->private_data = d;
        d->protocol = deviceDescToBoot.protocol;
        d->dev_addr = mvnc_strdup(deviceDescToBoot.name);
        d->device_mon_stream_id = INVALID_LINK_ID;
        d->graph_monitor_stream_id = INVALID_LINK_ID;
        d->wd_interval = watchdogInterval;
        *deviceHandlePtr = dH;
    } else {
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        free(d);
        free(dH);
        GLOBAL_UNLOCK();
        return NC_OUT_OF_MEMORY;
    }

    if (d->dev_addr == NULL) {
        destroyDeviceHandle(deviceHandlePtr);
        GLOBAL_UNLOCK();
        return NC_OUT_OF_MEMORY;
    }

    //--------------------------------------------------------
    //      Boot device

    XLinkHandler_t* handler = calloc(1, sizeof(XLinkHandler_t));
    if (!handler) {
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        destroyDeviceHandle(deviceHandlePtr);
        GLOBAL_UNLOCK();
        return NC_OUT_OF_MEMORY;
    }

    handler->protocol = d->protocol;
    handler->devicePath = (char*)d->dev_addr;


#if (defined(NO_BOOT))
    d->protocol_booted = d->protocol;
    d->dev_addr_booted = mvnc_strdup(d->dev_addr);
    handler->protocol = d->protocol_booted;
    handler->devicePath = d->dev_addr_booted;
    rc = XLinkConnect(handler);
#else
    if (handler->protocol == X_LINK_PCIE) {                             // PCIe
        ncStatus_t sc;
        char mv_cmd_file_path[MAX_PATH_LENGTH] = { 0 };

        if (customFirmwareDirectory && strnlen(customFirmwareDirectory, MAX_PATH_LENGTH) > 1) {
            mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, customFirmwareDirectory, MAX_PATH_LENGTH - 1);
            addEndPathSeparator(mv_cmd_file_path, MAX_PATH_LENGTH);
            mv_cmd_file_path[MAX_PATH_LENGTH - 1] = '\0';
        }

        if ((sc = getFirmwarePath(mv_cmd_file_path, MAX_PATH_LENGTH, deviceDescToBoot)) != 0) {
            mvLog(MVLOG_ERROR, "Can't get firmware, error: %s", ncStatusToStr(sc));
            free(handler);
            destroyDeviceHandle(deviceHandlePtr);
            GLOBAL_UNLOCK();
            return NC_MVCMD_NOT_FOUND;
        }

        sc = bootDevice(&deviceDescToBoot, mv_cmd_file_path, bootOptions);
        if (sc) {
            mvLog(MVLOG_WARN, "%s() XLinkBootRemote returned error %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
            free(handler);
            destroyDeviceHandle(deviceHandlePtr);
            GLOBAL_UNLOCK();
            return NC_ERROR;
        } else {
            mvLog(MVLOG_INFO, "%s() XLinkBootRemote returned success %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
        }
        // Search and connect for booted device
        deviceDesc_t tempDeviceDesc = { 0 };

        d->protocol_booted = d->protocol;
        d->dev_addr_booted = mvnc_strdup(d->dev_addr);
        handler->protocol = d->protocol_booted;
        handler->devicePath = d->dev_addr_booted;

        int isDeviceAppeared = 0;
        int isDeviceConnected = 0;
        waittm = timeInSeconds() + g_deviceConnectTimeoutSec;
        do {
            if(!isDeviceAppeared) {
                rc = XLinkFindFirstSuitableDevice(X_LINK_BOOTED,
                    deviceDescToBoot, &tempDeviceDesc);

                if(rc != X_LINK_SUCCESS) {
                    continue;
                } else {
                    isDeviceAppeared = 1;
                }
            }

            rc = XLinkConnect(handler);
            if(rc == X_LINK_SUCCESS) {
                isDeviceConnected = 1;
            }

            if(isDeviceAppeared && isDeviceConnected) {
                break;
            }
        } while(timeInSeconds() < waittm);

        if (!isDeviceAppeared) {
            mvLog(MVLOG_ERROR, "Failed to find booted device after boot");
        }
    } else {                                        // USB
        // Find firmware and boot device with it
        char mv_cmd_file_path[MAX_PATH_LENGTH] = { 0 };

        // If have firmware directory path as function input, use it
        if (customFirmwareDirectory && strnlen(customFirmwareDirectory, MAX_PATH_LENGTH) > 1) {
            mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, customFirmwareDirectory, MAX_PATH_LENGTH - 1);
            addEndPathSeparator(mv_cmd_file_path, MAX_PATH_LENGTH);
        }

        ncStatus_t sc;

        if ((sc = getFirmwarePath(mv_cmd_file_path, MAX_PATH_LENGTH, deviceDescToBoot)) != 0) {
            mvLog(MVLOG_ERROR, "Can't get firmware, error: %s", ncStatusToStr(sc));
            free(handler);
            destroyDeviceHandle(deviceHandlePtr);
            GLOBAL_UNLOCK();
            return NC_MVCMD_NOT_FOUND;
        }

        mvLog(MVLOG_INFO, "%s() XLinkBootRemote is running for %s...\n", __func__, d->dev_addr);

        // Remember all currently available devices
        deviceDesc_t beforeBootDevices[NC_MAX_DEVICES] = {{0}};
        unsigned int numberOfDevicesBeforeBoot = 0;
        deviceDesc_t deviceDesc = {
            .platform = X_LINK_ANY_PLATFORM,
            .protocol = X_LINK_USB_VSC
        };

        XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, deviceDesc, beforeBootDevices,
                                    NC_MAX_DEVICES, &numberOfDevicesBeforeBoot);

        sc = bootDevice(&deviceDescToBoot, mv_cmd_file_path, bootOptions);
        if (sc) {
            mvLog(MVLOG_WARN, "%s() XLinkBootRemote returned error %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
        } else {
            mvLog(MVLOG_INFO, "%s() XLinkBootRemote returned success %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
        }

        // After boot name should change. Find
        deviceDesc_t foundBootedDevice = {0};
        int found_new_booted_device = 0;
        int device_disappear        = 0;
        int isDeviceConnected       = 0;

        deviceDesc_t afterBootDevices[NC_MAX_DEVICES] = { { 0 } };
        unsigned int numberOfDevicesAfterBoot = 0;

        waittm = timeInSeconds() + g_deviceConnectTimeoutSec;
        do {
            if(!found_new_booted_device) {
                XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, deviceDesc, afterBootDevices,
                                            NC_MAX_DEVICES, &numberOfDevicesAfterBoot);
                if (numberOfDevicesAfterBoot != numberOfDevicesBeforeBoot) {
                    continue;
                }
                deviceDesc_t tempDevicDescr = { 0 };

                // Device should disappear from unbooted list
                if (X_LINK_DEVICE_NOT_FOUND != XLinkFindFirstSuitableDevice(
                                                  X_LINK_ANY_STATE, deviceDescToBoot, &tempDevicDescr)) {
                    continue;
                } else {
                    device_disappear = 1;
                }
                int i, j;
                for (i = 0; i < (int)numberOfDevicesAfterBoot; ++i) {
                    int found_in_before_boot_list = 0;
                    for (j = 0; j < (int)numberOfDevicesBeforeBoot; ++j) {
                        if(strcmp(afterBootDevices[i].name, beforeBootDevices[j].name) == 0) {
                            found_in_before_boot_list = 1;
                        }
                    }
                    if (!found_in_before_boot_list) {
                        mv_strcpy(foundBootedDevice.name, XLINK_MAX_NAME_SIZE,
                                  afterBootDevices[i].name);
                        foundBootedDevice.platform = afterBootDevices[i].platform;
                        foundBootedDevice.protocol = afterBootDevices[i].protocol;
                        found_new_booted_device = 1;
                    }
                }
            }

            if(!found_new_booted_device) {
                continue;
            }

            handler->protocol = foundBootedDevice.protocol;
            handler->devicePath = (char *) foundBootedDevice.name;

            rc = XLinkConnect(handler);
            if(rc == X_LINK_SUCCESS) {
                isDeviceConnected = 1;
            }

            if(found_new_booted_device && isDeviceConnected) {
                d->protocol_booted = d->protocol;
                d->dev_addr_booted = mvnc_strdup(foundBootedDevice.name);
                break;
            }

        } while (timeInSeconds() < waittm);

        if (!found_new_booted_device) {
            mvLog(MVLOG_ERROR, "Failed to find booted device after boot");
            if (!device_disappear) {
                mvLog(MVLOG_WARN, "Device (%s) doesn't disappear after firmware loading",
                      deviceDescToBoot.name);
            }
            free(handler);
            destroyDeviceHandle(deviceHandlePtr);
            GLOBAL_UNLOCK();
            return NC_ERROR;
        }
    }
#endif

    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Failed connection to device (%s) with error %d", d->dev_addr, rc);
        free(handler);
        destroyDeviceHandle(deviceHandlePtr);
        GLOBAL_UNLOCK();
        return parseXLinkError(rc);
    }

    // After this line calling free(handler) and destroyDeviceHandle after each other is double-free corruption
    d->xlink = handler;
    d->next = devices;

    // Check device handle
    if (d->dev_addr == NULL || d->dev_addr_booted == NULL || d->xlink == NULL) {
        mvLog(MVLOG_ERROR, "device is invalid");
        destroyDeviceHandle(deviceHandlePtr);
        GLOBAL_UNLOCK();
        return NC_INVALID_HANDLE;
    }

    devices = d;

    mvLog(MVLOG_INFO, "XLinkConnect done - link Id %d\n", handler->linkId);
    int error = 0;
    if ((error = pthread_mutex_init(&d->dev_data_m, NULL)) != 0) {
        mvLog(MVLOG_ERROR, "pthread_mutex_init (dev_data_m) failed with error: %d", error);
        destroyDeviceHandle(deviceHandlePtr);
        GLOBAL_UNLOCK();
        return NC_ERROR;
    }
    // If current mutex initialization failed, destroy previous
    if ((error = pthread_mutex_init(&d->dev_stream_m, NULL)) != 0) {
        mvLog(MVLOG_ERROR, "pthread_mutex_init (dev_stream_m) failed with error: %d", error);
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));
        destroyDeviceHandle(deviceHandlePtr);
        GLOBAL_UNLOCK();
        return NC_ERROR;
    }
    if ((error = pthread_mutex_init(&d->graph_stream_m, NULL)) != 0) {
        mvLog(MVLOG_ERROR, "pthread_mutex_init (graph_stream_m) failed with error: %d", error);
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_stream_m));
        destroyDeviceHandle(deviceHandlePtr);
        GLOBAL_UNLOCK();
        return NC_ERROR;
    }

    if (handler->protocol != X_LINK_PCIE) {
        mvLog(MVLOG_INFO, "Booted %s (%s) -> %s\n",
              d->dev_addr, d->dev_addr_booted,
              d->dev_file ? d->dev_file : "VSC");
    } else {
        mvLog(MVLOG_INFO, "Booted %s -> %s\n",
              d->dev_addr, d->dev_file ? d->dev_file : "PCIe");
    }

    sleepForSeconds(1);

    GLOBAL_UNLOCK();

    streamId_t deviceMonitorStreamId = XLinkOpenStream(d->xlink->linkId, "deviceMonitor", CONFIG_STREAM_SIZE);
    CHECK_STREAM_ID(
        deviceMonitorStreamId,
        {
            CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->graph_stream_m));
            CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_stream_m));
            CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));
            destroyDeviceHandle(deviceHandlePtr);
        },
        "can't open deviceMonitor stream");

    d->device_mon_stream_id = deviceMonitorStreamId;

#if !(defined(NO_BOOT))
    if(bootOptions.wdEnable) {
        wd_error_t wd_rc = xlink_device_create(&d->watchdog_device, d);
        if (wd_rc) {
            mvLog(MVLOG_ERROR, "failed to start watchdog for device %p", d->xlink);
        } else {
            watchdog_register_device(deviceOpenParams.watchdogHndl, d->watchdog_device);
        }
    } else {
        mvLog(MVLOG_WARN, "watchdog is not started for device %p", d->xlink);
    }
#endif

    getDevAttributes(d);

#if (!defined(_WIN32) && !defined(_WIN64) && !defined(ANDROID))
    printfOverXLinkOpen(d);
#endif

    streamId_t graphMonitorStreamId = XLinkOpenStream(d->xlink->linkId, "graphMonitor", BLOB_STREAM_SIZE);

#if (!defined(_WIN32) && !defined(_WIN64) && !defined(ANDROID))
    CHECK_STREAM_ID(graphMonitorStreamId, {
        printfOverXLinkClose(d);
        // TODO NO_BOOT case
        if (d->watchdog_device != NULL) {
            watchdog_unregister_device(deviceOpenParams.watchdogHndl, d->watchdog_device);
            xlink_device_destroy(d->watchdog_device);
        }
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_stream_m));
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->graph_stream_m));
        XLinkError_t closed = XLinkCloseStream(deviceMonitorStreamId);
        if (closed != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Failed to close deviceMonitor stream");
        }

        destroyDeviceHandle(deviceHandlePtr);
    }, "can't open graphMonitor stream");
#else
    CHECK_STREAM_ID(graphMonitorStreamId, {
        // TODO NO_BOOT case
        if (d->watchdog_device != NULL) {
            watchdog_unregister_device(deviceOpenParams.watchdogHndl, d->watchdog_device);
            xlink_device_destroy(d->watchdog_device);
        }
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_stream_m));
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->graph_stream_m));
        XLinkError_t closed = XLinkCloseStream(deviceMonitorStreamId);
        if (closed != X_LINK_SUCCESS) {
            mvLog(MVLOG_ERROR, "Failed to close deviceMonitor stream");
        }

        destroyDeviceHandle(deviceHandlePtr);
    }, "can't open graphMonitor stream");
#endif

    d->graph_monitor_stream_id = graphMonitorStreamId;
    d->state = NC_DEVICE_OPENED;

    return NC_OK;
}

ncStatus_t ncAvailableDevices(struct ncDeviceDescr_t *deviceDescrPtr,
                              int maxDevices, int* out_countDevices) {
    CHECK_HANDLE_CORRECT(deviceDescrPtr);
    CHECK_HANDLE_CORRECT(out_countDevices);

    XLinkPlatformInit();
    memset(deviceDescrPtr, 0, maxDevices * sizeof(struct ncDeviceDescr_t));

    deviceDesc_t in_deviceDsc = {
        .platform = X_LINK_ANY_PLATFORM,
        .protocol = X_LINK_ANY_PROTOCOL
    };

    deviceDesc_t deviceDescArray[NC_MAX_DEVICES] = { { 0 } };
    unsigned int amountOfFoundDevices = 0;
    XLinkFindAllSuitableDevices(
            X_LINK_UNBOOTED, in_deviceDsc, deviceDescArray, NC_MAX_DEVICES, &amountOfFoundDevices);
    int i;
    for (i = 0; i < (int)amountOfFoundDevices; ++i) {
        copyXLinkDeviceDescrToNc(&deviceDescArray[i], &deviceDescrPtr[i]);
    }

    *out_countDevices = amountOfFoundDevices;
    return NC_OK;
}

ncStatus_t ncDeviceLoadFirmware(const ncDevicePlatform_t devicePlatform, const char* customFirmwareDir) {
    mvLog(MVLOG_WARN, "Boot (%s) without connecting to it", ncPlatformToStr(devicePlatform));
    XLinkError_t rc;
    ncStatus_t sc;

    // Find device with specific platform
    deviceDesc_t deviceDesc = {0};
    deviceDesc_t in_deviceDesc = {
        .platform = convertPlatformToXlink(devicePlatform),
        .protocol = X_LINK_USB_VSC
    };

    rc = XLinkFindFirstSuitableDevice(X_LINK_UNBOOTED, in_deviceDesc, &deviceDesc);
    if (rc) {
        mvLog(MVLOG_WARN, "Failed to find (%s) platform device", ncPlatformToStr(devicePlatform));
        return NC_DEVICE_NOT_FOUND;
    }

    if(deviceDesc.protocol == X_LINK_PCIE)
    {
        mvLog(MVLOG_WARN, "Firmware for PCIe can't be loaded with this application");
        return NC_ERROR;
    }

    // Find firmware
    char mv_cmd_file_path[MAX_PATH_LENGTH] = "\0";
    if (customFirmwareDir && strnlen(customFirmwareDir, MAX_PATH_LENGTH) > 1) {
        mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, customFirmwareDir, MAX_PATH_LENGTH - 1);
        addEndPathSeparator(mv_cmd_file_path, MAX_PATH_LENGTH);
        if (!isPathExists(customFirmwareDir)) {
            return NC_MVCMD_NOT_FOUND;
        }
    }

    if ((sc = getFirmwarePath(mv_cmd_file_path, MAX_PATH_LENGTH, deviceDesc)) != 0) {
        mvLog(MVLOG_ERROR, "Can't get firmware, error: %s", ncStatusToStr(sc));
        return NC_MVCMD_NOT_FOUND;
    }

    mvLog(MVLOG_INFO, "Trying to boot %s device", deviceDesc.name);
    rc = XLinkBoot(&deviceDesc, mv_cmd_file_path);
    if (rc) {
        mvLog(MVLOG_WARN, "%s() XLinkBootRemote returned error %s\n", __func__, XLinkErrorToStr(rc));
    } else {
        mvLog(MVLOG_INFO, "%s() XLinkBootRemote returned success %s\n", __func__, XLinkErrorToStr(rc));
          sleepForSeconds(DEVICE_APPEAR_TIMEOUT_ON_OPEN);
    }

    return parseXLinkError(rc);
}

static ncStatus_t getDevAttributes(struct _devicePrivate_t *d) {
    XLinkError_t rc = X_LINK_SUCCESS;
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&d->dev_stream_m), NC_ERROR);
    deviceCommand_t config = {0};
    config.type = DEVICE_GET_CAPABILITIES;
    rc = XLinkWriteData(d->device_mon_stream_id, (const uint8_t*)&config, sizeof(config));
    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Failed to write data, rc: %s", XLinkErrorToStr(rc));
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return parseXLinkError(rc);
    }
    streamPacketDesc_t* packet = 0;
    rc = XLinkReadData(d->device_mon_stream_id, &packet);
    if (rc != X_LINK_SUCCESS || !packet) {
        mvLog(MVLOG_ERROR, "Failed to read data, rc: %s", XLinkErrorToStr(rc));
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return parseXLinkError(rc);
    }
    if(packet->length != sizeof(d->dev_attr)) {
        mvLog(MVLOG_ERROR, "Broken protocol. DevData can't be read\n");
        if (XLinkReleaseData(d->device_mon_stream_id) != X_LINK_SUCCESS) {
            mvLog(MVLOG_WARN, "Failed to release data\n");
        }
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }
    d->dev_attr = *(deviceCapabilities_t*)packet->data;
    rc = XLinkReleaseData(d->device_mon_stream_id);
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_unlock(&d->dev_stream_m), NC_ERROR);
    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_WARN, "Failed to release data, rc: %s", XLinkErrorToStr(rc));
    }
    mvLog(MVLOG_INFO, "Device attributes\n");
    mvLog(MVLOG_INFO, "Device FW version: %x.%x.%x.%x\n", d->dev_attr.fw_version[0],
          d->dev_attr.fw_version[1], d->dev_attr.fw_version[2], d->dev_attr.fw_version[3]);
    mvLog(MVLOG_INFO, "Maximum graphs: %d\n", d->dev_attr.max_graphs);
    mvLog(MVLOG_INFO, "Maximum fifos: %d\n", d->dev_attr.max_fifos);
    mvLog(MVLOG_INFO, "Device memory capacity: %d\n", d->dev_attr.max_memory);
    return NC_OK;
}

static ncStatus_t getThermalStats(struct _devicePrivate_t *d){
    if (!d->thermal_stats){
        d->thermal_stats = calloc(THERMAL_THROTTLING_BUFFER_SIZE, 1);
        if (!d->thermal_stats)
            return NC_OUT_OF_MEMORY;
    }
    deviceCommand_t config;
    config.type = DEVICE_GET_THERMAL_STATS;
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&d->dev_stream_m), NC_ERROR);
    XLinkError_t rc = X_LINK_SUCCESS;
    rc = XLinkWriteData(d->device_mon_stream_id, (const uint8_t*)&config, sizeof(config));
    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_ERROR, "Failed to write data, rc: %s", XLinkErrorToStr(rc));
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return parseXLinkError(rc);
    }
    streamPacketDesc_t* packet = 0;
    rc = XLinkReadData(d->device_mon_stream_id, &packet);
    if (rc != X_LINK_SUCCESS || !packet) {
        mvLog(MVLOG_ERROR, "Failed to read data, rc: %s", XLinkErrorToStr(rc));
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return parseXLinkError(rc);
    }
    if( packet->length != THERMAL_THROTTLING_BUFFER_SIZE) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }
    mvnc_memcpy(d->thermal_stats, THERMAL_THROTTLING_BUFFER_SIZE, packet->data, packet->length);
    rc = XLinkReleaseData(d->device_mon_stream_id);
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_unlock(&d->dev_stream_m), NC_ERROR);
    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_WARN,"Failed to release data, rc: %s", XLinkErrorToStr(rc));
    }
    return NC_OK;
}

static ncStatus_t deviceGetDeviceMemory(struct _devicePrivate_t *d,
                                        uint32_t * mem)
{
    deviceCommand_t config;
    config.type = DEVICE_GET_USED_MEMORY;
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&d->dev_stream_m), NC_ERROR);
    if (XLinkWriteData(d->device_mon_stream_id, (const uint8_t *) &config,
                       sizeof(config)) != 0) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }
    streamPacketDesc_t *packet = 0;

    if (XLinkReadData(d->device_mon_stream_id, &packet) != 0 || !packet) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }

    if (packet->length != (sizeof(uint32_t))) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }
    mvnc_memcpy(mem, sizeof(uint32_t), packet->data, packet->length);
    XLinkReleaseData(d->device_mon_stream_id);
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_unlock(&d->dev_stream_m), NC_ERROR);
    return NC_OK;
}

static ncStatus_t deviceSetStdIO2XLink(struct _devicePrivate_t *d, uint32_t data)
{
    deviceCommand_t config;
    config.type = DEVICE_SET_STDIO_REDIRECT_XLINK;
    config.arg = data;
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&d->dev_stream_m), NC_ERROR);
    if (XLinkWriteData(d->device_mon_stream_id, (const uint8_t *) &config,
                       sizeof(config)) != 0) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_unlock(&d->dev_stream_m), NC_ERROR);
    return NC_OK;
}

#if (!defined(_WIN32) && !defined(_WIN64) && !defined(ANDROID))

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>

static void fprintfsock( int s, const char* fmt, ... ) {
    char* buf = NULL;
    const char* ptext;
    int len;
    va_list args;
    if(fmt == NULL) {
        va_start( args, fmt);
        ptext = va_arg(args,const char*);
        len = va_arg(args, int);
        va_end(args);
    } else {
        va_start( args, fmt );
        len = vsnprintf( 0, 0, fmt, args ) + 1;
        buf = (char*) malloc(len);
        if (buf == NULL) {
            va_end (args);
            return;
        }
        va_start( args, fmt );
        vsnprintf( buf, len, fmt, args );
        va_end (args);
        ptext = buf;
    }

    if(s < 0) {
        if(write( 1, ptext, len) != len) {
            fprintf(stderr, "Error in fprintfsock: write failed\n");
        }
    } else {
        if(send( s, ptext, len, 0 ) < 0)
        {
            fprintf(stderr,"WARNING in fprintfsock: not all data has been sent\n");
        }
    }

    if(buf)
        free( buf );
}

static void* debugConsoleThreadReader(void* ctx) {
    struct _devicePrivate_t *d = (struct _devicePrivate_t *) ctx;
    streamId_t streamId = d->printf_over_xlink_stream_id;
    int connfd = d->printf_over_xlink_conn_fd;
    streamPacketDesc_t * packet;
    XLinkError_t xerr;

    fprintfsock(connfd, "XLinkConsole receiving loop begins\n");
    fprintfsock(connfd, "=========================================\n");
    while(1){
        // use 0 as the timeout to prevent trigger false reset
        xerr = XLinkReadDataWithTimeOut(streamId, &packet, 0);
        if(X_LINK_SUCCESS != xerr || packet == NULL)
            break;
        fprintfsock(connfd, NULL, packet->data, packet->length);
        XLinkReleaseData(streamId);
    }
    fprintfsock(connfd, "=========================================\n"
                        "Session closed (%d)\n", xerr);
    close(connfd);
    return NULL;
}

static void printfOverXLinkClose(struct _devicePrivate_t *d) {
    if(d->printf_over_xlink_stream_id != INVALID_STREAM_ID) {
        /* Tell device stop redirect STDIO to XLink Console */
        deviceSetStdIO2XLink(d, 0);
        XLinkCloseStream(d->printf_over_xlink_stream_id);
        d->printf_over_xlink_stream_id = INVALID_STREAM_ID;
    }

    if(d->printf_over_xlink_thr_valid) {
        pthread_cancel(d->printf_over_xlink_thr);
        d->printf_over_xlink_thr_valid = 0;
    }

    if(d->printf_over_xlink_conn_fd >= 0) {
        close(d->printf_over_xlink_conn_fd);
        d->printf_over_xlink_conn_fd = -1;
    }
}

// FIXME: update the function below to use mvLog instead of printf for consistency: #16773
static void printfOverXLinkOpen(struct _devicePrivate_t *d) {
    int linkId = d->xlink->linkId;
    const char * streamName = "console";
    streamId_t streamId = INVALID_STREAM_ID;
    char * cfg_use_xlink_printf = NULL;

    d->printf_over_xlink_stream_id = INVALID_STREAM_ID;
    d->printf_over_xlink_conn_fd = -1;
    d->printf_over_xlink_thr_valid = 0;

    /* export XLINK_PRINTF=1 to enable this feature */
    cfg_use_xlink_printf = getenv("XLINK_PRINTF");
    if(cfg_use_xlink_printf == NULL)
        return;
    if(strcmp(cfg_use_xlink_printf, "1") != 0)
        return;

    /* Tell device redirect STDIO to XLink Console */
    deviceSetStdIO2XLink(d, 1);

    streamId = XLinkOpenStream(linkId, streamName, 10*1024);
    if(streamId == INVALID_STREAM_ID) {
        fprintf(stderr,"ERROR in XLinkOpenStream: %s\n", streamName);
        return;
    }

    const char * servername = "localhost";
    struct hostent *server;
    struct sockaddr_in serv_addr;

    server = gethostbyname(servername);
    if (server == NULL) {
        fprintf(stderr,"ERROR in gethostbyname: %s\n", servername);
        return;
    }

    int portNum = 7788;
    int connfd = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);

    bzero((char *) &serv_addr, sizeof(serv_addr));
    serv_addr.sin_family = AF_INET;
    bcopy((char *)server->h_addr, (char *)&serv_addr.sin_addr.s_addr, server->h_length);
    serv_addr.sin_port = htons(portNum);

    /* Now connect to the server */
    if (connect(connfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        printf("WARNNING: Cannot connect to XlinkPrintf debug console server, will print in current console instead\n");
        // even when no debug server, we still need drain possible debug information out of the XLink
        // or it will hang
        close(connfd);
        connfd = -1;
    }

    d->printf_over_xlink_stream_id = streamId;
    d->printf_over_xlink_conn_fd = connfd;
    if(pthread_create(&d->printf_over_xlink_thr, NULL, debugConsoleThreadReader, (void*) d)){
        fprintf(stderr,"ERROR in creating XlinkPrintf debug console reader thread!\n");
        printfOverXLinkClose (d);
    }else {
        d->printf_over_xlink_thr_valid = 1;
    }
}

#endif


static int findDevice(struct _devicePrivate_t *deviceHandle)
{

    struct _devicePrivate_t *d = devices;

    while (d) {
        if (d == deviceHandle)
            return 0;
        d = d->next;
    }

    return -1;
}

static int findGraph(struct _graphPrivate_t *graphHandle)
{
    struct _devicePrivate_t *d = devices;

    while (d) {
        struct _graphPrivate_t *g = d->graphs;
        while (g) {
            if (g == graphHandle)
                return 0;
            g = g->next;
        }
        d = d->next;
    }

    return -1;
}

// Defined here as it will be used twice
static int deallocateGraph(struct _graphPrivate_t *g)
{
    int found = 0;
    if (!g) {
        return -!found;
    }
    // Remove it from the list of the associated device
    if (g->dev->graphs == g) {
        g->dev->graphs = g->next;
        found = 1;
    } else {
        struct _graphPrivate_t *gp = g->dev->graphs;
        while (gp->next) {
            if (gp->next == g) {
                found = 1;
                gp->next = gp->next->next;
                break;
            }
            gp = gp->next;
        }
    }

    // Free it with all its data
    if (found) {
        free(g->aux_buffer);
    }
    g->state = NC_GRAPH_DEALLOCATED;
    return -!found;
}

static int findFifo(struct _fifoPrivate_t *f)
{
    if (!f || !f->dev)
        return 0;

    if (f->dev->fifos == f) {
        return 1;
    } else {
        struct _fifoPrivate_t *fp = f->dev->fifos;
        while (fp->next) {
            if (fp->next == f) {
                return 1;
            }
            fp = fp->next;
        }
    }
    return 0;
}

static int deallocateFifo(struct _fifoPrivate_t *f)
{
    int found = 0;
    if (!f) {
        return -!found;
    }
    // Remove it from the list of the associated device
    if (f->dev->fifos == f) {
        f->dev->fifos = f->next;
        found = 1;
    } else {
        struct _fifoPrivate_t *fp = f->dev->fifos;
        while (fp->next) {
            if (fp->next == f) {
                found = 1;
                fp->next = fp->next->next;
                break;
            }
            fp = fp->next;
        }
    }

    // Free it with all its data
    if (found) {
        //deallocate on device
        XLinkCloseStream(f->streamId);
        struct _userParamPrivate_t *temp;
        while (f->user_param_in) {
            temp = f->user_param_in;
            f->user_param_in = f->user_param_in->next;
            free(temp);
        }
        while (f->user_param_out) {
            temp = f->user_param_out;
            f->user_param_out = f->user_param_out->next;
            free(temp);
        }
    }
    f->state = NC_FIFO_DEALLOCATED;
    return -!found;
}

static ncStatus_t destroyDeviceHandle(struct ncDeviceHandle_t **deviceHandlePtr) {
    if (!deviceHandlePtr) {
        mvLog(MVLOG_ERROR, "Handle is NULL");
        return NC_INVALID_HANDLE;
    }
    if (!(*deviceHandlePtr)) {
        mvLog(MVLOG_INFO, "Handle already destroyed");
        return NC_OK;
    }

    mvLog(MVLOG_INFO, "Destroying device handler");

    struct _devicePrivate_t *d = (*deviceHandlePtr)->private_data;

    if(d->next) {
        mvLog(MVLOG_WARN, "Device could be in mvnc devices list");
    }

    free(d->thermal_stats);
    free(d->dev_addr);
    free(d->dev_addr_booted);

    free(d->dev_file);
    free(d->optimisation_list);

    free(d->xlink);

    free(d);
    (*deviceHandlePtr)->private_data = NULL;
    free((*deviceHandlePtr));
    *deviceHandlePtr = NULL;

    return NC_OK;
}

ncStatus_t ncDeviceClose(struct ncDeviceHandle_t **deviceHandlePtr, WatchdogHndl_t* watchdogHndl) {
    int found = 0;
    XLinkError_t rc = X_LINK_SUCCESS;

    if (!deviceHandlePtr) {
        mvLog(MVLOG_ERROR, "Handle is NULL");
        return NC_INVALID_HANDLE;
    }
    if (!(*deviceHandlePtr)) {
        mvLog(MVLOG_INFO, "Handle already destroyed");
        return NC_OK;
    }

    struct _devicePrivate_t *d = (*deviceHandlePtr)->private_data;
    if (!d) {
        mvLog(MVLOG_ERROR, "Device has been destroyed");
        return NC_INVALID_HANDLE;
    }

    int wasConnectedToBooted = 0;
    if (d->dev_addr != NULL && d->dev_addr_booted != NULL &&
        strncmp(d->dev_addr, d->dev_addr_booted, NC_MAX_NAME_SIZE) == 0) {
        // PCIe device have same booted and unbooted addr
        if (d->protocol != X_LINK_PCIE)
            wasConnectedToBooted = 1;
    }

    GLOBAL_LOCK();
    if (findDevice(d)) {
        GLOBAL_UNLOCK();
        return NC_INVALID_PARAMETERS;
    }
    mvLog(MVLOG_INFO, "Removing device...");

    // Remove it from our list
    if (devices == d) {
        devices = d->next;
        found = 1;
    } else {
        struct _devicePrivate_t *dp = devices;
        while (dp->next) {
            if (dp->next == d) {
                found = 1;
                dp->next = dp->next->next;
                break;
            }
            dp = dp->next;
        }
    }
    d->next = NULL;

    if (!found) {
        GLOBAL_UNLOCK();
        return NC_INVALID_PARAMETERS;
    }
    // Deallocate all associated graphs
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->dev_data_m));
    if (d->graphs) {
        mvLog(MVLOG_WARN,
              "Graphs on the device hasn't been destroyed! Graphs will be deallocated");
        while (deallocateGraph(d->graphs) != -1) {
            mvLog(MVLOG_INFO, "Graph was deallocated");
        }
    }
    // Deallocate all associated fifos
    if (d->fifos) {
        mvLog(MVLOG_WARN,
              "Fifos on the device hasn't been destroyed! Fifos will be deallocated");
        while (deallocateFifo(d->fifos) != -1) {
            mvLog(MVLOG_INFO, "Fifo was deallocated");
        }
    }

#if (!defined(_WIN32) && !defined(_WIN64) && !defined(ANDROID))
    printfOverXLinkClose(d);
#endif

#if !defined(NO_BOOT)
    if (d->watchdog_device != NULL) {
        watchdog_unregister_device(watchdogHndl, d->watchdog_device);
        xlink_device_destroy(d->watchdog_device);
    }
#endif

    // Save all devices before reset
    deviceDesc_t in_deviceDesc = {
            .platform = X_LINK_ANY_PLATFORM,
            .protocol = d->protocol
    };
    deviceDesc_t beforeResetDevices[NC_MAX_DEVICES] = {{0}};
    unsigned int foundDevicesBeforeReset = 0;
    XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, in_deviceDesc, beforeResetDevices,
                                NC_MAX_DEVICES, &foundDevicesBeforeReset);

        // #17801
#if !defined(NO_BOOT)
    if (d->graph_monitor_stream_id != INVALID_LINK_ID) {
        if (d->device_mon_stream_id != INVALID_LINK_ID) {
            rc = XLinkCloseStream(d->device_mon_stream_id);
            if (rc)
                mvLog(MVLOG_WARN,"Failed to close stream, rc: %s", XLinkErrorToStr(rc));
        }
        rc = XLinkCloseStream(d->graph_monitor_stream_id);
        if (rc)
            mvLog(MVLOG_WARN,"Failed to close stream, rc: %s", XLinkErrorToStr(rc));
    }
#endif
    // Reset device
    // In case when we open already booted device (or PCIE), just close connection to device
    rc = XLinkResetRemote(d->xlink->linkId);
    if (wasConnectedToBooted) {
        mvLog(MVLOG_INFO, "Only device handle will be released and link to device closed");
        if (rc)
            mvLog(MVLOG_WARN, "Failed to close link to device, rc: %s", XLinkErrorToStr(rc));
    } else {
        if (rc)
            mvLog(MVLOG_WARN, "Failed to reset, rc: %s", XLinkErrorToStr(rc));
    }

    d->state = NC_DEVICE_CLOSED;

    CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->graph_stream_m));
    CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_stream_m));

    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_data_m));
    CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));


    if (!wasConnectedToBooted && d->protocol != X_LINK_PCIE) {
        deviceDesc_t bootedDeviceDesc = {
                .protocol = d->protocol,
                .platform = X_LINK_ANY_PLATFORM
        };
        mv_strcpy(bootedDeviceDesc.name, XLINK_MAX_NAME_SIZE, d->dev_addr_booted);

        int booted_disappeared = 0;
        int unbooted_appeared = 0;

        //  Wait for unbooted device appear in usb list
        double waittm = timeInSeconds() + DEVICE_APPEAR_TIMEOUT_ON_CLOSE;

        deviceDesc_t afterResetDevices[NC_MAX_DEVICES] = {{0}};
        unsigned int foundDevicesAfterReset = 0;
        do {
            XLinkFindAllSuitableDevices(X_LINK_ANY_STATE, in_deviceDesc, afterResetDevices,
                                        NC_MAX_DEVICES, &foundDevicesAfterReset);
            if (foundDevicesAfterReset != foundDevicesBeforeReset) {
                continue;
            }

            deviceDesc_t deviceDesc = { 0 };

            rc = XLinkFindFirstSuitableDevice(X_LINK_BOOTED, bootedDeviceDesc, &deviceDesc);
            if (rc == X_LINK_SUCCESS) {
                continue;
            } else {
                booted_disappeared = 1;
            }
            int i, j;
            for (i = 0; i < (int)foundDevicesAfterReset; ++i) {
                int found_in_before_reset_list = 0;
                for (j = 0; j < (int)foundDevicesBeforeReset; ++j) {
                    if(strcmp(beforeResetDevices[i].name, afterResetDevices[j].name) == 0) {
                        found_in_before_reset_list = 1;
                    }
                }
                if (!found_in_before_reset_list) {
                    unbooted_appeared = 1;
                }
            }


        } while (!(booted_disappeared && unbooted_appeared) && timeInSeconds() < waittm);

        if (!booted_disappeared || !unbooted_appeared) {
            mvLog(MVLOG_ERROR, "Device didn't appear after reboot");
        }
    }

    ncStatus_t status = destroyDeviceHandle(deviceHandlePtr);
    GLOBAL_UNLOCK();
    if (status != NC_OK)
        mvLog(MVLOG_WARN, "Destroying device handle failed with error %s", ncStatusToStr(status));
    return status;
}

ncStatus_t ncGraphCreate(const char *name,
                         struct ncGraphHandle_t ** graphHandle)
{
    if ((!name) || (!graphHandle)) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    struct ncGraphHandle_t *gH = calloc(1, sizeof(*gH));
    struct _graphPrivate_t *g = calloc(1, sizeof(*g));

    if (!gH || !g) {
        free(g);
        free(gH);
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        return NC_OUT_OF_MEMORY;
    }

    gH->private_data = g;
    mv_strncpy(g->name, NC_MAX_NAME_SIZE, name, NC_MAX_NAME_SIZE - 1);
    g->batch_size = 1;
    g->dev = NULL;
    g->executors_number = 1;
    g->started = 0;
    g->state = NC_GRAPH_CREATED;
    *graphHandle = gH;
    return NC_OK;
}

ncStatus_t trySendCommand(streamId_t graphMonStream, void* buffer, int size) {
    XLinkError_t rc = XLinkWriteData(graphMonStream, (uint8_t*)buffer, size);
    return parseXLinkError(rc);
}

ncStatus_t getGraphMonitorResponseValue(streamId_t graphMonStream, ncMvNCIErrorCode_t *value) {
    streamPacketDesc_t *ack = NULL;
    XLinkError_t rc = X_LINK_SUCCESS;
    rc = XLinkReadData(graphMonStream, &ack);
    if (rc) {
        mvLog(MVLOG_ERROR, "XLink error, rc: %s", XLinkErrorToStr(rc));
        return parseXLinkError(rc);
    }

    if (value == NULL)
        return NC_ERROR;

    *value = MVNCI_SUCCESS;
    if (ack) {
        *value = *((int*)ack->data);
    } else {
        mvLog(MVLOG_ERROR, "Error with stream packet");
        return NC_ERROR;
    }

    rc = XLinkReleaseData(graphMonStream);
    if (rc) {
        mvLog(MVLOG_ERROR, "XLink error, rc: %s", XLinkErrorToStr(rc));
    }

    return NC_OK;
}

ncStatus_t checkGraphMonitorResponse(streamId_t graphMonStream) {
    ncMvNCIErrorCode_t value = MVNCI_SUCCESS;
    int rc = getGraphMonitorResponseValue(graphMonStream, &value);

    if (rc) {
        return rc;
    }

    if (value != MVNCI_SUCCESS){
        mvLog(MVLOG_ERROR, "Graph monitor request returned error %d", value);
        return NC_MYRIAD_ERROR;
    }

    return NC_OK;
}

static ncStatus_t lockAllInferences() {
    GLOBAL_LOCK();
    struct _devicePrivate_t *d = devices;
    while (d) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));
        d = d->next;
    }
    return NC_OK;
}

static ncStatus_t unlockAllInferences() {
    struct _devicePrivate_t *d = devices;
    while (d) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        d = d->next;
    }
    GLOBAL_UNLOCK();
    return NC_OK;
}

ncStatus_t ncGraphAllocate(struct ncDeviceHandle_t * deviceHandle,
                           struct ncGraphHandle_t * graphHandle,
                           const void *graphBuffer,
                           unsigned int graphBufferLength,
                           const void *graphHeader,
                           unsigned int graphHeaderLength)
{
    CHECK_HANDLE_CORRECT(deviceHandle);
    CHECK_HANDLE_CORRECT(graphHandle);
    CHECK_HANDLE_CORRECT(graphHeader);
    CHECK_HANDLE_CORRECT(graphBuffer);

    ncStatus_t rc = NC_OK;
    XLinkError_t xl_error = X_LINK_SUCCESS;
    mvLog(MVLOG_INFO, "Starting Graph allocation sequence\n");



    if (graphHeaderLength > graphBufferLength) {
        mvLog(MVLOG_ERROR, "graphHeaderLength > graphBufferLength");
        return NC_INVALID_PARAMETERS;
    }

    static int graphIdCount = 0;
    struct _graphPrivate_t *g = graphHandle->private_data;

    struct _devicePrivate_t *d = devices;
    GLOBAL_LOCK();
    while (d) {
        if (d == deviceHandle->private_data)
            break;
        d = d->next;
    }
    //TODO: review lists of devices and graphs internally.
    //TODO: check if the graph is not already on the device
    if (!d) {
        GLOBAL_UNLOCK();
        mvLog(MVLOG_ERROR, "Device not found!");
        return NC_INVALID_PARAMETERS;
    }
    GLOBAL_UNLOCK();

    if (graphBufferLength > d->dev_attr.max_memory) {
        mvLog(MVLOG_ERROR, "The graph file is bigger than the device memory");
        return NC_OUT_OF_MEMORY;
    }

    rc = lockAllInferences();
    if (rc != 0) {
        mvLog(MVLOG_ERROR, "can't lock all inferences");
        unlockAllInferences();
        return rc;
    }
    g->id = graphIdCount++;
    streamId_t streamId;

    if (g->executors_number > (int)d->dev_attr.max_executors) {
        mvLog(MVLOG_ERROR, "Executors number is greater than max allowed!");
        unlockAllInferences();
        return NC_INVALID_PARAMETERS;
    }

    graphCMDCommand_t cmd;
    cmd.type = GRAPH_VERIFY_CMD;
    snprintf(cmd.streamName, MAX_STREAM_NAME_LENGTH, "graphBuffer%d", g->id);
    streamId = XLinkOpenStream(d->xlink->linkId, cmd.streamName, graphBufferLength);
    CHECK_STREAM_ID(streamId, unlockAllInferences(), "can't open stream for graphBuffer transmission");

    cmd.id = g->id;
    cmd.executors_number = g->executors_number;

    rc = trySendCommand(d->graph_monitor_stream_id, &cmd, sizeof(cmd));
    if(rc != 0){
        mvLog(MVLOG_ERROR, "can't send graph allocation command");
        unlockAllInferences();
        return rc;
    }
    xl_error = XLinkWriteData(streamId, graphHeader, graphHeaderLength);
    if (xl_error) {
        mvLog(MVLOG_ERROR, "can't send graph header data to device, rc: %s", XLinkErrorToStr(xl_error));
        unlockAllInferences();
        return parseXLinkError(xl_error);
    }
    // for now simple status code used for graph header analysis result
    if ((rc = checkGraphMonitorResponse(d->graph_monitor_stream_id)) != 0) {
        mvLog(MVLOG_ERROR, "can't receive graph header verification response");
        unlockAllInferences();
        return rc;
    }

    // now sending whole graph with same header
    cmd.type = GRAPH_ALLOCATE_CMD;

    rc = trySendCommand(d->graph_monitor_stream_id, &cmd, sizeof(cmd));
    if(rc != 0){
        mvLog(MVLOG_ERROR, "can't send graph allocation command");
        unlockAllInferences();
        return rc;
    }
    xl_error = XLinkWriteData(streamId, graphBuffer, graphBufferLength);
    if (xl_error) {
        mvLog(MVLOG_ERROR, "can't send graph data to device, rc: %s", XLinkErrorToStr(xl_error));
        unlockAllInferences();
        return parseXLinkError(xl_error);
    }
    mvLog(MVLOG_INFO, "Sent graph");
    streamPacketDesc_t * tensorDescIn = 0;
    streamPacketDesc_t * tensorDescOut = 0;
    streamPacketDesc_t * nstages = 0;


    xl_error = XLinkReadData(streamId, &tensorDescIn);
    if (xl_error) {
        mvLog(MVLOG_ERROR, "Can't read input tensor descriptors of the graph, rc: %s", XLinkErrorToStr(xl_error));
        unlockAllInferences();
        return parseXLinkError(xl_error);
    }
    xl_error = XLinkReadData(streamId, &tensorDescOut);
    if (xl_error) {
        mvLog(MVLOG_ERROR, "Can't read output tensor descriptors of the graph, rc: %s", XLinkErrorToStr(xl_error));
        unlockAllInferences();
        return parseXLinkError(xl_error);
    }
    xl_error = XLinkReadData(streamId, &nstages);
    if (xl_error || nstages == NULL) {
        mvLog(MVLOG_WARN, "Can't read nstages, rc: %s", XLinkErrorToStr(xl_error));
        unlockAllInferences();
        return parseXLinkError(xl_error);
    }
    // for now, support only count 1
    if(!tensorDescIn ||
        tensorDescIn->length % sizeof(struct tensorDescriptor_t) ||
        tensorDescIn->length / sizeof(struct tensorDescriptor_t) > 1) {
        mvLog(MVLOG_ERROR, "Input tensor descriptors of the graph are invalid\n");
        if (tensorDescIn)
            mvLog(MVLOG_ERROR, "Received data from graph %d\n", *(int*)tensorDescIn->data);
        rc = NC_MYRIAD_ERROR;
    }
    // for now, support only count 1
    if(!tensorDescOut ||
        tensorDescOut->length % sizeof(struct tensorDescriptor_t) ||
        tensorDescOut->length / sizeof(struct tensorDescriptor_t) > 1) {
        mvLog(MVLOG_ERROR, "Output tensor descriptors of the graph are invalid\n");
        rc = NC_MYRIAD_ERROR;
    }
    if (rc == NC_OK){
        g->input_count = tensorDescIn->length / sizeof(struct tensorDescriptor_t);
        mvnc_memcpy(&g->input_tensor_desc, sizeof(struct tensorDescriptor_t),
                 tensorDescIn->data, sizeof(struct tensorDescriptor_t));
        g->output_count = tensorDescOut->length / sizeof(struct tensorDescriptor_t);
        mvnc_memcpy(&g->output_tensor_desc, sizeof(struct tensorDescriptor_t),
                 tensorDescOut->data, sizeof(struct tensorDescriptor_t));
        g->nstages = *(uint32_t*)nstages->data;
        g->batch_size = g->input_tensor_desc.n;
        g->timingsCount = g->nstages + 2;       // For time_receive timing and thread execution
    }

    xl_error = XLinkReleaseData(streamId);
    if (xl_error)
        mvLog(MVLOG_WARN, "Can't release data, rc: %s", XLinkErrorToStr(xl_error));

    xl_error = XLinkReleaseData(streamId);
    if (xl_error)
        mvLog(MVLOG_WARN, "Can't release data, rc: %s", XLinkErrorToStr(xl_error));

    xl_error = XLinkReleaseData(streamId);
    if (xl_error)
        mvLog(MVLOG_WARN, "Can't release data, rc: %s", XLinkErrorToStr(xl_error));

    g->graph_stream_id = streamId;
    if(checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        mvLog(MVLOG_ERROR, "The device didn't accept the graph\n");
        unlockAllInferences();
        return NC_ERROR;
    }
    if (rc) {
        unlockAllInferences();
        return rc;
    }

    ncMvNCIErrorCode_t allocation_error = MVNCI_SUCCESS;
    cmd.type = GRAPH_ALLOCATION_VERIFY_CMD;

    rc = trySendCommand(d->graph_monitor_stream_id, &cmd, sizeof(cmd));
    if(rc != 0){
        mvLog(MVLOG_ERROR, "can't send graph verification command");
        unlockAllInferences();
        return rc;
    }
    if (getGraphMonitorResponseValue(d->graph_monitor_stream_id, &allocation_error) != 0) {
        mvLog(MVLOG_ERROR, "Can't receive graph allocation verification response");
        unlockAllInferences();
        return NC_ERROR;
    }
    if (allocation_error != MVNCI_SUCCESS) {
        if (allocation_error == MVNCI_OUT_OF_MEMORY) {
            mvLog(MVLOG_ERROR, "Not enough memory to allocate intermediate tensors on remote device");
            unlockAllInferences();
            return NC_OUT_OF_MEMORY;
        } else {
            mvLog(MVLOG_ERROR, "Graph allocation caused an error %s on the device",
                  ncMvNCIErrorCodeToStr(allocation_error));
            unlockAllInferences();
            return NC_MYRIAD_ERROR;
        }
    }

    // aux_buffer
    g->aux_buffer = calloc(1, 224 + g->timingsCount * sizeof(*g->time_taken));
    if (!g->aux_buffer) {
        unlockAllInferences();
        return NC_OUT_OF_MEMORY;
    }
    // output_data

    g->debug_buffer = g->aux_buffer;
    g->time_taken = (float *) (g->aux_buffer + 120);
    rc = unlockAllInferences();
    if (rc != 0) {
        mvLog(MVLOG_ERROR, "Can't unlock all inferences");
        return rc;
    }

    GLOBAL_LOCK();
    g->dev = d;

    if (d->graphs)
        g->next = d->graphs;
    d->graphs = g;
    g->state = NC_GRAPH_ALLOCATED;
    GLOBAL_UNLOCK();
    mvLog(MVLOG_INFO, "Graph allocation completed successfully\n");

    return NC_OK;
}

ncStatus_t ncGraphDestroy(struct ncGraphHandle_t ** graphHandle)
{
    CHECK_HANDLE_CORRECT(graphHandle);

    ncStatus_t rc = NC_OK;
    struct ncGraphHandle_t *gh = *graphHandle;
    if (!gh) {
        mvLog(MVLOG_INFO, "handle is already destroyed");
        return NC_OK;
    }
    struct _graphPrivate_t *g = gh->private_data;
    CHECK_HANDLE_CORRECT_WINFO(g, MVLOG_ERROR, "Graph handle is corrupt or has been destroyed");

    if (g->state == NC_GRAPH_CREATED || g->state == NC_GRAPH_DEALLOCATED) {
        free(g);
        gh->private_data = NULL;
        free(gh);
        *graphHandle = NULL;
        return NC_OK;
    }
    GLOBAL_LOCK();
    if (findGraph(g)) {
        GLOBAL_UNLOCK();
        mvLog(MVLOG_ERROR, "This graph is corrupt or has been destroyed");
        return NC_INVALID_HANDLE;
    }

    GLOBAL_UNLOCK();
    struct _devicePrivate_t *d = (gh->private_data)->dev;

    graphCMDCommand_t cmd;
    cmd.type = GRAPH_DEALLOCATE_CMD;
    cmd.id = g->id;
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));
    rc = trySendCommand(d->graph_monitor_stream_id, &cmd, sizeof(cmd));
    if(rc != 0){
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        return rc;
    }
    if (checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        return NC_ERROR;
    }
    XLinkCloseStream(g->graph_stream_id);
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->dev_data_m));
    if (deallocateGraph(gh->private_data)) {
        mvLog(MVLOG_ERROR, "This graph has already been destroyed");
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_data_m));
        return NC_INVALID_PARAMETERS;
    }
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_data_m));
    free(g);
    gh->private_data = NULL;
    free(gh);
    *graphHandle = NULL;
    return NC_OK;
}

ncStatus_t ncGraphSetOption(struct ncGraphHandle_t * graphHandle,
                            ncGraphOption_t option, const void *data,
                            unsigned int dataLength)
{
    CHECK_HANDLE_CORRECT(graphHandle);
    CHECK_HANDLE_CORRECT_WINFO(graphHandle->private_data, MVLOG_ERROR, "graphHandle has been destroyed");
    if (!data) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    if (option < GRAPH_OPTION_BASE ||
        option > (GRAPH_OPTION_BASE + OPTION_CLASS_SIZE * NC_OP_ACCESS_LAST)) {
        mvLog(MVLOG_ERROR, "Option %d is invalid", option);
        return NC_INVALID_PARAMETERS;
    }
    if (option >= GRAPH_OPTION_BASE &&
        option <= (GRAPH_OPTION_BASE + OPTION_CLASS_SIZE)) {
        mvLog(MVLOG_ERROR, "Option %d is read only", option);
        return NC_UNAUTHORIZED;
    }
    struct _graphPrivate_t *g = graphHandle->private_data;

    GLOBAL_LOCK();
    if (option == NC_RW_GRAPH_EXECUTORS_NUM && g->state != NC_GRAPH_CREATED) {
        mvLog(MVLOG_ERROR,
              "This graph has already been alocated - cannot set option");
        GLOBAL_UNLOCK();
        return NC_UNAUTHORIZED;
    }
    if (option != NC_RW_GRAPH_EXECUTORS_NUM && g->state == NC_GRAPH_CREATED) {
        mvLog(MVLOG_ERROR,
              "This graph hasn't been allocated - cannot set option");
        GLOBAL_UNLOCK();
        return NC_UNAUTHORIZED;
    }
    if (option != NC_RW_GRAPH_EXECUTORS_NUM && findGraph(g)) {
        mvLog(MVLOG_ERROR, "This graph is corrupt or has been destroyed");
        GLOBAL_UNLOCK();
        return NC_INVALID_HANDLE;
    }
    GLOBAL_UNLOCK();

    ncOptionAccess_t opAccess = getOptionAccess(option, GRAPH_OPTION_BASE);
    if(opAccess == NC_OP_ACCESS_READ_ONLY) {
        mvLog(MVLOG_ERROR, "Option is read-only");
        return NC_UNAUTHORIZED;
    }

    if (opAccess != NC_OP_ACCESS_READ_WRITE) {
        mvLog(MVLOG_ERROR, "There is no such option");
        return NC_INVALID_PARAMETERS;
    }

    switch (option) {
    case NC_RW_GRAPH_EXECUTORS_NUM:
        {
            if (dataLength < sizeof(int)) {
                mvLog(MVLOG_ERROR, "The dataLength is smaller that required %zu",
                      sizeof(int));
                return NC_INVALID_DATA_LENGTH;
            }

            if (g->state != NC_GRAPH_CREATED) {
                mvLog(MVLOG_ERROR, "Can't set NCE number after graph allocation");
                return NC_UNAUTHORIZED;
            }
            g->executors_number = *(int *) data;
            break;
        }
    default:
        mvLog(MVLOG_ERROR, "There is no such option");
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}

static ncStatus_t getGraphOption(struct _graphPrivate_t *g,
                                       ncGraphOption_t option,
                                       void *data, unsigned int *dataLength)
{
    if ((option == NC_RO_GRAPH_INPUT_COUNT ||
         option == NC_RO_GRAPH_OUTPUT_COUNT ||
         option == NC_RW_GRAPH_EXECUTORS_NUM) && *dataLength < sizeof(int)) {
        mvLog(MVLOG_ERROR,
              "data length of data (%d) is smaller that required (%zu)!\n",
              *dataLength, sizeof(int));
        *dataLength = sizeof(int);
        return NC_INVALID_DATA_LENGTH;
    }

    ncStatus_t rc = NC_OK;
    graphCommonCommand_t cmd;
    streamPacketDesc_t* pack = 0;

    switch (option) {
    case NC_RO_GRAPH_INPUT_COUNT:
        *(int *) data = g->input_count;
        *dataLength = sizeof(int);
        break;
    case NC_RO_GRAPH_OUTPUT_COUNT:
        *(int *) data = g->output_count;
        *dataLength = sizeof(int);
        break;
    case NC_RO_GRAPH_TIME_TAKEN_ARRAY_SIZE:
        *(int *) data = sizeof(float) * g->timingsCount;
        *dataLength = sizeof(int);
        break;
    case NC_RO_GRAPH_TIME_TAKEN:
        CHECK_HANDLE_CORRECT(g->dev);
        if (*dataLength < sizeof(float) * g->timingsCount) {
            mvLog(MVLOG_ERROR,
                  "data length of output buffer (%d) is smaller that required (%zu)!\n",
                  *dataLength, sizeof(float) * g->timingsCount);
            *dataLength = sizeof(float) * g->timingsCount;
            return NC_INVALID_DATA_LENGTH;
        }
        cmd.id = g->id;
        cmd.type = GRAPH_GET_TIMING_DATA;
        CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&g->dev->graph_stream_m), NC_ERROR);
        rc = trySendCommand(g->dev->graph_monitor_stream_id, &cmd, sizeof(cmd));
        if(rc != 0){
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return rc;
        }
        if (XLinkReadData(g->dev->graph_monitor_stream_id, &pack) || !pack) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return NC_ERROR;
        }
        if (pack->length != sizeof(float) * g->timingsCount) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            XLinkReleaseData(g->dev->graph_monitor_stream_id);
            return NC_ERROR;
        }
        //Need to copy data before we check the response, since checkGraphMonitorResponse
        //calls releaseData
        mvnc_memcpy((float *) data, *dataLength, pack->data, pack->length);
        XLinkReleaseData(g->dev->graph_monitor_stream_id);

        if (checkGraphMonitorResponse(g->dev->graph_monitor_stream_id)) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return NC_ERROR;
        }

        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        *dataLength = sizeof(float) * g->timingsCount;
        break;
    case NC_RO_GRAPH_DEBUG_INFO:
        CHECK_HANDLE_CORRECT(g->dev);
        if (*dataLength < NC_DEBUG_BUFFER_SIZE) {
            mvLog(MVLOG_ERROR,
                  "data length of output buffer (%d) is smaller that required (%d)!\n",
                  *dataLength, NC_DEBUG_BUFFER_SIZE);
            *dataLength = NC_DEBUG_BUFFER_SIZE;
            return NC_INVALID_DATA_LENGTH;
        }

        cmd.type = GRAPH_GET_DEBUG_DATA;
        cmd.id = g->id;
        CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&g->dev->graph_stream_m), NC_ERROR);
        rc = trySendCommand(g->dev->graph_monitor_stream_id, &cmd, sizeof(cmd));
        if(rc != 0){
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return rc;
        }

        if (XLinkReadData(g->dev->graph_monitor_stream_id, &pack) || !pack) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return NC_ERROR;
        }

        if (pack->length != NC_DEBUG_BUFFER_SIZE) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            XLinkReleaseData(g->dev->graph_monitor_stream_id);
            return NC_ERROR;
        }

        mvnc_memcpy((char *) data, *dataLength, pack->data, pack->length);
        XLinkReleaseData(g->dev->graph_monitor_stream_id);
        if (checkGraphMonitorResponse(g->dev->graph_monitor_stream_id)) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return NC_ERROR;
        }
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));

        *dataLength = NC_DEBUG_BUFFER_SIZE;
        break;
    case NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS:{
            unsigned int size =
                sizeof(struct ncTensorDescriptor_t) * g->input_count;
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            mvnc_memcpy((struct ncTensorDescriptor_t *) data, *dataLength,
                       &g->input_tensor_desc, size);
            *dataLength = size;
            break;
        }
    case NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS:{
            unsigned int size =
                sizeof(struct ncTensorDescriptor_t) * g->output_count;
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            mvnc_memcpy((struct ncTensorDescriptor_t *) data, *dataLength,
                       &g->output_tensor_desc, size);
            *dataLength = size;
            break;
        }
    case NC_RO_GRAPH_VERSION:{
            unsigned int size = sizeof(g->blob_version);
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            mvnc_memcpy((int *) data, *dataLength, g->blob_version, size);
            *dataLength = size;
            break;
        }
    case NC_RW_GRAPH_EXECUTORS_NUM:{
        unsigned size = sizeof(int);
        if (*dataLength < size) {
            mvLog(MVLOG_ERROR,
                  "data length of data (%d) is smaller that required (%d)!\n",
                  *dataLength, size);
            *dataLength = size;
            return NC_INVALID_DATA_LENGTH;
        }
        *(int *) data = g->executors_number;
        *dataLength = size;
        break;
        }
    default:
        mvLog(MVLOG_ERROR, "There is no such option");
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}

ncStatus_t ncGraphGetOption(struct ncGraphHandle_t * graphHandle,
                            ncGraphOption_t option, void *data, unsigned int *dataLength)
{
    CHECK_HANDLE_CORRECT(graphHandle);
    CHECK_HANDLE_CORRECT_WINFO(graphHandle->private_data, MVLOG_ERROR, "graphHandle has been destroyed");

    if (!dataLength || (*dataLength != 0 && !data)) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    if (option < GRAPH_OPTION_BASE ||
        option > (GRAPH_OPTION_BASE + OPTION_CLASS_SIZE * NC_OP_ACCESS_LAST)) {
        mvLog(MVLOG_ERROR, "Option %d is invalid", option);
        return NC_INVALID_PARAMETERS;
    }

    struct _graphPrivate_t *g = graphHandle->private_data;
    CHECK_HANDLE_CORRECT(g);

    GLOBAL_LOCK();
    if (option != NC_RW_GRAPH_EXECUTORS_NUM && g->state == NC_GRAPH_CREATED) {
        mvLog(MVLOG_ERROR, "This graph hasn't been allocated");
        GLOBAL_UNLOCK();
        return NC_NOT_ALLOCATED;
    }
    ncOptionAccess_t opAccess = getOptionAccess(option, GRAPH_OPTION_BASE);
    if (opAccess != NC_OP_ACCESS_READ_ONLY &&
        opAccess != NC_OP_ACCESS_READ_WRITE) {
        mvLog(MVLOG_ERROR, "There is no such option");
        GLOBAL_UNLOCK();
        return NC_INVALID_PARAMETERS;
    }

    GLOBAL_UNLOCK();

    return getGraphOption(g, option, data, dataLength);
}

ncStatus_t ncGlobalSetOption(ncGlobalOption_t option, const void *data,
                             unsigned int dataLength)
{
    if (!data) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    switch (option) {
    case NC_RW_LOG_LEVEL:
        {
            mvLog_t log_level = *(mvLog_t *) data;
            if (log_level >= MVLOG_LAST || log_level < 0) {
                mvLog(MVLOG_ERROR, "log_level value is invalid %d\n",
                      log_level);
                return NC_INVALID_PARAMETERS;
            }
            mvLogLevelSet(*(mvLog_t *) data);
            mvLogDefaultLevelSet(*(mvLog_t *) data);    //Allow turning off warnings and errors
        }
        break;
    case NC_RO_API_VERSION:
        mvLog(MVLOG_ERROR, "API version is read-only");
        return NC_UNAUTHORIZED;
    case NC_RW_RESET_ALL:
        if (!initialized)
            reset_all = *(int*)data;
        break;
    case NC_RW_COMMON_TIMEOUT_MSEC: {
        int gTimeout = *(int *) data;
        XLinkError_t rc = XLinkSetCommonTimeOutMsec(gTimeout);
        if (rc) {
            mvLog(MVLOG_ERROR, "Set global common timeout failed, rc = %s\n", XLinkErrorToStr(rc));
            return NC_INVALID_PARAMETERS;
        }
        break;
    }
    case NC_RW_DEVICE_OPEN_TIMEOUT_MSEC: {
        int gTimeout = *(int *) data;
        XLinkError_t rc = XLinkSetDeviceOpenTimeOutMsec(gTimeout);
        if (rc) {
            mvLog(MVLOG_ERROR, "Set global open device timeout failed, rc = %s\n", XLinkErrorToStr(rc));
            return NC_INVALID_PARAMETERS;
        }
        break;
    }
    default:
        mvLog(MVLOG_ERROR, "No such option");
        return NC_INVALID_PARAMETERS;
    }

    return NC_OK;
}

ncStatus_t ncGlobalGetOption(ncGlobalOption_t option, void *data, unsigned int *dataLength)
{
    if (!data || !dataLength) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    switch (option) {
    case NC_RW_LOG_LEVEL:
        *(int *) data = mvLogLevel_ncAPI;
        *dataLength = sizeof(mvLogLevel_ncAPI);
        break;
    case NC_RO_API_VERSION:
        return NC_UNSUPPORTED_FEATURE;
        break;
    case NC_RW_RESET_ALL:
        *(int*)data = reset_all;
        *dataLength = sizeof(reset_all);
        break;
    default:
        mvLog(MVLOG_ERROR, "No such option");
        return NC_INVALID_PARAMETERS;
    }

    return NC_OK;
}

static ncStatus_t getDeviceOption(struct _devicePrivate_t *d,
                                        ncDeviceOption_t option,
                                        void *data, unsigned int *dataLength)
{
    ncStatus_t rc = NC_OK;

    switch (option) {
    case NC_RO_DEVICE_THERMAL_STATS:
        if (*dataLength < NC_THERMAL_BUFFER_SIZE) {
            mvLog(MVLOG_ERROR,
                  "data length of output buffer (%d) is smaller that required (%d)!\n",
                  *dataLength, NC_THERMAL_BUFFER_SIZE);
            *dataLength = NC_THERMAL_BUFFER_SIZE;
            return NC_INVALID_DATA_LENGTH;
        }
        rc = getThermalStats(d);
        if (rc) {
            return rc;
        }
        mvnc_memcpy((float *) data, *dataLength, &d->thermal_stats[1], NC_THERMAL_BUFFER_SIZE);
        *dataLength = NC_THERMAL_BUFFER_SIZE;
        break;
    case NC_RO_DEVICE_THERMAL_THROTTLING_LEVEL:
        rc = getThermalStats(d);
        if (rc) {
            return rc;
        }
        d->throttle_happened = (int)d->thermal_stats[0];
        *(int *) data = d->throttle_happened;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MEMORY_SIZE:
        *(int *) data = d->dev_attr.max_memory;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MAX_GRAPH_NUM:
        *(int *) data = d->dev_attr.max_graphs;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_NAME:
        if (*dataLength < strlen(d->dev_addr) + 1) {
            mvLog(MVLOG_ERROR,
                  "data length of output buffer (%d) is smaller that required (%zu)!\n",
                  *dataLength, strlen(d->dev_addr) + 1);
            *dataLength = (unsigned)(strlen(d->dev_addr) + 1);
            return NC_INVALID_DATA_LENGTH;
        }
        *dataLength = (unsigned)(strlen(d->dev_addr) + 1);
        mv_strncpy((char *) data, *dataLength, d->dev_addr, *dataLength - 1);
        break;
    case NC_RO_DEVICE_PLATFORM:
        if (d->dev_attr.fw_version[1] == 0x2480){
            *(ncDevicePlatform_t *) data = NC_MYRIAD_X;
        } else if (d->dev_attr.fw_version[1] == 0x2450) {
            *(ncDevicePlatform_t *) data = NC_MYRIAD_2;
        } else {
            *(ncDevicePlatform_t *) data = NC_ANY_PLATFORM;
        }
        *dataLength = sizeof(ncDevicePlatform_t);
        break;
    case NC_RO_DEVICE_PROTOCOL:
        *(ncDeviceProtocol_t *) data = convertProtocolToNC(d->protocol);
        *dataLength = sizeof(ncDeviceProtocol_t);
        break;
    case NC_RO_DEVICE_CURRENT_MEMORY_USED:{
            uint32_t mem;
            if (deviceGetDeviceMemory(d, &mem)) {
                rc = NC_ERROR;
                break;
            }
            *(int *) data = mem;
            *dataLength = sizeof(int);
            break;
        }
    default:
        mvLog(MVLOG_ERROR, "No such option");
        return NC_INVALID_PARAMETERS;
    }
    return rc;
}

static ncStatus_t setDevicePowerConfig(struct _devicePrivate_t *d,
                                        ncDeviceOption_t option,
                                        const void *data, unsigned int dataLength){
    XLinkError_t rc = X_LINK_SUCCESS;
    deviceCommand_t config;

    if (option != NC_RW_DEVICE_POWER_CONFIG_RESET && option != NC_RW_DEVICE_POWER_CONFIG) {
        mvLog(MVLOG_ERROR, "No such option");
        return NC_INVALID_PARAMETERS;
    }

    config.type = (option == NC_RW_DEVICE_POWER_CONFIG ? DEVICE_SET_POWER_CONFIG : DEVICE_RESET_POWER_CONFIG);
    config.arg = *(uint32_t*)data;

    rc = XLinkWriteData(d->device_mon_stream_id, (const uint8_t *)&config, sizeof(config));

    if (rc != X_LINK_SUCCESS)
    {
        mvLog(MVLOG_ERROR, "Failed to write data, rc: %s", XLinkErrorToStr(rc));
        return parseXLinkError(rc);
    }

    return NC_OK;
}

static ncStatus_t enableAsyncDMA(struct _devicePrivate_t *d,
                                 ncDeviceOption_t option,
                                 const void *data, unsigned int dataLength){
    XLinkError_t rc = X_LINK_SUCCESS;
    deviceCommand_t config;

    config.type = DEVICE_ENABLE_ASYNC_DMA;
    config.arg = *(uint32_t*)data;
    rc = XLinkWriteData(d->device_mon_stream_id, (const uint8_t *)&config, sizeof(config));
    if (rc != X_LINK_SUCCESS)
    {
        mvLog(MVLOG_ERROR, "Failed to write data, rc: %s", XLinkErrorToStr(rc));
        return parseXLinkError(rc);
    }

    return NC_OK;
}

ncStatus_t ncDeviceSetOption(struct ncDeviceHandle_t *deviceHandle,
                            ncDeviceOption_t option,
                            const void *data, unsigned int dataLength){
    ncStatus_t rc = NC_OK;
    if (!deviceHandle || !data){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    if (dataLength != sizeof(int) && dataLength != sizeof(void*)){
        mvLog(MVLOG_ERROR, "The dataLength must be %zu or %zu", sizeof(int), sizeof(void*));
        return NC_INVALID_PARAMETERS;
    }

    if (option < DEVICE_OPTION_BASE ||
        option > (DEVICE_OPTION_BASE + OPTION_CLASS_SIZE * NC_OP_ACCESS_LAST)) {
        mvLog(MVLOG_ERROR, "Option %d is invalid", option);
        return NC_INVALID_PARAMETERS;
    }

    ncOptionAccess_t opAccess = getOptionAccess(option, DEVICE_OPTION_BASE);
    if(opAccess == NC_OP_ACCESS_READ_ONLY) {
        mvLog(MVLOG_ERROR, "Option is read-only");
        return NC_UNAUTHORIZED;
    }

    struct _devicePrivate_t *d = deviceHandle->private_data;
    GLOBAL_LOCK();

    if (findDevice(d)) {
        mvLog(MVLOG_ERROR,
              "This device handle is corrupt or has been destroyed");
        GLOBAL_UNLOCK();

        return NC_INVALID_HANDLE;
    }

    if (opAccess != NC_OP_ACCESS_READ_WRITE) {
        mvLog(MVLOG_ERROR, "There is no such option");
        GLOBAL_UNLOCK();
        return NC_INVALID_PARAMETERS;
    }

    switch (option) {
        case NC_RW_DEVICE_POWER_CONFIG:
        case NC_RW_DEVICE_POWER_CONFIG_RESET:
        {
            rc = setDevicePowerConfig(d, option, data, dataLength);
            break;
        }
        case NC_RW_ENABLE_ASYNC_DMA:
        {
            rc = enableAsyncDMA(d, option, data, dataLength);
            break;
        }
        default:
            rc = NC_INVALID_PARAMETERS;
            mvLog(MVLOG_ERROR, "There is no such option");
    }

    GLOBAL_UNLOCK();
    return rc;
}

//static options can be read before device is open
static int isDeviceStaticOption(int option)
{
    switch (option) {
    case NC_RO_DEVICE_NAME:
        return 1;
    default:
        return 0;
    }
}

ncStatus_t ncDeviceGetOption(struct ncDeviceHandle_t * deviceHandle,
        ncDeviceOption_t option, void *data, unsigned int *dataLength)
{
    CHECK_HANDLE_CORRECT(deviceHandle);
    ncStatus_t rc;

    if (!dataLength || (*dataLength != 0 && !data)) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    if (option < DEVICE_OPTION_BASE ||
        option > (DEVICE_OPTION_BASE + OPTION_CLASS_SIZE * NC_OP_ACCESS_LAST)) {
        mvLog(MVLOG_ERROR, "Option %d is invalid", option);
        return NC_INVALID_PARAMETERS;
    }

    struct _devicePrivate_t *d = deviceHandle->private_data;

    GLOBAL_LOCK();
    if (!isDeviceStaticOption(option) && d->state != NC_DEVICE_OPENED) {
        mvLog(MVLOG_ERROR, "This device hasn't been opened");
        GLOBAL_UNLOCK();
        return NC_UNAUTHORIZED;
    }

    if (!isDeviceStaticOption(option)) {
        if (findDevice(d)) {
            mvLog(MVLOG_ERROR,
                  "This device handle is corrupt or has been destroyed");
            GLOBAL_UNLOCK();
            return NC_INVALID_HANDLE;
        }
    }

    ncOptionAccess_t opAccess = getOptionAccess(option, DEVICE_OPTION_BASE);
    if (opAccess != NC_OP_ACCESS_READ_ONLY &&
       opAccess != NC_OP_ACCESS_READ_WRITE) {
        mvLog(MVLOG_ERROR, "There is no such option");
        GLOBAL_UNLOCK();
        return NC_INVALID_PARAMETERS;
    }

    rc = getDeviceOption(d, option, data, dataLength);
    GLOBAL_UNLOCK();

    return rc;
}

static int fifoWriteAccess(struct _fifoPrivate_t *fifoHandle)
{
    if (fifoHandle->type == NC_FIFO_HOST_WO) {
        return 1;
    }
    return 0;
}

static int fifoReadAccess(struct _fifoPrivate_t *fifoHandle)
{
    if (fifoHandle->type == NC_FIFO_HOST_RO) {
        return 1;
    }
    return 0;
}

ncStatus_t ncFifoCreate(const char *name, ncFifoType_t type,
                        struct ncFifoHandle_t ** fifoHandle)
{
    mvLog(MVLOG_INFO, "Init fifo");
    CHECK_HANDLE_CORRECT(fifoHandle);
    CHECK_HANDLE_CORRECT(name);

    if (type != NC_FIFO_HOST_RO && type != NC_FIFO_HOST_WO) {
        mvLog(MVLOG_ERROR, "Fifo typo not supported!");
        return NC_UNSUPPORTED_FEATURE;
    }

    static int fifoIdCounter = 0;
    *fifoHandle = (struct ncFifoHandle_t *) malloc(sizeof(struct ncFifoHandle_t));
    if (!(*fifoHandle)) {
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        return NC_OUT_OF_MEMORY;
    }

    struct _fifoPrivate_t *handle = (struct _fifoPrivate_t *) malloc(sizeof(struct _fifoPrivate_t));
    (*fifoHandle)->private_data = handle;
    if (!handle) {
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        return NC_OUT_OF_MEMORY;
    }

    handle->type = type;
    handle->consumer_cnt = 1;   //default consumers

    handle->state = NC_FIFO_CREATED;
    CHECK_MUTEX_SUCCESS(pthread_mutex_init(&handle->fifo_mutex, NULL));
    handle->consumed_by_graph = 0;
    handle->write_count = 0;
    handle->user_param_in = NULL;
    handle->user_param_out = NULL;
    handle->api_read_element = 0;
    handle->id = fifoIdCounter++;
    handle->num_elements = 0;
    memset(&handle->host_tensor_desc, 0, sizeof(struct ncTensorDescriptor_t));
    mv_strncpy(handle->name, NC_MAX_NAME_SIZE, name, NC_MAX_NAME_SIZE - 1);

    return NC_OK;
}

int pushUserParam(struct _fifoPrivate_t *fH, void *user_param, int isIn)
{
    struct _userParamPrivate_t *new_user_param =
        calloc(1, sizeof(struct _userParamPrivate_t));
    if (!new_user_param) {
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        return NC_OUT_OF_MEMORY;
    }
    new_user_param->next = NULL;
    new_user_param->data = user_param;
    if (isIn) {
        new_user_param->next = fH->user_param_in;
        fH->user_param_in = new_user_param;
    } else {
        new_user_param->next = fH->user_param_out;
        fH->user_param_out = new_user_param;
    }
    return NC_OK;
}
int popUserParam(struct _fifoPrivate_t* fH, void** user_param, int isIn)
{
    struct _userParamPrivate_t* prev = NULL;
    struct _userParamPrivate_t* curr = NULL;
    if (isIn)
        curr = fH->user_param_in;
    else
        curr = fH->user_param_out;

    if (curr == NULL) {
        *user_param = NULL;
        mvLog(MVLOG_ERROR, "Trying to read user param from an empty queue!");
        return NC_ERROR;
    }

    while (curr->next != NULL)
    {
        prev = curr;
        curr = curr->next;
    }

    *user_param = curr->data;

    if (prev)
        prev->next = NULL;
    else {
        if (isIn)
            fH->user_param_in = NULL;
        else
            fH->user_param_out = NULL;
    }
    free(curr);
    curr = NULL;
    return NC_OK;
}

ncStatus_t ncFifoAllocate(struct ncFifoHandle_t * fifoHandle,
                          struct ncDeviceHandle_t * device,
                          struct ncTensorDescriptor_t * tensor_desc,
                          unsigned int numElem)
{
    mvLog(MVLOG_INFO, "Creating fifo");
    CHECK_HANDLE_CORRECT(fifoHandle);
    CHECK_HANDLE_CORRECT(device);

    if (!tensor_desc || !numElem) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    if (tensor_desc->n * tensor_desc->c * tensor_desc->w * tensor_desc->h == 0
        || !tensor_desc->totalSize) {
        mvLog(MVLOG_ERROR,
              "Tensor descriptor is invalid. Total size 0 or other element is zero");
        return NC_INVALID_PARAMETERS;
    }
    struct _fifoPrivate_t *handle = fifoHandle->private_data;
    if (handle->state == NC_FIFO_ALLOCATED) {
        mvLog(MVLOG_ERROR, "Fifo has already been allocated");
        return NC_UNAUTHORIZED;
    }
    if (handle->state != NC_FIFO_CREATED) {
        mvLog(MVLOG_ERROR, "Fifo handle is corrupt or has been destroyed");
        return NC_INVALID_HANDLE;
    }
    struct _devicePrivate_t *d = devices;
    GLOBAL_LOCK();
    while (d) {
        if (d == device->private_data)
            break;
        d = d->next;
    }
    if (!d) {
        GLOBAL_UNLOCK();
        mvLog(MVLOG_ERROR, "Device not found!\n");
        return NC_INVALID_PARAMETERS;
    }
    GLOBAL_UNLOCK();

    ncStatus_t rc = NC_OK;
    handle->graph_tensor_desc = *tensor_desc;
    handle->host_tensor_desc = *tensor_desc;
    handle->user_param_in = NULL;
    handle->user_param_out = NULL;
    handle->num_elements = numElem;
    handle->consumers_remaining = handle->consumer_cnt; //default consumers
    handle->dev = d;
    handle->next = NULL;

    handle->datasize = handle->host_tensor_desc.totalSize;

    if (d->fifos)
        handle->next = d->fifos;
    d->fifos = handle;

    bufferAllocateCommand_t cmd;
    cmd.type = GRAPH_BUFFER_ALLOCATE_CMD;
    struct tensorDescriptor_t privateDesc;
    privateDesc.c = tensor_desc->c;
    privateDesc.n = tensor_desc->n;
    privateDesc.h = tensor_desc->h;
    privateDesc.w = tensor_desc->w;
    // should be removiedd: #-17902
    privateDesc.totalSize = tensor_desc->totalSize;

    cmd.desc  = privateDesc;
    cmd.elemCnt = numElem;
    snprintf(cmd.name, MAX_STREAM_NAME_LENGTH, "FIFO%d", handle->id);
    cmd.id = handle->id;

    uint32_t writeSize;
    if (fifoWriteAccess(handle)) {
        writeSize = tensor_desc->totalSize * numElem;
        cmd.writeChannel = 1;
    } else {
        cmd.writeChannel = 0;
        writeSize = 8; // no write permission on this buffer, so we shouldn't bother allocating buffer on the device
    }
    if (fifoReadAccess(handle)) {
        cmd.readChannel = 1;
    } else {
        cmd.readChannel = 0;
    }
    streamId_t streamId = XLinkOpenStream(d->xlink->linkId, cmd.name, writeSize);

    char out_msg[NC_MAX_NAME_SIZE * 2];
    snprintf(out_msg, NC_MAX_NAME_SIZE * 2, "%s %s", "can't open stream: ", cmd.name);

    CHECK_STREAM_ID(streamId, {
            handle->state = NC_FIFO_FAILED;
            handle->dev->state = NC_DEVICE_FAILED;
        }, out_msg);

    handle->streamId = streamId;
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));

    rc = trySendCommand(d->graph_monitor_stream_id, &cmd, sizeof(cmd));
    if(rc != 0){
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        mvLog(MVLOG_ERROR, "can't send command\n");
        return rc;
    }
    if (checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        mvLog(MVLOG_ERROR, "myriad NACK\n");
        return NC_ERROR;
    }
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));

    handle->state = NC_FIFO_ALLOCATED;
    return NC_OK;

}

ncStatus_t ncFifoDestroy(struct ncFifoHandle_t ** fifoHandle)
{
    CHECK_HANDLE_CORRECT(fifoHandle);
    struct ncFifoHandle_t *fh = *fifoHandle;
    if (!fh) {
        mvLog(MVLOG_INFO, "handle is already destroyed");
        return NC_OK;
    }

    struct _fifoPrivate_t *handle = fh->private_data;

    if (handle->state == NC_FIFO_CREATED || handle->state == NC_FIFO_DEALLOCATED) {
        pthread_mutex_t * fifo_mutex = &fh->private_data->fifo_mutex;
#if !(defined(_WIN32) || defined(_WIN64))
        /**
         * There is no wrapper for pthread_mutex_trylock on windows at the moment.
         */
        int error = pthread_mutex_trylock(fifo_mutex);
        if (error && error != EBUSY) {
            /**
             * Calling pthread_mutex_unlock with not locked mutex is undefined behavior.
             * There is no standard C-API functions for checking whether mutex is locked or not as well as state entry.
             * After pthread_mutex_trylock mutex can be safely unlocked since it is already locked.
             * EBUSY error code stands for already locked mutex that is not an error in this case.
             */
             mvLog(MVLOG_ERROR, "pthread_mutex_trylock(fifo_mutex) failed with error: %d", error);
        }
#endif

        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(fifo_mutex));
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(fifo_mutex));

        free(fh->private_data);
        fh->private_data = NULL;

        free(fh);
        *fifoHandle = NULL;

        return NC_OK;
    }
    if (!findFifo(handle)) {
        mvLog(MVLOG_ERROR,
              "fifo handle seems to be corrupt or has been destroyed");
        return NC_INVALID_HANDLE;
    }
    //clean up fifo
    /*if (fifoReadAccess(handle)) {
        int fillLevel;
        int rc = XLinkGetFillLevel(handle->streamId, 0, &fillLevel);
        if (rc == X_LINK_SUCCESS) {
            while (fillLevel && rc == X_LINK_SUCCESS) {
                rc = XLinkReleaseData(handle->streamId);
                fillLevel--;
            }
        }
    }*/
    //First write to the fifo to stop it's thread
    if (fifoWriteAccess(handle)) {
        int msg = 0xdead;
        if (XLinkWriteData(handle->streamId, (uint8_t *) & msg, sizeof(msg)) !=
            0) {
            mvLog(MVLOG_ERROR, "Failed to write to fifo before deleting it!");
            return NC_ERROR;
        }
    }

    ncStatus_t rc = NC_OK;
    graphCommonCommand_t cmd;
    cmd.type = GRAPH_BUFFER_DEALLOCATE_CMD;
    cmd.id = handle->id;

    struct _devicePrivate_t *d = handle->dev;
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));
    rc = trySendCommand(d->graph_monitor_stream_id, &cmd, sizeof(cmd));
    if(rc != 0){
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        mvLog(MVLOG_WARN, "can't send command\n");
        return rc;
    }
    if (checkGraphMonitorResponse(d->graph_monitor_stream_id)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        mvLog(MVLOG_WARN, "myriad NACK\n");
        return NC_ERROR;
    }
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));

    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->dev_data_m));
    if (deallocateFifo(handle)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_data_m));
        return NC_INVALID_PARAMETERS;
    }
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_data_m));

    free(fh->private_data);
    fh->private_data = NULL;
    free(fh);
    *fifoHandle = NULL;
    return NC_OK;

}

ncStatus_t ncFifoWriteElem(struct ncFifoHandle_t * fifoHandle,
                           const void *inputTensor,
                           unsigned int * inputTensorLength,
                           void *userParam)
{
    CHECK_HANDLE_CORRECT(fifoHandle);

    if (inputTensorLength == NULL || *inputTensorLength <= 0) {
        mvLog(MVLOG_ERROR, "inputTensorSize is null or invalid value");
        return NC_INVALID_PARAMETERS;
    }
    struct _fifoPrivate_t *handle = fifoHandle->private_data;
    if (!findFifo(handle)) {
        if (!handle) {
            mvLog(MVLOG_ERROR,
                  "fifo handle seems to be corrupt or has been destroyed");
            return NC_INVALID_HANDLE;
        }
        if (handle->state == NC_FIFO_CREATED) {
            mvLog(MVLOG_ERROR, "FIFO is not yet allocated");
            return NC_NOT_ALLOCATED;
        }
        if (handle->state != NC_FIFO_ALLOCATED) {
            mvLog(MVLOG_ERROR,
                  "FIFO is not yet allocated or have been destroyed.");
            return NC_UNAUTHORIZED;
        }
    }

    CHECK_HANDLE_CORRECT_RC(inputTensor, NC_INVALID_PARAMETERS);

    if (!fifoWriteAccess(handle)) {
        mvLog(MVLOG_ERROR, "No write access to fifo");
        return NC_UNAUTHORIZED;
    }
    if (*inputTensorLength != handle->datasize) {
            mvLog(MVLOG_ERROR,
                  "input tensor length (%d) doesnt match expected value (%d)",
                  *inputTensorLength, handle->datasize);
            *inputTensorLength = handle->datasize;
            return NC_INVALID_DATA_LENGTH;
    }
    int rc = XLinkWriteData(handle->streamId, inputTensor, *inputTensorLength);
    if (rc != 0)
        return NC_ERROR;

    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&handle->fifo_mutex), NC_ERROR);
    rc = pushUserParam(handle, userParam, 1);
    if (rc != NC_OK) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&handle->fifo_mutex));
        return rc;
    }
    handle->write_count++;
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&handle->fifo_mutex));

    mvLog(MVLOG_DEBUG, "write count %d num_elements %d userparam %p\n",
          handle->write_count - 1, handle->num_elements, userParam);
    return NC_OK;

}

ncStatus_t ncFifoReadElem(struct ncFifoHandle_t * fifoHandle, void *outputData,
                          unsigned int *outputDataLen, void **userParam)
{
    if (!fifoHandle) {
        mvLog(MVLOG_ERROR, "fifo handle is NULL");
        return NC_INVALID_HANDLE;
    }
    if (!outputDataLen || (*outputDataLen != 0 && !outputData)) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    struct _fifoPrivate_t *handle = fifoHandle->private_data;
    if (!findFifo(handle)) {
        if (!handle) {
            mvLog(MVLOG_ERROR,
                  "fifo handle seems to be corrupt or has been destroyed");
            return NC_INVALID_HANDLE;
        }
        if (handle->state == NC_FIFO_CREATED) {
            mvLog(MVLOG_ERROR, "FIFO is not yet allocated");
            return NC_NOT_ALLOCATED;
        }
    }

    if (handle->state != NC_FIFO_ALLOCATED) {
        mvLog(MVLOG_ERROR, "FIFO is not yet allocated or have been destroyed.");
        return NC_UNAUTHORIZED;
    }

    if (*outputDataLen < (unsigned)handle->datasize) {
        mvLog(MVLOG_ERROR,
              "This datasize in tensorDesc (%d) is smaller than required (%d)!",
              *outputDataLen, handle->datasize);
        *outputDataLen = handle->datasize;
        return NC_INVALID_DATA_LENGTH;
    }

    if (!fifoReadAccess(handle)) {
        mvLog(MVLOG_ERROR, "FIFO has no read access");
        return NC_UNAUTHORIZED;
    }
    if (handle->api_read_element != 0) {
        mvLog(MVLOG_ERROR, "API already read this element");
        return NC_UNAUTHORIZED;
    }
    streamPacketDesc_t *packet = 0;
    if (!XLinkReadData(handle->streamId, &packet) && packet) {
        mvnc_memcpy(outputData, *outputDataLen, packet->data, packet->length);
        XLinkReleaseData(handle->streamId);
    } else {
        mvLog(MVLOG_ERROR, "Packet reading is failed.");
        return NC_ERROR;
    }

    //As user should see an API read to be the same as Graph read, we need to write the element in 2 queues.
    //if we read it here, we will need to remove the element on the device side
    //to avoid sending a message just for this purpose, we can send it at the next trigger which touches this FIFO.
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&handle->fifo_mutex), NC_ERROR);
    handle->api_read_element = 1;

    handle->consumers_remaining--;
    if (handle->consumers_remaining == 0) {
        handle->api_read_element = 0;
        handle->consumers_remaining = handle->consumer_cnt;
        //no other action required when the element is consumed
    }
    popUserParam(handle, userParam, 0);
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&handle->fifo_mutex));
    *outputDataLen = handle->datasize;
    mvLog(MVLOG_DEBUG, "num_elements %d userparam %p output length %d\n",
          handle->num_elements, userParam, handle->datasize);
    return NC_OK;
}

static ncStatus_t tensorCompatibility(struct ncTensorDescriptor_t *tens1,
                                      struct ncTensorDescriptor_t *tens2)
{
    if (tens1->totalSize != tens2->totalSize ||
        tens1->n != tens2->n || tens1->c != tens2->c ||
        tens1->h != tens2->h || tens1->w != tens2->w)
        return NC_ERROR;
    return NC_OK;
}

ncStatus_t ncGraphQueueInference(struct ncGraphHandle_t * graphHandle,
                                 struct ncFifoHandle_t ** fifoIn,
                                 unsigned int inFifoCount,
                                 struct ncFifoHandle_t ** fifoOut,
                                 unsigned int outFifoCount)
{
    mvLog(MVLOG_DEBUG, "Trigger start");
    CHECK_HANDLE_CORRECT(graphHandle);
    CHECK_HANDLE_CORRECT(fifoIn);
    CHECK_HANDLE_CORRECT(fifoOut);

    if (!fifoIn[0] || !fifoOut[0]) {
        mvLog(MVLOG_ERROR, "Fifos data are NULL");
        return NC_INVALID_HANDLE;
    }
    if (!inFifoCount || !outFifoCount)
        return NC_INVALID_PARAMETERS;

    struct _graphPrivate_t *g = graphHandle->private_data;

    if(g) {
        CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&g->dev->graph_stream_m), NC_ERROR);
    } else {
        return NC_NOT_ALLOCATED;
    }

    if (!g || g->state != NC_GRAPH_ALLOCATED) {
        mvLog(MVLOG_ERROR, "Graph hasn't been allocated");
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_NOT_ALLOCATED;
    }

    if (g->input_count != inFifoCount || g->output_count != outFifoCount) {
        mvLog(MVLOG_ERROR,
              "number of input or output fifos is not compatible with graph");
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_INVALID_PARAMETERS;
    }

    if (inFifoCount != 1 || outFifoCount != 1) {
        mvLog(MVLOG_ERROR,
              "Currently multiple inputs and outputs are not supported");
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_UNSUPPORTED_FEATURE;
    }
    struct _fifoPrivate_t *fi = fifoIn[0]->private_data;
    struct _fifoPrivate_t *fo = fifoOut[0]->private_data;
    ncStatus_t rc;
    if (fi->state != NC_FIFO_ALLOCATED || fo->state != NC_FIFO_ALLOCATED) {
        mvLog(MVLOG_ERROR, "ffos hasn't been allocated");
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_NOT_ALLOCATED;
    }
    //WO fifos have no graph access
    if (fo->type == NC_FIFO_HOST_WO) {
        //graphs have no access to one of the fifos
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_INVALID_PARAMETERS;
    }
    if (tensorCompatibility(&fi->graph_tensor_desc, &g->input_tensor_desc) != NC_OK ||
        tensorCompatibility(&fo->graph_tensor_desc,
                            &g->output_tensor_desc) != NC_OK) {
        mvLog(MVLOG_WARN,
              "Input/Output tensor shape is not compatible with graph");
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_INVALID_PARAMETERS;
    }

    graphCMDCommand_t cmd;
    cmd.type = GRAPH_TRIGGER_CMD;
    cmd.id = g->id;
    cmd.buffId1 = fi->id;
    cmd.buffId2 = fo->id;

    void* user_param;
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&fi->fifo_mutex), NC_ERROR);
    fi->consumers_remaining--;

    if (fi->consumers_remaining == 0) {
        if (!fi->api_read_element && fifoReadAccess(fi)) {//the element was entirely consumed by graphs. This means we need to free it up from XLink
            streamPacketDesc_t* packet = 0;
            XLinkError_t rc = XLinkReadData(fi->streamId, &packet);
            if (rc) {
                mvLog(MVLOG_ERROR, "Can't read packet, rc: %s", XLinkErrorToStr(rc));
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&fi->fifo_mutex));
                fi->dev->state = NC_DEVICE_FAILED;
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
                return parseXLinkError(rc);
            }
            rc = XLinkReleaseData(fi->streamId);
            if (rc) {
                mvLog(MVLOG_ERROR,"Failed to release data, rc: %s", XLinkErrorToStr(rc));
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&fi->fifo_mutex));
                fi->dev->state = NC_DEVICE_FAILED;
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
                return parseXLinkError(rc);
            }
        }
        fi->consumers_remaining = fi->consumer_cnt;
        fi->api_read_element = 0;
    }
    popUserParam(fi, &user_param, 1);
    if (fi->write_count <= fi->consumed_by_graph) {
        mvLog(MVLOG_WARN, "No point on triggering graph. There are no more elements in the input FIFO");
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&fi->fifo_mutex));
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_UNAUTHORIZED;
    }
    fi->consumed_by_graph++;
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&fi->fifo_mutex));

    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&fo->fifo_mutex), NC_ERROR);
    rc = pushUserParam(fo, user_param , 0);
    if(rc != NC_OK) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&fo->fifo_mutex));
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return rc;
    }
    fo->write_count++;
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&fo->fifo_mutex));

    rc = trySendCommand(g->dev->graph_monitor_stream_id, &cmd, sizeof(cmd));
    if(rc != 0){
        mvLog(MVLOG_ERROR, "Can't send trigger request");
        g->dev->state = NC_DEVICE_FAILED;
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return rc;
    }
    if(checkGraphMonitorResponse(g->dev->graph_monitor_stream_id)) {
        mvLog(MVLOG_ERROR, "Can't get trigger response");
        g->dev->state = NC_DEVICE_FAILED;
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_ERROR;
    }
    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
    g->started = 1;
    mvLog(MVLOG_DEBUG, "Trigger end");
    return NC_OK;
}

ncStatus_t ncGraphQueueInferenceWithFifoElem(struct ncGraphHandle_t *
                                             graphHandle,
                                             struct ncFifoHandle_t * fifoIn,
                                             struct ncFifoHandle_t * fifoOut,
                                             const void *inputTensor,
                                             unsigned int * inputTensorLength,
                                             void *userParam)
{
    ncStatus_t rc = ncFifoWriteElem(fifoIn, inputTensor, inputTensorLength,
                                    userParam);
    if (rc != NC_OK)
        return rc;

    return ncGraphQueueInference(graphHandle, &fifoIn, 1, &fifoOut, 1);
}
