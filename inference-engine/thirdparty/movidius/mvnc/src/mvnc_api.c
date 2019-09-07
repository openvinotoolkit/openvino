// Copyright (C) 2018-2019 Intel Corporation
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
#include "gettime.h"
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
#include "mvLog.h"
#include "mvnc_tool.h"
#include "mvMacros.h"
#include "mvStringUtils.h"
#include "watchdog.h"

#define THERMAL_BUFFER_SIZE 100
#define THERMAL_THROTTLING_BUFFER_SIZE (THERMAL_BUFFER_SIZE + sizeof(int))
#define DEBUG_BUFFER_SIZE     120

#define MAX_TENSORS_TO_LOAD (2)
#define BLOB_STREAM_SIZE 4096
#define TENSOR_STREAM_SIZE 320*1024   * MAX_TENSORS_TO_LOAD
#define OUTPUT_STREAM_SIZE 8 //read only from PC

#define CONFIG_STREAM_SIZE 2000

#define NAME_LENGTH         40
#define MAX_PATH_LENGTH         255
#define MAX_RELATED_PATH_LENGTH   100

//      Timeouts
#define STATUS_WAIT_TIMEOUT     15
#define DEVICE_APPEAR_TIMEOUT_ON_OPEN   (2)
#define DEVICE_APPEAR_TIMEOUT_ON_CLOSE   (10)

#define SLEEP_MS        250
#define MAX_ITERATIONS  20

#define GRAPH_CLASS0_BASE   1000
#define DEVICE_CLASS0_BASE  2000
#define OPTION_CLASS_SIZE   100

#define FP16_DATA_SIZE 2

static int initialized = 0;
static int reset_all = 1;

pthread_mutex_t deviceOpenMutex = PTHREAD_MUTEX_INITIALIZER;

#if (defined(_WIN32) || defined(_WIN64))
static HANDLE global_lock_fd = NULL;
static OVERLAPPED global_lock_overlap = { 0 };
#define GLOBAL_LOCK() LockFileEx(global_lock_fd, LOCKFILE_EXCLUSIVE_LOCK, 0, MAXDWORD, MAXDWORD, &global_lock_overlap)
#define GLOBAL_UNLOCK() UnlockFileEx(global_lock_fd, 0, MAXDWORD, MAXDWORD, &global_lock_overlap)
#else
static int global_lock_fd = -1;
#define GLOBAL_LOCK() flock(global_lock_fd, LOCK_EX)
#define GLOBAL_UNLOCK() flock(global_lock_fd, LOCK_UN)
#endif


// To suppress warning in the macro below
#pragma GCC diagnostic ignored "-Wformat-extra-args"

/**
 * @brief The macro checks a stream id passed to it
 * @param id Stream id to check
 * @param callReleasingResources if it is needed to release resource in case of error, put your code of releasing
 *        to { you code here }. If no need to release resource pass {} to the parameter
 * @param errorMsg Message to be written in case of error. It is a format string
 */
#ifndef CHECK_STREAM_ID
#define CHECK_STREAM_ID(id, callReleasingResources, errorMsg) {                                                   \
    char errorMsgWithReason[255];                                                                                  \
    if (id == INVALID_STREAM_ID_OUT_OF_MEMORY) {                                                                   \
        snprintf(errorMsgWithReason, 255, "%s %s", errorMsg, "due to not enough memory on device");                \
        mvLog(MVLOG_ERROR, errorMsgWithReason);                                                                    \
        callReleasingResources;                                                                                        \
        return NC_OUT_OF_MEMORY;                                                                                   \
    } else if (id == INVALID_STREAM_ID) {                                                                          \
         snprintf(errorMsgWithReason, 255, "%s %s", errorMsg, "due to unknown error");              \
         callReleasingResources;                                                                                       \
         return NC_ERROR;                                                                                          \
    }                                                                                                              \
    mvLog(MVLOG_DEBUG, "Stream opened");                                                                           \
}
#endif // CHECK_STREAM_ID

static XLinkGlobalHandler_t ghandler;

#define TRACE_SIZE (24)

static const int profUpperBound = 64 * 4 * 1024 * TRACE_SIZE;

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
    case NC_OK:                 return "NC_OK";
    case NC_BUSY:               return "NC_BUSY";
    case NC_OUT_OF_MEMORY:      return "NC_OUT_OF_MEMORY";
    case NC_DEVICE_NOT_FOUND:   return "NC_DEVICE_NOT_FOUND";
    case NC_INVALID_PARAMETERS: return "NC_INVALID_PARAMETERS";
    case NC_TIMEOUT:            return "NC_TIMEOUT";
    case NC_MVCMD_NOT_FOUND:    return "NC_MVCMD_NOT_FOUND";
    case NC_NOT_ALLOCATED:      return "NC_NOT_ALLOCATED";
    case NC_UNAUTHORIZED:       return "NC_UNAUTHORIZED";
    case NC_UNSUPPORTED_GRAPH_FILE: return "NC_UNSUPPORTED_GRAPH_FILE";
    case NC_UNSUPPORTED_CONFIGURATION_FILE: return "NC_UNSUPPORTED_CONFIGURATION_FILE";
    case NC_UNSUPPORTED_FEATURE: return "NC_UNSUPPORTED_FEATURE";
    case NC_MYRIAD_ERROR:       return "NC_MYRIAD_ERROR";
    case NC_INVALID_DATA_LENGTH: return "NC_INVALID_DATA_LENGTH";
    case NC_INVALID_HANDLE:     return "NC_INVALID_HANDLE";
    default: return "NC_ERROR";
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

static char* getProductName(const char* name) {

#if (defined(_WIN32) || defined(_WIN64)) && !defined(PCIE_NAME_STR)
    const char PCIeName[] = "mxlink";
#else
    const char PCIeName[] = "mxlk";
#endif

    if (!name) return NULL;
    if (strstr(name, PCIeName)) {
        return "-mv0262";
    } else {        // USB
        char* p = strchr(name, '-');
        if (p == NULL)
            return "";
        return p;
    }
}

static ncOptionClass_t getOptionClass(int option, int base)
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

static ncFifoLayout_t getLayout(struct ncTensorDescriptor_t* td) {
    unsigned int max = MAX_3(td->hStride, td->wStride, td->cStride);
    if (max == td->hStride) {
        if (MAX(td->wStride, td->cStride) == td->wStride)
            return NC_FIFO_HWC;
        else
            return NC_FIFO_HCW;
    } else if (max == td->cStride) {
        if (MAX(td->wStride, td->hStride) == td->hStride)
            return NC_FIFO_CHW;
        else
            return NC_FIFO_CWH;
    } else { //W is major
        if (MAX(td->hStride, td->cStride) == td->hStride)
            return NC_FIFO_WHC;
        else
            return NC_FIFO_WCH;
    }
}

void printImg(unsigned char* inputTensor, struct ncTensorDescriptor_t* inputDesc) {
    int c = 0;
    for (; c < inputDesc->c; c++) {
        int row = 0;
        for (; row < inputDesc->h; row++) { //row
            int col = 0;
            for (; col < inputDesc->w; col++) {
                printf("%x ", inputTensor[col + row * inputDesc->hStride +
                        c * inputDesc->cStride]);
            }
            printf(" ===== ROW %d (channel %d) Done === \n", row, c);
        }
        printf("\n");
    }
}

static void resetAll()
{
#if defined(NO_BOOT)
    mvLog(MVLOG_INFO, "Devices will not be restarted for this configuration (NO_BOOT)");
#else
    int index = 0;
    int stalled_count = 0;
    int iters = 0;
    int bootrom_count = 0;
    int after_reset_count = 0;
    XLinkError_t rc;
    deviceDesc_t out_deviceDesc;
    deviceDesc_t in_deviceDesc = {
        .protocol = X_LINK_USB_VSC,
        .platform = NC_ANY_PLATFORM
    };

    double waittm = timeInSeconds() + STATUS_WAIT_TIMEOUT;
    while (timeInSeconds() < waittm) {
        rc = XLinkFindDevice(index, X_LINK_ANY_STATE, &in_deviceDesc, &out_deviceDesc);
        if (rc != X_LINK_SUCCESS)
            break; //no more devices found

        if (strlen(getProductName(out_deviceDesc.name)) == 1 &&
            out_deviceDesc.protocol != X_LINK_PCIE) { //name doesn't have product number
            //device is already booted, need to reset
            mvLog(MVLOG_DEBUG,"Found stalled device %s\n", out_deviceDesc.name);
            XLinkHandler_t* handler = calloc(1, sizeof(XLinkHandler_t));

            if (!handler){
                mvLog(MVLOG_ERROR, "Memory allocation failed");
                break;
            }
            handler->protocol = out_deviceDesc.protocol;
            handler->devicePath = (char*)out_deviceDesc.name;
            rc = XLinkConnect(handler);
            if (rc) {
                mvLog(MVLOG_ERROR," Failed to connect to stalled device, rc: %s", XLinkErrorToStr(rc));
            }
            stalled_count++;
            free(handler);

        } else {
            bootrom_count++;
        }
        index++;
    }

    if (stalled_count) {
        mvLog(MVLOG_INFO,"Stalled devices found, Reseting...");
        rc = XLinkResetAll();
        if (rc) {
            mvLog(MVLOG_WARN,"Failed to reset all device, rc: %s", XLinkErrorToStr(rc));
        }

        iters = 0;

        while ((after_reset_count < bootrom_count + stalled_count) &&
                iters < MAX_ITERATIONS) {
            usleep(SLEEP_MS*1000);
            after_reset_count = 0;
            index = 0;
            waittm = timeInSeconds() + STATUS_WAIT_TIMEOUT;
            while (timeInSeconds() < waittm) {
                XLinkError_t rc = XLinkFindDevice(index, X_LINK_ANY_STATE, &in_deviceDesc, &out_deviceDesc);
                if (rc != X_LINK_SUCCESS)
                break; //no more devices found

                if (strlen(getProductName(out_deviceDesc.name)) > 1 &&
                    out_deviceDesc.protocol != X_LINK_PCIE) { //name has product number
                    after_reset_count++;
                }
                index++;
            }
            iters++;
            mvLog(MVLOG_INFO,"...");
        }
        usleep(SLEEP_MS*1000);
    }
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
    XLinkSetCommonTimeOutMsec(3 * 60 * 10000);
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

static int isDeviceOpened(const char *name)
{
    struct _devicePrivate_t *d = devices;
    while (d) {
        if (strcmp(d->dev_addr, name) == 0)
            return 0;
        d = d->next;
    }
    return -1;
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

static void addEndPathSeparator(char* filePath) {
    const int filePathLen = strnlen(filePath, MAX_PATH_LENGTH);
    if (filePathLen > 1 && filePathLen < MAX_PATH_LENGTH - 1 && filePath[filePathLen - 1] != getPathSeparator()) {
        filePath[filePathLen] = getPathSeparator();
        filePath[filePathLen + 1] = 0;
    }
}

ncStatus_t getFirmwarePath(char* mv_cmd_file_path, const char* dev_addr) {

    if (!mv_cmd_file_path || !dev_addr) {
        return NC_INVALID_PARAMETERS;
    }

    char *p;
    char mv_cmd_file_name[NAME_LENGTH] = "MvNCAPI-maXXXX.mvcmd";

    // Search the mvnc executable in the same directory of this library
    // in the future there will ideally be one FW file for all, for now they are separate
    const char* productName = getProductName(dev_addr);
    if (productName == NULL || strlen(productName) <= 1) {
        mvLog(MVLOG_WARN, "Can't get product name");
        GLOBAL_UNLOCK();
        return NC_ERROR;
    }

    // Get firmware name
    int useUniversalFirmware = 0;
    if (strstr(productName, "ma2480")) {
        snprintf(mv_cmd_file_name, NAME_LENGTH, "MvNCAPI%s.mvcmd", "-ma2x8x");
        useUniversalFirmware = 1;
    } else {
        snprintf(mv_cmd_file_name, NAME_LENGTH, "MvNCAPI%s.mvcmd", productName);
    }
    mvLog(MVLOG_DEBUG, "Firmware name %s\n", mv_cmd_file_name);

    // If mv_cmd_file_path contain path, use it.
    // It's case when mv_cmd_file_path was set by ncDeviceOpen custom path argument
    if (strlen(mv_cmd_file_path) > 1) {
        addEndPathSeparator(mv_cmd_file_path);
    } else {
        // Get dll full path
#if (defined(_WIN32) || defined(_WIN64))
        HMODULE hm = NULL;
        if (!GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                  GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                              (LPCSTR) "ncDeviceOpen", &hm)) {
            int ret = GetLastError();
            fprintf(stderr, "GetModuleHandle returned %d", ret);
        }
        GetModuleFileNameA(hm, mv_cmd_file_path, MAX_PATH_LENGTH - 1);
#else
        Dl_info info;
        dladdr(ncDeviceOpen, &info);
        mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, info.dli_fname, MAX_PATH_LENGTH - NAME_LENGTH);
#endif
    }

    p = strrchr(mv_cmd_file_path, getPathSeparator());
    size_t size_of_p = MAX_PATH_LENGTH - (p - mv_cmd_file_path);

    if (p)
        mv_strcpy(p + 1, size_of_p - 1, mv_cmd_file_name);
    else
        mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, mv_cmd_file_name, NAME_LENGTH - 1);
    mv_cmd_file_path[MAX_PATH_LENGTH - 1] = 0;

    // there is no universal firmware available, use a special one
    if (useUniversalFirmware && !isPathExists(mv_cmd_file_path)) {
        mvLog(MVLOG_INFO, "Cannot find universal firmware for ma2x8x. Try to find special one.");
        char *pos = strstr(mv_cmd_file_path, "-ma2x8x");
        if (pos == NULL) {
            mvLog(MVLOG_ERROR, "Incorrect firmware path.");
            return NC_MVCMD_NOT_FOUND;
        }
        pos[4] = productName[4]; pos[6] = productName[6];
    }

    if (!isPathExists(mv_cmd_file_path)) {
        mvLog(MVLOG_ERROR, "Firmware not found in: %s", mv_cmd_file_path);

        // Firmware also could be in "mvnc" subdirectory
        char mv_cmd_file_with_subdirectory[MAX_RELATED_PATH_LENGTH] = "mvnc/";
        char *p_sub = strrchr(mv_cmd_file_with_subdirectory, '/');

        if (!p_sub)
            return NC_MVCMD_NOT_FOUND;

        size_t size_of_p_sub = MAX_RELATED_PATH_LENGTH - (p_sub - mv_cmd_file_with_subdirectory);
        mv_strcpy(p_sub + 1, size_of_p_sub - 1, mv_cmd_file_name);
        if (p)
            mv_strncpy(p + 1, size_of_p - 1, mv_cmd_file_with_subdirectory, MAX_RELATED_PATH_LENGTH - 1);
        else
            mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, mv_cmd_file_with_subdirectory, MAX_RELATED_PATH_LENGTH - 1);

        // Is firmware was found in /mvnc subdir
        if (!isPathExists(mv_cmd_file_path)) {
            return NC_MVCMD_NOT_FOUND;
        } else {
            mvLog(MVLOG_WARN, "Firmware was found in: %s", mv_cmd_file_path);
        }
    }

    mvLog(MVLOG_DEBUG, "File path %s\n", mv_cmd_file_path);
    return 0;
}

static ncStatus_t getPCIeFirmwarePath(char* mv_cmd_file_path, const char* dev_addr) {
#ifndef NDEBUG  // Debug mode
    char* customPCIeFirmware;
    customPCIeFirmware = getenv("MVNC_PCIE_CUSTOM_FIRMWARE");
    if (customPCIeFirmware != NULL) {
        mvLog(MVLOG_INFO, "For PCIe will be used custom firmware %s",
              customPCIeFirmware);
        strncpy(mv_cmd_file_path, customPCIeFirmware, MAX_PATH_LENGTH);
        return NC_OK;
    } else {
        return getFirmwarePath(mv_cmd_file_path, dev_addr);
    }
#else
        return getFirmwarePath(mv_cmd_file_path, dev_addr);
#endif
}

static ncStatus_t getDevAttributes(struct _devicePrivate_t *d);
static void printfOverXLinkOpen(struct _devicePrivate_t *d);
static void printfOverXLinkClose(struct _devicePrivate_t *d);
static ncStatus_t destroyDeviceHandle(struct ncDeviceHandle_t **deviceHandlePtr);

ncStatus_t ncDeviceOpen(struct ncDeviceHandle_t **deviceHandlePtr,
    struct ncDeviceDescr_t in_ncDeviceDesc, int watchdogInterval, const char* customFirmwareDirectory) {
    deviceDesc_t out_deviceDesc = {0};
    deviceDesc_t in_deviceDesc = {0};
    copyNcDeviceDescrToXLink(&in_ncDeviceDesc, &in_deviceDesc);

    CHECK_HANDLE_CORRECT_RC(deviceHandlePtr, NC_INVALID_PARAMETERS);
    if (watchdogInterval < 0) {
        mvLog(MVLOG_ERROR, "Invalid watchdogInterval");
        return NC_INVALID_PARAMETERS;
    }

    if(!XLinkPlatformIsDescriptionValid(&in_deviceDesc)) {
        mvLog(MVLOG_ERROR, "Invalid in_ncDeviceDesc");
        return NC_INVALID_PARAMETERS;
    }

#ifdef NO_BOOT
    if (watchdogInterval > 0) {
        mvLog(MVLOG_INFO, "Watchdog for already booted device would be disabled");
        watchdogInterval = 0;
    }

    // If trying open already booted device, we should not reset_all device on
    mvLog(MVLOG_INFO, "Connect to already booted device");
    reset_all = 0;
#endif

    if (*deviceHandlePtr && (*deviceHandlePtr)->private_data->state == NC_DEVICE_OPENED) {
        mvLog(MVLOG_WARN, "Device was already opened");
        return NC_OK;
    }

    // Initialize handler

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
        global_lock_fd = open("/tmp/mvnc.mutex", O_CREAT, 0660);
        if (global_lock_fd == -1) {
            mvLog(MVLOG_ERROR, "global mutex initialization failed");
            exit(1);
        }
#endif
    }

    GLOBAL_LOCK();
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&deviceOpenMutex), NC_ERROR);
    if (!initialized) {
        ncStatus_t sc;
        if ((sc = initializeXLink()) != 0) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
            GLOBAL_UNLOCK();
            return sc;
        }
    }

#if defined(NO_BOOT)
    XLinkDeviceState_t state = X_LINK_BOOTED;
#else
    XLinkDeviceState_t state = X_LINK_UNBOOTED;
#endif

    // Find any unbooted device or booted device and create deviceHandle
    // TODO: PCIE could be found at once. Otherwise, it would cause a lot of errors about the opening file error.
    XLinkError_t rc = X_LINK_ERROR;
    double waittm = timeInSeconds() + DEVICE_APPEAR_TIMEOUT_ON_OPEN;
    while ((rc != X_LINK_SUCCESS) && (timeInSeconds() < waittm)) {
        rc = XLinkFindDevice(0, state, &in_deviceDesc, &out_deviceDesc);
    }

    if (rc != X_LINK_SUCCESS) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
        GLOBAL_UNLOCK();
        if (in_ncDeviceDesc.platform == NC_ANY_PLATFORM) {
            mvLog(MVLOG_WARN, "Failed to find a device, rc: %s", XLinkErrorToStr(rc));
        } else {
            /**
             * If user asked for a specific device and there is no suitable one then it can be an expected behavior
             * e.g. compile blob without a connected device
             */
            mvLog(MVLOG_WARN, "Failed to find %s device, rc: %s", ncPlatformToStr(in_ncDeviceDesc.platform), XLinkErrorToStr(rc));
        }
        return parseXLinkError(NC_ERROR);
    }

    // Allocate handler

    struct ncDeviceHandle_t *dH = calloc(1, sizeof(*dH));
    struct _devicePrivate_t *d = calloc(1, sizeof(*d));

    if (dH && d) {
        dH->private_data = d;
        d->protocol = out_deviceDesc.protocol;
        d->dev_addr = strdup(out_deviceDesc.name);
        d->device_mon_stream_id = INVALID_LINK_ID;
        d->graph_monitor_stream_id = INVALID_LINK_ID;
        d->wd_interval = watchdogInterval;
        *deviceHandlePtr = dH;
    } else {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
        GLOBAL_UNLOCK();
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        free(d);
        free(dH);
        return NC_OUT_OF_MEMORY;
    }

    if (d->dev_addr == NULL) {
        destroyDeviceHandle(deviceHandlePtr);
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
        GLOBAL_UNLOCK();
        return NC_OUT_OF_MEMORY;
    }

    // Boot device
    XLinkHandler_t* handler = calloc(1, sizeof(XLinkHandler_t));
    if (!handler) {
        mvLog(MVLOG_ERROR, "Memory allocation failed");
        destroyDeviceHandle(deviceHandlePtr);
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
        GLOBAL_UNLOCK();
        return NC_OUT_OF_MEMORY;
    }

    handler->protocol = d->protocol;
    handler->devicePath = (char*)d->dev_addr;


#if (defined(NO_BOOT))
    d->protocol_booted = d->protocol;
    d->dev_addr_booted = strdup(d->dev_addr);
    handler->protocol = d->protocol_booted;
    handler->devicePath = d->dev_addr_booted;
    rc = XLinkConnect(handler);
#else
    if (handler->protocol == X_LINK_PCIE) {          // PCIe
#if (!defined(_WIN32) && !defined(_WIN64))
        ncStatus_t sc;
        char mv_cmd_file_path[MAX_PATH_LENGTH] = {};
        if (customFirmwareDirectory && strnlen(customFirmwareDirectory, MAX_PATH_LENGTH) > 1) {
            mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, customFirmwareDirectory, MAX_PATH_LENGTH - 1);
            addEndPathSeparator(mv_cmd_file_path);
            mv_cmd_file_path[MAX_PATH_LENGTH - 1] = '\0';
        }

        if ((sc = getPCIeFirmwarePath(mv_cmd_file_path, d->dev_addr)) != 0) {
            mvLog(MVLOG_ERROR, "Can't get firmware, error: %s", ncStatusToStr(sc));
            free(handler);
            destroyDeviceHandle(deviceHandlePtr);
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
            GLOBAL_UNLOCK();
            return NC_MVCMD_NOT_FOUND;
        }
        rc = XLinkBootRemote(&out_deviceDesc, mv_cmd_file_path);
        if (rc) {
            mvLog(MVLOG_WARN, "%s() XLinkBootRemote returned error %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
        } else {
            mvLog(MVLOG_INFO, "%s() XLinkBootRemote returned success %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
        }
#endif
        d->protocol_booted = d->protocol;
        d->dev_addr_booted = strdup(d->dev_addr);
        handler->protocol = d->protocol_booted;
        handler->devicePath = d->dev_addr_booted;
        rc = XLinkConnect(handler);
    } else {                                        // USB
        // Find firmware and boot device with it
        char mv_cmd_file_path[MAX_PATH_LENGTH] = {0};

        // If have firmware directory path as function input, use it
        if (customFirmwareDirectory && strnlen(customFirmwareDirectory, MAX_PATH_LENGTH) > 1) {
            mv_strncpy(mv_cmd_file_path, MAX_PATH_LENGTH, customFirmwareDirectory, MAX_PATH_LENGTH - 1);
            addEndPathSeparator(mv_cmd_file_path);
        }

        ncStatus_t sc;

        if ((sc = getFirmwarePath(mv_cmd_file_path, d->dev_addr)) != 0) {
            mvLog(MVLOG_ERROR, "Can't get firmware, error: %s", ncStatusToStr(sc));
            free(handler);
            destroyDeviceHandle(deviceHandlePtr);
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
            GLOBAL_UNLOCK();
            return NC_MVCMD_NOT_FOUND;
        }

        mvLog(MVLOG_INFO, "%s() XLinkBootRemote is running for %s...\n", __func__, d->dev_addr);

        // remember all currently available devices
        deviceDesc_t beforeBootDevices[NC_MAX_DEVICES] = {{0}};
        deviceDesc_t simpleDeviceDesc = {
            .platform = NC_ANY_PLATFORM,
            .protocol = convertProtocolToXlink(in_ncDeviceDesc.protocol)
        };

        int n = 0;
        for (; n < NC_MAX_DEVICES; ++n) {
            if (XLinkFindDevice(n, X_LINK_ANY_STATE, &simpleDeviceDesc, &beforeBootDevices[n]))
                break;
        }

        rc = XLinkBootRemote(&out_deviceDesc, mv_cmd_file_path);
        if (rc) {
            mvLog(MVLOG_WARN, "%s() XLinkBootRemote returned error %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
        } else {
            mvLog(MVLOG_INFO, "%s() XLinkBootRemote returned success %s for %s",
                  __func__, XLinkErrorToStr(rc), d->dev_addr);
        }

        deviceDesc_t booted_device = {0};

        // After boot name should change
        double waittm = timeInSeconds() + STATUS_WAIT_TIMEOUT;
        int deviceBooted = 0;
        while ((timeInSeconds() < waittm) && !deviceBooted) {
            int dev_indx = 0;
            for (; dev_indx < NC_MAX_DEVICES; ++dev_indx) {
                rc = XLinkFindDevice(dev_indx, X_LINK_ANY_STATE, &simpleDeviceDesc, &booted_device);
                booted_device.name[NC_MAX_NAME_SIZE - 1] = 0;
                if (rc != X_LINK_SUCCESS)
                    break;

                // if beforeBootDevices contains booted_name this is not a device we are looking for
                int not_found = 0;
                n = 0;
                for (; n < NC_MAX_DEVICES; ++n) {
                    if (strcmp(booted_device.name, beforeBootDevices[n].name) == 0 ||
                        booted_device.protocol == X_LINK_PCIE) {
                        not_found = 1;
                        break;
                    }
                }

                if (not_found)
                    continue;
                handler->protocol = booted_device.protocol;
                handler->devicePath = (char *) booted_device.name;

                rc = XLinkConnect(handler);
                // Device mustn't be in devices pool
                if (isDeviceOpened(booted_device.name) < 0 && rc == X_LINK_SUCCESS) {
                    deviceBooted = 1;
                    d->protocol_booted = booted_device.protocol;
                    d->dev_addr_booted = strdup(booted_device.name);
                    break;
                }
            }
        }
    }
#endif

    if (rc != X_LINK_SUCCESS) {
        // If PCIE device was booted then we will find it but can not connect.
        mvLog_t logLevel = MVLOG_ERROR;
        if(in_deviceDesc.protocol == X_LINK_PCIE) {
            logLevel = MVLOG_WARN;
        }

        mvLog(logLevel, "Failed connection to device (%s) with error %d", d->dev_addr, rc);
        free(handler);
        destroyDeviceHandle(deviceHandlePtr);
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
        GLOBAL_UNLOCK();
        return parseXLinkError(rc);
    }
    mvLog(MVLOG_INFO, "XLinkConnect done - link Id %d\n", handler->linkId);

    int error = 0;
    if ((error = pthread_mutex_init(&d->dev_data_m, NULL)) != 0) {
        mvLog(MVLOG_ERROR, "pthread_mutex_init (dev_data_m) failed with error: %d", error);
        free(handler);
        destroyDeviceHandle(deviceHandlePtr);
        return NC_ERROR;
    }
    // If current mutex initialization failed, destroy previous
    if ((error = pthread_mutex_init(&d->dev_stream_m, NULL)) != 0) {
        mvLog(MVLOG_ERROR, "pthread_mutex_init (dev_stream_m) failed with error: %d", error);
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));
        free(handler);
        destroyDeviceHandle(deviceHandlePtr);
        return NC_ERROR;
    }
    if ((error = pthread_mutex_init(&d->graph_stream_m, NULL)) != 0) {
        mvLog(MVLOG_ERROR, "pthread_mutex_init (graph_stream_m) failed with error: %d", error);
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));
        CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_stream_m));
        free(handler);
        destroyDeviceHandle(deviceHandlePtr);
        return NC_ERROR;
    }

    d->xlink = handler;
    d->next = devices;
    devices = d;

    if (handler->protocol != X_LINK_PCIE) {
        mvLog(MVLOG_INFO, "Booted %s (%s) -> %s\n",
              d->dev_addr, d->dev_addr_booted,
              d->dev_file ? d->dev_file : "VSC");
    } else {
        mvLog(MVLOG_INFO, "Booted %s -> %s\n",
              d->dev_addr, d->dev_file ? d->dev_file : "X_LINK_PCIE");
    }

    sleepForSeconds(1);

    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&deviceOpenMutex));
    GLOBAL_UNLOCK();

    streamId_t streamId = XLinkOpenStream(d->xlink->linkId, "deviceMonitor", CONFIG_STREAM_SIZE);
    CHECK_STREAM_ID(streamId, {}, "can't open deviceMonitor stream");

    d->device_mon_stream_id = streamId;

#if !(defined(NO_BOOT))
    if(d->protocol != X_LINK_PCIE)
    {
        watchdog_init_context(&d->watchdog_ctx);
        watchdog_register_device(&d->watchdog_ctx, d);
    }
#endif

    getDevAttributes(d);

#if (!defined(_WIN32) && !defined(_WIN64))
    printfOverXLinkOpen(d);
#endif

    streamId = XLinkOpenStream(d->xlink->linkId, "graphMonitor",
                                BLOB_STREAM_SIZE);

#if (!defined(_WIN32) && !defined(_WIN64))
    CHECK_STREAM_ID(streamId, {
           printfOverXLinkClose(d);
    }, "can't open graphMonitor stream");
#else
    CHECK_STREAM_ID(streamId, {}, "can't open graphMonitor stream");
#endif

    d->graph_monitor_stream_id = streamId;
    d->state = NC_DEVICE_OPENED;

    return NC_OK;
}

ncStatus_t ncAvailableDevices(struct ncDeviceDescr_t *deviceDescrPtr,
                              int maxDevices, int* out_countDevices) {
    //TODO: PCIe device support can be performed after #-17972 is completed
    CHECK_HANDLE_CORRECT(deviceDescrPtr);
    CHECK_HANDLE_CORRECT(out_countDevices);

    XLinkPlatformInit();
    memset(deviceDescrPtr, 0, maxDevices * sizeof(struct ncDeviceDescr_t));

    deviceDesc_t in_deviceDsc = {
        .platform = NC_ANY_PLATFORM,
        .protocol = X_LINK_USB_VSC
    };

    int n = 0;
    for (; n < maxDevices; ++n) {
        deviceDesc_t deviceDsc = {0};
        if (XLinkFindDevice(n, X_LINK_UNBOOTED, &in_deviceDsc, &deviceDsc))
            break;

        copyXLinkDeviceDescrToNc(&deviceDsc, &deviceDescrPtr[n]);
    }

    *out_countDevices = n;
    return NC_OK;
}

ncStatus_t ncDeviceLoadFirmware(const ncDevicePlatform_t devicePlatform, const char* customFirmwareDir) {
    mvLog(MVLOG_WARN, "Boot (%s) without connecting to it", ncPlatformToStr(devicePlatform));
    XLinkError_t rc;
    ncStatus_t sc;

    // Find device with specific platform
    deviceDesc_t deviceDesc = {0};
    deviceDesc_t in_deviceDesc = {
        .platform = devicePlatform,
        .protocol = X_LINK_USB_VSC
    };

    rc = XLinkFindDevice(0, X_LINK_UNBOOTED, &in_deviceDesc, &deviceDesc);
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
        addEndPathSeparator(mv_cmd_file_path);
        if (!isPathExists(customFirmwareDir)) {
            return NC_MVCMD_NOT_FOUND;
        }
    }

    if ((sc = getFirmwarePath(mv_cmd_file_path, deviceDesc.name)) != 0) {
        mvLog(MVLOG_ERROR, "Can't get firmware, error: %s", ncStatusToStr(sc));
        return NC_MVCMD_NOT_FOUND;
    }

    mvLog(MVLOG_INFO, "Trying to boot %s device", deviceDesc.name);
    rc = XLinkBootRemote(&deviceDesc, mv_cmd_file_path);
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
    deviceCommand_t config;
    config.type.c0 = CLASS0_DEVICE_CAPABILITIES;
    config.optionClass = NC_OPTION_CLASS0;
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
    mvLog(MVLOG_INFO, "Maximum graph option class: %d\n", d->dev_attr.max_graph_opt_class);
    mvLog(MVLOG_INFO, "Maximum device option class: %d\n", d->dev_attr.max_device_opt_class);
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
    config.type.c0 = CLASS0_THERMAL_STATS;
    config.optionClass = NC_OPTION_CLASS0;
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

static ncStatus_t getDeviceFrequency(struct _devicePrivate_t *d){
    deviceCommand_t config;
    config.type.c0 = CLASS0_DEVICE_QUERY_CLOCKS;
    config.optionClass = NC_OPTION_CLASS0;
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

    if( packet->length != sizeof(uint32_t)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }
    mvnc_memcpy(&d->deviceFreq, sizeof(d->deviceFreq), packet->data, packet->length);
    rc = XLinkReleaseData(d->device_mon_stream_id);
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_unlock(&d->dev_stream_m), NC_ERROR);
    if (rc != X_LINK_SUCCESS) {
        mvLog(MVLOG_WARN,"Failed to release data, rc: %s", XLinkErrorToStr(rc));
    }
    return NC_OK;
}

static ncStatus_t getDeviceProfilingData(struct _devicePrivate_t *d){
    deviceCommand_t config;
    config.type.c0 = CLASS0_DEVICE_PROFILING_DATA;
    config.optionClass = NC_OPTION_CLASS0;
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

    d->receivedData = packet->length;
    if (d->profilingBuffer == 0) {
        d->profilingBuffer = (uint8_t*) malloc(profUpperBound);
    }

    if( packet->length > profUpperBound) {
        d->receivedData = profUpperBound;
    }
    mvnc_memcpy(d->profilingBuffer, profUpperBound, packet->data, d->receivedData);
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
    config.type.c0 = CLASS0_DEVICE_USED_MEMORY;
    config.optionClass = NC_OPTION_CLASS0;
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
    config.type.c2 = CLASS2_SET_STDIO_REDIRECT_XLINK;
    config.optionClass = NC_OPTION_CLASS2;
    config.data = data;
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&d->dev_stream_m), NC_ERROR);
    if (XLinkWriteData(d->device_mon_stream_id, (const uint8_t *) &config,
                       sizeof(config)) != 0) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_stream_m));
        return NC_ERROR;
    }
    CHECK_MUTEX_SUCCESS_RC(pthread_mutex_unlock(&d->dev_stream_m), NC_ERROR);
    return NC_OK;
}

#if (!defined(_WIN32) && !defined(_WIN64))

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
        (void) write( 1, ptext, len );
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

static int deviceGetNumberOfGraphs(struct _devicePrivate_t *deviceHandle)
{
    if (deviceHandle == NULL)
        return 0;
    int num = 0;
    struct _graphPrivate_t *g = deviceHandle->graphs;
    while (g) {
        num++;
        g = g->next;
    }
    return num;
}

static int deviceGetNumberOfFifos(struct _devicePrivate_t *deviceHandle)
{
    if (deviceHandle == NULL)
        return 0;
    int num = 0;
    struct _fifoPrivate_t *f = deviceHandle->fifos;
    while (f) {
        num++;
        f = f->next;
    }
    return num;
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

    free(d->profilingBuffer);

    free(d);
    (*deviceHandlePtr)->private_data = NULL;
    free((*deviceHandlePtr));
    *deviceHandlePtr = NULL;

    return NC_OK;
}


ncStatus_t ncDeviceClose(struct ncDeviceHandle_t **deviceHandlePtr) {
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
        wasConnectedToBooted = 1;       // For PCIE that also would work
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

#if (!defined(_WIN32) && !defined(_WIN64))
    printfOverXLinkClose(d);
#endif

    if (d->state != NC_DEVICE_FAILED) {
        // #17801
#if !defined(NO_BOOT)
        if (d->device_mon_stream_id != INVALID_LINK_ID &&
            d->protocol != X_LINK_PCIE) {
            rc = XLinkCloseStream(d->device_mon_stream_id);
            if (rc)
                mvLog(MVLOG_WARN,"Failed to close stream, rc: %s", XLinkErrorToStr(rc));
        }
        if (d->graph_monitor_stream_id != INVALID_LINK_ID &&
            d->protocol != X_LINK_PCIE) {
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
    }

#if !defined(NO_BOOT)
    if(d->protocol != X_LINK_PCIE) {
        watchdog_unregister_device(&d->watchdog_ctx);
    }
#endif

    d->state = NC_DEVICE_CLOSED;

    CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->graph_stream_m));
    CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_stream_m));

    CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->dev_data_m));
    CHECK_MUTEX_SUCCESS(pthread_mutex_destroy(&d->dev_data_m));

    if (!wasConnectedToBooted) {
        int device_appear_after_reboot = 0;

        //  Wait for unbooted device appear in usb list
        double waittm = timeInSeconds() + DEVICE_APPEAR_TIMEOUT_ON_CLOSE;
        while (timeInSeconds() < waittm) {
            // check current devices
            // wait for booted name to disappear
            // wait for unbooted name to appear
            // sometimes both names can be present in the list of usb devices
            deviceDesc_t device_desc = {0};
            deviceDesc_t in_deviceDesc = {
                .platform = NC_ANY_PLATFORM,
                .protocol = d->protocol
            };

            int booted_disappeared = 1;
            int unbooted_appeared = 0;

            int n = 0;
            while (XLinkFindDevice(n++, X_LINK_ANY_STATE, &in_deviceDesc, &device_desc) == X_LINK_SUCCESS) {
                if (d->dev_addr_booted != NULL &&
                    strcmp(device_desc.name, d->dev_addr_booted) == 0) {
                    booted_disappeared = 0;
                    break;
                }

                if (d->dev_addr != NULL &&
                    strcmp(device_desc.name, d->dev_addr) == 0) {
                    unbooted_appeared = 1;
                }
            }

            if (!(booted_disappeared && unbooted_appeared)) {
                continue;
            } else {
                device_appear_after_reboot = 1;
                break;
            }
        }

        if (device_appear_after_reboot == 0) {
            mvLog(MVLOG_ERROR, "Device didn't appear after reboot");
        }
    } else {
        // #16971
        sleepForSeconds(2);
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

ncStatus_t sendGraphMonitorRequest(streamId_t graphMonStream, graphMonCommand_t *cmd) {
    XLinkError_t rc = XLinkWriteData(graphMonStream, (uint8_t*)cmd, sizeof(*cmd));
    if (rc)
        return parseXLinkError(rc);
    return NC_OK;
}

ncStatus_t checkGraphMonitorResponse(streamId_t graphMonStream) {
    streamPacketDesc_t *ack = NULL;
    XLinkError_t rc = X_LINK_SUCCESS;
    rc = XLinkReadData(graphMonStream, &ack);
    if (rc) {
        mvLog(MVLOG_ERROR, "XLink error, rc: %s", XLinkErrorToStr(rc));
        return parseXLinkError(rc);
    }

    int value = 0;
    if (ack) {
        value = *((int*)ack->data);
    } else {
        mvLog(MVLOG_ERROR, "Error with stream packet");
        return NC_ERROR;
    }

    rc = XLinkReleaseData(graphMonStream);
    if (rc) {
        mvLog(MVLOG_ERROR, "XLink error, rc: %s", XLinkErrorToStr(rc));
    }
    if (value != 0){
        mvLog(MVLOG_ERROR, "Graph monitor request returned error %d", value);
        return NC_MYRIAD_ERROR;
    }

    return NC_OK;
}

static void lockAllInferences() {
    struct _devicePrivate_t *d = devices;
    while (d) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));
        d = d->next;
    }
    return;
}

static void unlockAllInferences() {
    struct _devicePrivate_t *d = devices;
    while (d) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        d = d->next;
    }
    return;
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
    if (graphBufferLength > d->dev_attr.max_memory) {
        mvLog(MVLOG_ERROR, "The graph file is bigger than the device memory");
        return NC_OUT_OF_MEMORY;
    }

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

    lockAllInferences();
    g->id = graphIdCount++;
    streamId_t streamId;

    if (g->executors_number > d->dev_attr.max_executors) {
        mvLog(MVLOG_ERROR, "Executors number is greater than max allowed!");
        unlockAllInferences();
        return NC_INVALID_PARAMETERS;
    }

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_GRAPH_CMD;
    cmd.cmd.graphCmd.type = GRAPH_VERIFY_CMD;
    snprintf(cmd.cmd.graphCmd.streamName, MAX_STREAM_NAME_LENGTH, "graphBuffer%d", g->id);
    streamId = XLinkOpenStream(d->xlink->linkId, cmd.cmd.graphCmd.streamName, graphBufferLength);
    CHECK_STREAM_ID(streamId, unlockAllInferences(), "can't open stream for graphBuffer transmission");

    cmd.cmd.graphCmd.id = g->id;
    cmd.cmd.graphCmd.executors_number = g->executors_number;

    if((rc = sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)) != 0){
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
    cmd.cmd.graphCmd.type = GRAPH_ALLOCATE_CMD;

    if(sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)){
        mvLog(MVLOG_ERROR, "can't send graph allocation command");
        unlockAllInferences();
        return NC_ERROR;
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

    // aux_buffer
    g->aux_buffer = calloc(1, 224 + g->timingsCount * sizeof(*g->time_taken));
    if (!g->aux_buffer) {
        unlockAllInferences();
        return NC_OUT_OF_MEMORY;
    }
    // output_data

    g->debug_buffer = g->aux_buffer;
    g->time_taken = (float *) (g->aux_buffer + 120);
    unlockAllInferences();

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

    struct ncGraphHandle_t *gh = *graphHandle;
    if (!gh) {
        mvLog(MVLOG_INFO, "handle is already destroyed");
        return NC_OK;
    }
    struct _graphPrivate_t *g = gh->private_data;
    CHECK_HANDLE_CORRECT_WINFO(g, MVLOG_ERROR, "Graph handle is corrupt or has been destroyed")

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

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_GRAPH_CMD;
    cmd.cmd.graphCmd.type = GRAPH_DEALLOCATE_CMD;
    cmd.cmd.graphCmd.id = g->id;
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));
    if (sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        return NC_ERROR;
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

static ncStatus_t setGraphOptionClass1(struct _graphPrivate_t *g,
                                       ncGraphOption_t option,
                                       const void *data,
                                       unsigned int dataLength)
{
    if (dataLength < sizeof(int)) {
        mvLog(MVLOG_ERROR, "The dataLength is smaller that required %zu",
              sizeof(int));
        return NC_INVALID_DATA_LENGTH;
    }
    switch (option) {
    case NC_RW_GRAPH_EXECUTORS_NUM:
        if (g->state != NC_GRAPH_CREATED) {
            mvLog(MVLOG_ERROR, "Can't set NCE number after graph allocation");
            return NC_UNAUTHORIZED;
        }
        g->executors_number = *(int *) data;;
        break;
    default:
        mvLog(MVLOG_ERROR, "There is no such option in class 1");
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}

static int isGraphPreAllocateOption(int option)
{
    switch (option) {
    case NC_RO_GRAPH_NAME:
    case NC_RO_GRAPH_STATE:
    case NC_RW_GRAPH_EXECUTORS_NUM:
        return 1;
    default:
        return 0;
    }
}

ncStatus_t ncGraphSetOption(struct ncGraphHandle_t * graphHandle,
                            int option, const void *data,
                            unsigned int dataLength)
{
    CHECK_HANDLE_CORRECT(graphHandle);
    CHECK_HANDLE_CORRECT_WINFO(graphHandle->private_data, MVLOG_ERROR, "graphHandle has been destroyed");
    if (!data) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    if (option < GRAPH_CLASS0_BASE ||
        option > (GRAPH_CLASS0_BASE + OPTION_CLASS_SIZE * NC_OPTION_CLASS3)) {
        mvLog(MVLOG_ERROR, "Option %d is invalid", option);
        return NC_INVALID_PARAMETERS;
    }
    if (option >= GRAPH_CLASS0_BASE &&
        option <= (GRAPH_CLASS0_BASE + OPTION_CLASS_SIZE)) {
        mvLog(MVLOG_ERROR, "Option %d is read only", option);
        return NC_UNAUTHORIZED;
    }
    struct _graphPrivate_t *g = graphHandle->private_data;
    GLOBAL_LOCK();
    if (isGraphPreAllocateOption(option) && g->state != NC_GRAPH_CREATED) {
        mvLog(MVLOG_ERROR,
              "This graph has already been alocated - cannot set option");
        GLOBAL_UNLOCK();
        return NC_UNAUTHORIZED;
    }
    if (!isGraphPreAllocateOption(option) && g->state == NC_GRAPH_CREATED) {
        mvLog(MVLOG_ERROR,
              "This graph hasn't been allocated - cannot set option");
        GLOBAL_UNLOCK();
        return NC_UNAUTHORIZED;
    }
    if (!isGraphPreAllocateOption(option) && findGraph(g)) {
        mvLog(MVLOG_ERROR, "This graph is corrupt or has been destroyed");
        GLOBAL_UNLOCK();
        return NC_INVALID_HANDLE;
    }
    GLOBAL_UNLOCK();
    //we check what we can at this point, later we might fail if
    //user set a class that was not permitted
    ncOptionClass_t opClass = getOptionClass(option, GRAPH_CLASS0_BASE);
    if (g->dev != NULL && opClass > g->dev->dev_attr.max_graph_opt_class) {
        mvLog(MVLOG_ERROR, "This device FW does not support NC_OPTION_CLASS%d",
              opClass);
        return NC_UNAUTHORIZED;
    }
    ncStatus_t rc;
    switch (opClass) {
    case NC_OPTION_CLASS0:
        mvLog(MVLOG_ERROR, "Class 0 options are read-only");
        rc = NC_UNAUTHORIZED; // option class 0 consists of read-only value
        break;
    case NC_OPTION_CLASS1:
        rc = setGraphOptionClass1(g, option, data, dataLength);
        break;
    default:
        mvLog(MVLOG_ERROR, "There is no such option class");
        rc = NC_INVALID_PARAMETERS;
        break;
    }
    return rc;
}

static ncStatus_t getGraphOptionClass0(struct _graphPrivate_t *g,
                                       ncGraphOption_t option,
                                       void *data, unsigned int *dataLength)
{
    if ((option == NC_RO_GRAPH_STATE ||
         option == NC_RO_GRAPH_INPUT_COUNT ||
         option == NC_RO_GRAPH_OUTPUT_COUNT ||
         option == NC_RO_GRAPH_OPTION_CLASS_LIMIT ||
         option == NC_RW_GRAPH_EXECUTORS_NUM) && *dataLength < sizeof(int)) {
        mvLog(MVLOG_ERROR,
              "data length of data (%d) is smaller that required (%zu)!\n",
              *dataLength, sizeof(int));
        *dataLength = sizeof(int);
        return NC_INVALID_DATA_LENGTH;
    }

    graphMonCommand_t cmd;
    streamPacketDesc_t* pack = 0;
    cmd.cmdClass = GRAPH_MON_CLASS_GET_CLASS0;

    switch (option) {
    case NC_RO_GRAPH_STATE:
        if (g->state == NC_GRAPH_CREATED ||
            (g->state == NC_GRAPH_ALLOCATED && !g->started)) {
            *(int *) data = g->state;
        } else {
            CHECK_HANDLE_CORRECT(g->dev);
            //it has been started we must read from graph
            cmd.cmd.optionCmd.type.c0 = CLASS0_STATE;
            cmd.cmd.optionCmd.id = g->id;
            CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&g->dev->graph_stream_m), NC_ERROR);
            if (XLinkWriteData(g->dev->graph_monitor_stream_id,
                               (const uint8_t *) &cmd, sizeof(cmd)) != 0) {
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
                return NC_ERROR;
            }

            if (XLinkReadData(g->dev->graph_monitor_stream_id, &pack) || !pack) {
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
                return NC_ERROR;
            }

            if (pack->length != sizeof(graphState_t)) {
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
                XLinkReleaseData(g->dev->graph_monitor_stream_id);
                return NC_ERROR;
            }
            int state = *(int *) pack->data;
            XLinkReleaseData(g->dev->graph_monitor_stream_id);
            if (checkGraphMonitorResponse(g->dev->graph_monitor_stream_id)) {
                CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
                return NC_ERROR;
            }
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            if (state == GRAPH_RUNNING)
                g->state = NC_GRAPH_RUNNING;
            else
                g->state = NC_GRAPH_WAITING_FOR_BUFFERS;
            *(int *) data = g->state;
        }
        *dataLength = sizeof(ncGraphState_t);
        break;
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
        cmd.cmd.optionCmd.id = g->id;
        cmd.cmd.optionCmd.type.c0 = CLASS0_TIMING_DATA;
        CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&g->dev->graph_stream_m), NC_ERROR);
        if (sendGraphMonitorRequest(g->dev->graph_monitor_stream_id, &cmd)) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return NC_ERROR;
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

        cmd.cmd.optionCmd.type.c0 = CLASS0_DEBUG_DATA;
        cmd.cmd.optionCmd.id = g->id;
        CHECK_MUTEX_SUCCESS_RC(pthread_mutex_lock(&g->dev->graph_stream_m), NC_ERROR);
        if (XLinkWriteData(g->dev->graph_monitor_stream_id, (const uint8_t *) &cmd,
             sizeof(cmd)) != 0) {
            CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
            return NC_ERROR;
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
    case NC_RO_GRAPH_NAME:
        if (*dataLength < strlen(g->name) + 1) {
            mvLog(MVLOG_ERROR,
                  "data length of output buffer (%d) is smaller that required (%zu)!\n",
                  *dataLength, strlen(g->name) + 1);
            *dataLength = strlen(g->name) + 1;
            return NC_INVALID_DATA_LENGTH;
        }
        *dataLength = strlen(g->name) + 1;
        mv_strncpy((char *) data, *dataLength, g->name, *dataLength - 1);
        break;
    case NC_RO_GRAPH_OPTION_CLASS_LIMIT:
        CHECK_HANDLE_CORRECT(g->dev);
        *(int *) data = g->dev->dev_attr.max_graph_opt_class;
        *dataLength = sizeof(int);
        break;
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
    default:
        mvLog(MVLOG_ERROR, "There is no such option in class 0");
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}

static ncStatus_t getGraphOptionClass1(struct _graphPrivate_t *g,
                                       ncGraphOption_t option,
                                       void *data, unsigned int *dataLength)
{
    switch (option) {
    case NC_RW_GRAPH_EXECUTORS_NUM:{
            int size = sizeof(int);
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
        mvLog(MVLOG_ERROR, "There is no such option in class 1");
        return NC_INVALID_PARAMETERS;
    }
    return NC_OK;
}

ncStatus_t ncGraphGetOption(struct ncGraphHandle_t * graphHandle,
                            int option, void *data, unsigned int *dataLength)
{
    CHECK_HANDLE_CORRECT(graphHandle);
    CHECK_HANDLE_CORRECT_WINFO(graphHandle->private_data, MVLOG_ERROR, "graphHandle has been destroyed");

    if (!dataLength || (*dataLength != 0 && !data)) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    if (option < GRAPH_CLASS0_BASE ||
        option > (GRAPH_CLASS0_BASE + OPTION_CLASS_SIZE * NC_OPTION_CLASS3)) {
        mvLog(MVLOG_ERROR, "Option %d is invalid", option);
        return NC_INVALID_PARAMETERS;
    }

    struct _graphPrivate_t *g = graphHandle->private_data;
    CHECK_HANDLE_CORRECT(g);

    GLOBAL_LOCK();
    if (!isGraphPreAllocateOption(option) && g->state == NC_GRAPH_CREATED) {
        mvLog(MVLOG_ERROR, "This graph hasn't been allocated");
        GLOBAL_UNLOCK();
        return NC_NOT_ALLOCATED;
    }
    ncOptionClass_t class = getOptionClass(option, GRAPH_CLASS0_BASE);
    if (g->dev != NULL && class > g->dev->dev_attr.max_graph_opt_class) {
        mvLog(MVLOG_ERROR, "This device FW does not support NC_OPTION_CLASS%d",
              class);
        return NC_UNAUTHORIZED;
    }
    GLOBAL_UNLOCK();
    ncStatus_t rc;
    switch (class) {
    case NC_OPTION_CLASS0:
        rc = getGraphOptionClass0(g, option, data, dataLength);
        break;
    case NC_OPTION_CLASS1:
        rc = getGraphOptionClass1(g, option, data, dataLength);
        break;
    default:
        mvLog(MVLOG_ERROR, "There is no such option class");
        rc = NC_INVALID_PARAMETERS;
        break;
    }
    return rc;
}

ncStatus_t ncGraphAllocateWithFifos(struct ncDeviceHandle_t * deviceHandle,
                                    struct ncGraphHandle_t * graphHandle,
                                    const void *graphBuffer,
                                    unsigned int graphBufferLength,
                                    const void *graphHeader,
                                    unsigned int graphHeaderLength,
                                    struct ncFifoHandle_t ** inFifoHandle,
                                    struct ncFifoHandle_t ** outFifoHandle)
{
    return ncGraphAllocateWithFifosEx(deviceHandle,
                                      graphHandle, graphBuffer,
                                      graphBufferLength,
                                      graphHeader,
                                      graphHeaderLength,
                                      inFifoHandle,
                                      NC_FIFO_HOST_WO, 2, NC_FIFO_FP32,
                                      outFifoHandle, NC_FIFO_HOST_RO, 2,
                                      NC_FIFO_FP32);
}

ncStatus_t ncGraphAllocateWithFifosEx(struct ncDeviceHandle_t * deviceHandle,
                                      struct ncGraphHandle_t * graphHandle,
                                      const void *graphBuffer,
                                      unsigned int graphBufferLength,
                                      const void *graphHeader,
                                      unsigned int graphHeaderLength,
                                      struct ncFifoHandle_t ** inFifoHandle,
                                      ncFifoType_t inFifoType, unsigned int inNumElem,
                                      ncFifoDataType_t inDataType,
                                      struct ncFifoHandle_t ** outFifoHandle,
                                      ncFifoType_t outFifoType, unsigned int outNumElem,
                                      ncFifoDataType_t outDataType)
{
    CHECK_HANDLE_CORRECT(deviceHandle);
    CHECK_HANDLE_CORRECT(graphHandle);
    CHECK_HANDLE_CORRECT(graphBuffer);
    CHECK_HANDLE_CORRECT(graphHeader);
    CHECK_HANDLE_CORRECT(inFifoHandle);
    CHECK_HANDLE_CORRECT(outFifoHandle);
    if ( !inNumElem || !outNumElem ) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL or Zero!");
        return NC_INVALID_PARAMETERS;
    }
    ncStatus_t rc = ncGraphAllocate(deviceHandle, graphHandle, graphBuffer, graphBufferLength, graphHeader, graphHeaderLength);
    if (rc != NC_OK)
        return rc;

    if (inFifoType == NC_FIFO_HOST_RO) {
        mvLog(MVLOG_ERROR, "input fifo cannot be read-only");
        return NC_INVALID_PARAMETERS;
    }
    if (outFifoType == NC_FIFO_HOST_WO) {
        mvLog(MVLOG_ERROR, "output fifo cannot be write-only");
        return NC_INVALID_PARAMETERS;
    }
    // Read tensor descriptors
    struct ncTensorDescriptor_t inputTensorDesc;
    struct ncTensorDescriptor_t outputTensorDesc;
    unsigned int length = sizeof(struct ncTensorDescriptor_t);
    rc = ncGraphGetOption(graphHandle,
                          NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS,
                          &inputTensorDesc, &length);
    if (rc != NC_OK) {
        return rc;
    }
    rc = ncGraphGetOption(graphHandle,
                          NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS,
                          &outputTensorDesc, &length);
    if (rc != NC_OK) {
        return rc;
    }
    rc = ncFifoCreate("fifoIn0", inFifoType, inFifoHandle);
    if (rc != NC_OK) {
        return rc;
    }
    rc = ncFifoSetOption(*inFifoHandle, NC_RW_FIFO_DATA_TYPE, &inDataType,
                         sizeof(inDataType));
    if (rc != NC_OK) {
        return rc;
    }
    rc = ncFifoAllocate(*inFifoHandle, deviceHandle, &inputTensorDesc,
                        inNumElem);
    if (rc != NC_OK) {
        return rc;
    }
    rc = ncFifoCreate("fifoOut0", outFifoType, outFifoHandle);
    if (rc != NC_OK) {
        ncFifoDestroy(inFifoHandle);
        return rc;
    }
    rc = ncFifoSetOption(*outFifoHandle, NC_RW_FIFO_DATA_TYPE, &outDataType,
                         sizeof(outDataType));
    if (rc != NC_OK) {
        ncFifoDestroy(inFifoHandle);
        ncFifoDestroy(outFifoHandle);
        return rc;
    }
    rc = ncFifoAllocate(*outFifoHandle, deviceHandle, &outputTensorDesc,
                        outNumElem);
    if (rc != NC_OK) {
        ncFifoDestroy(inFifoHandle);
        ncFifoDestroy(outFifoHandle);
        return rc;
    }
    return rc;
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
        case NC_RW_RESET_ALL:
        case NC_RW_COMMON_TIMEOUT_MSEC:
        case NC_RW_DEVICE_OPEN_TIMEOUT_MSEC:
        case NC_RW_ALLOC_GRAPH_TIMEOUT_MSEC: {
            if (dataLength < sizeof(int)) {
                mvLog(MVLOG_ERROR, "The dataLength is smaller that required %zu",
                      sizeof(int));
                return NC_INVALID_PARAMETERS;
            }
            break;
        }
        default:
            break;
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
    case NC_RW_ALLOC_GRAPH_TIMEOUT_MSEC: {
        int gTimeout = *(int *) data;
        XLinkError_t rc = XLinkSetAllocateGraphTimeOutMsec(gTimeout);
        if (rc) {
            mvLog(MVLOG_ERROR, "Set global allocate graph timeout failed, rc = %s\n", XLinkErrorToStr(rc));
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

static ncStatus_t getDeviceOptionClass0(struct _devicePrivate_t *d,
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
        d->throttle_happened = d->thermal_stats[0];
        *(int *) data = d->throttle_happened;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_STATE:
        *(int *) data = d->state;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_ALLOCATED_GRAPH_NUM:
        *(int *) data = deviceGetNumberOfGraphs(d);
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_ALLOCATED_FIFO_NUM:
        *(int *) data = deviceGetNumberOfFifos(d);
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MEMORY_SIZE:
        *(int *) data = d->dev_attr.max_memory;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MAX_FIFO_NUM:
        *(int *) data = d->dev_attr.max_fifos;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_MAX_GRAPH_NUM:
        *(int *) data = d->dev_attr.max_graphs;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_OPTION_CLASS_LIMIT:
        *(int *) data = d->dev_attr.max_device_opt_class;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_NAME:
        if (*dataLength < strlen(d->dev_addr) + 1) {
            mvLog(MVLOG_ERROR,
                  "data length of output buffer (%d) is smaller that required (%zu)!\n",
                  *dataLength, strlen(d->dev_addr) + 1);
            *dataLength = strlen(d->dev_addr) + 1;
            return NC_INVALID_DATA_LENGTH;
        }
        *dataLength = strlen(d->dev_addr) + 1;
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
    case NC_RO_DEVICE_FW_VERSION:
        *(unsigned int **) data = d->dev_attr.fw_version;
        *dataLength = sizeof(unsigned int*);
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
    case NC_RO_DEVICE_MAX_EXECUTORS_NUM:
        *(int *) data = d->dev_attr.max_executors;
        *dataLength = sizeof(int);
        break;
    case NC_RO_DEVICE_DEBUG_INFO:
        return NC_UNSUPPORTED_FEATURE;
    default:
        mvLog(MVLOG_ERROR, "No such option");
        return NC_INVALID_PARAMETERS;
    }
    return rc;
}

ncStatus_t ncDeviceSetOption(struct ncDeviceHandle_t *deviceHandle,
                             ncDeviceOption_t option,
                             const void *data, unsigned int dataLength){
    if (!deviceHandle || !data){
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }
    if (dataLength != sizeof(int) && dataLength != sizeof(void*)){
        mvLog(MVLOG_ERROR, "The dataLength must be %zu or %zu", sizeof(int), sizeof(void*));
        return NC_INVALID_PARAMETERS;
    }

    if (option < DEVICE_CLASS0_BASE ||
        option > (DEVICE_CLASS0_BASE + OPTION_CLASS_SIZE * NC_OPTION_CLASS3)) {
        mvLog(MVLOG_ERROR, "Option %d is invalid", option);
        return NC_INVALID_PARAMETERS;
    }


    ncOptionClass_t opClass = getOptionClass(option, DEVICE_CLASS0_BASE);
    if (opClass < NC_OPTION_CLASS1) {
        mvLog(MVLOG_ERROR, "Class 0 options are read-only");
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
    GLOBAL_UNLOCK();
    if (opClass > d->dev_attr.max_device_opt_class) {
        mvLog(MVLOG_ERROR, "This device FW does not support NC_OPTION_CLASS%d",
              opClass);
        return NC_UNAUTHORIZED;
    }

    return NC_INVALID_PARAMETERS;
}

//static options can be read before device is open
static int isDeviceStaticOption(int option)
{
    switch (option) {
    case NC_RO_DEVICE_NAME:
    case NC_RO_DEVICE_STATE:
    case NC_RO_DEVICE_HW_VERSION:
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

    if (option < DEVICE_CLASS0_BASE ||
        option > (DEVICE_CLASS0_BASE + OPTION_CLASS_SIZE * NC_OPTION_CLASS3)) {
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

    ncOptionClass_t opClass = getOptionClass(option, DEVICE_CLASS0_BASE);
    if (!isDeviceStaticOption(option)) {
        if (findDevice(d)) {
            mvLog(MVLOG_ERROR,
                  "This device handle is corrupt or has been destroyed");
            GLOBAL_UNLOCK();
            return NC_INVALID_HANDLE;
        }

        if (d->dev_attr.max_device_opt_class < opClass) {
            mvLog(MVLOG_ERROR,
                  "This device FW does not support NC_OPTION_CLASS%d", opClass);
            GLOBAL_UNLOCK();
            return NC_UNAUTHORIZED;
        }
    }

    switch (opClass) {
    case NC_OPTION_CLASS0:
        rc = getDeviceOptionClass0(d, option, data, dataLength);
        break;
    default:
        rc = NC_INVALID_PARAMETERS;
        break;
    }

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
    handle->host_tensor_desc_set = 0;
    memset(&handle->host_tensor_desc, 0, sizeof(struct ncTensorDescriptor_t));
    handle->host_tensor_desc.dataType = NC_FIFO_FP16; //default app data type is FP16
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

void getStrides(ncFifoLayout_t layout, struct ncTensorDescriptor_t* desc,
    ncFifoDataType_t dataType) {
    int baseStride = dataType == NC_FIFO_FP16 ? FP16_DATA_SIZE : sizeof(float);
    switch (layout) {
        case NC_FIFO_HWC:
            desc->cStride = baseStride;
            desc->wStride = desc->cStride * desc->c;
            desc->hStride = desc->wStride * desc->w;
            break;
        case NC_FIFO_CHW:
            desc->wStride = baseStride;
            desc->hStride = desc->wStride * desc->w;
            desc->cStride = desc->hStride * desc->h;
            break;
        case NC_FIFO_HCW:
            desc->wStride = baseStride;
            desc->cStride = desc->wStride * desc->w;
            desc->hStride = desc->cStride * desc->c;
            break;
        case NC_FIFO_CWH:
            desc->hStride = baseStride;
            desc->wStride = desc->hStride * desc->h;
            desc->cStride = desc->wStride * desc->w;
            break;
        case NC_FIFO_WCH:
            desc->hStride = baseStride;
            desc->cStride = desc->hStride * desc->h;
            desc->wStride = desc->cStride * desc->c;
            break;
        case NC_FIFO_WHC:
            desc->cStride = baseStride;
            desc->hStride = desc->cStride * desc->c;
            desc->wStride = desc->hStride * desc->h;
            break;
        default:
            break;
    }
}

static unsigned int getTotalSize(struct ncTensorDescriptor_t* desc) {
    unsigned int maxStride;
    unsigned int maxDim;

    if (desc->wStride == desc->hStride &&
        desc->wStride == desc->cStride) {
        maxDim = MAX(desc->w, desc->h);
        maxDim = MAX(maxDim, desc->c);
        maxStride = desc->wStride;
    } else if (desc->wStride >= desc->hStride &&
               desc->wStride >= desc->cStride) {
        maxStride = desc->wStride;
        maxDim = desc->w;
        if (desc->wStride == desc->hStride)
            maxDim = MAX(desc->w, desc->h);
        else if (desc->wStride == desc->cStride)
            maxDim = MAX(desc->w, desc->c);
    } else if (desc->hStride >= desc->wStride &&
               desc->hStride >= desc->cStride) {
        maxStride = desc->hStride;
        maxDim = desc->h;
        if (desc->hStride == desc->wStride)
            maxDim = MAX(desc->h, desc->w);
        else if (desc->hStride == desc->cStride)
            maxDim = MAX(desc->h, desc->c);
    } else {
        maxStride = desc->cStride;
        maxDim = desc->c;
        if (desc->cStride == desc->wStride)
            maxDim = MAX(desc->c, desc->w);
        else if (desc->cStride == desc->hStride)
            maxDim = MAX(desc->c, desc->h);
    }
    return desc->n * maxStride * maxDim;
}
static unsigned int getElementSize(struct _fifoPrivate_t * handle) {
    return handle->host_tensor_desc.totalSize;
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

    handle->graph_tensor_desc = *tensor_desc;
    handle->host_tensor_desc = *tensor_desc;
    handle->graphLayout = getLayout(tensor_desc);
    handle->user_param_in = NULL;
    handle->user_param_out = NULL;
    handle->num_elements = numElem;
    handle->consumers_remaining = handle->consumer_cnt; //default consumers
    handle->dev = d;
    handle->next = NULL;

    handle->datasize = getElementSize(handle);

    if (d->fifos)
        handle->next = d->fifos;
    d->fifos = handle;

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_BUFFER_CMD;
    cmd.cmd.buffCmd.type = BUFFER_ALLOCATE_CMD;
    struct tensorDescriptor_t privateDesc;
    privateDesc.c = tensor_desc->c;
    privateDesc.n = tensor_desc->n;
    privateDesc.h = tensor_desc->h;
    privateDesc.w = tensor_desc->w;
    // should be removiedd: #-17902
    privateDesc.totalSize = tensor_desc->totalSize;
    privateDesc.widthStride = tensor_desc->wStride;
    privateDesc.heightStride = tensor_desc->hStride;
    privateDesc.channelsStride = tensor_desc->cStride;

    cmd.cmd.buffCmd.desc  = privateDesc;
    cmd.cmd.buffCmd.elemCnt = numElem;
    snprintf(cmd.cmd.buffCmd.name, MAX_STREAM_NAME_LENGTH, "FIFO%d", handle->id);
    cmd.cmd.buffCmd.name[NC_MAX_NAME_SIZE - 1] = 0;
    cmd.cmd.buffCmd.id = handle->id;

    uint32_t writeSize;
    if (fifoWriteAccess(handle)) {
        writeSize = tensor_desc->totalSize * numElem;
        cmd.cmd.buffCmd.writeChannel = 1;
    } else {
        cmd.cmd.buffCmd.writeChannel = 0;
        writeSize = 8; // no write permission on this buffer, so we shouldn't bother allocating buffer on the device
    }
    if (fifoReadAccess(handle)) {
        cmd.cmd.buffCmd.readChannel = 1;
    } else {
        cmd.cmd.buffCmd.readChannel = 0;
    }
    streamId_t streamId = XLinkOpenStream(d->xlink->linkId, cmd.cmd.buffCmd.name, writeSize);

    char out_msg[NC_MAX_NAME_SIZE * 2];
    snprintf(out_msg, NC_MAX_NAME_SIZE * 2, "%s %s", "can't open stream: ", cmd.cmd.buffCmd.name);

    CHECK_STREAM_ID(streamId, {
            handle->state = NC_FIFO_FAILED;
            handle->dev->state = NC_DEVICE_FAILED;
        }, out_msg);

    handle->streamId = streamId;
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));

    if (sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        mvLog(MVLOG_ERROR, "can't send command\n");
        return NC_ERROR;
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

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_BUFFER_CMD;
    cmd.cmd.buffCmd.type = BUFFER_DEALLOCATE_CMD;
    cmd.cmd.buffCmd.id = handle->id;

    struct _devicePrivate_t *d = handle->dev;
    CHECK_MUTEX_SUCCESS(pthread_mutex_lock(&d->graph_stream_m));
    if (sendGraphMonitorRequest(d->graph_monitor_stream_id, &cmd)) {
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&d->graph_stream_m));
        mvLog(MVLOG_WARN, "can't send command\n");
        return NC_ERROR;
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
    struct ncTensorDescriptor_t * inputDesc = &handle->graph_tensor_desc;

    int rc;
    // Convert fp32 to fp16 and/or input layout
    ncFifoLayout_t layout = getLayout(inputDesc);
    ncFifoLayout_t host_layout = getLayout(&handle->host_tensor_desc);
    if (handle->host_tensor_desc.dataType == NC_FIFO_FP32 || layout != host_layout) {
        mvLog(MVLOG_ERROR,
              "This version of mvnc does not support converting layout and precision on the host\n");

        return NC_UNSUPPORTED_FEATURE;
    } else {
        rc = XLinkWriteData(handle->streamId, inputTensor, *inputTensorLength);
    }
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

    if (*outputDataLen < handle->datasize) {
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
        // Convert fp16 to fp32 and/or layout
        struct ncTensorDescriptor_t * fifoDesc = &handle->graph_tensor_desc;
        ncFifoLayout_t layout = getLayout(fifoDesc);
        ncFifoLayout_t host_layout = getLayout(&handle->host_tensor_desc);

        if (handle->host_tensor_desc.dataType == NC_FIFO_FP32 ||
            layout != host_layout) {
            mvLog(MVLOG_ERROR,
                  "This version of mvnc does not support converting layout and precision on the host\n");

            return NC_UNSUPPORTED_FEATURE;
        } else {
            mvnc_memcpy(outputData, *outputDataLen, packet->data, packet->length);
        }
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

ncStatus_t ncFifoRemoveElem(struct ncFifoHandle_t* fifoHandle) {
    CHECK_HANDLE_CORRECT(fifoHandle)


    return NC_UNSUPPORTED_FEATURE;
}

ncStatus_t ncFifoSetOption(struct ncFifoHandle_t * fifoHandle, int option,
                           const void *data, unsigned int dataLength)
{
    CHECK_HANDLE_CORRECT(fifoHandle);
    CHECK_HANDLE_CORRECT_RC(data, NC_INVALID_PARAMETERS);
    CHECK_HANDLE_CORRECT_WINFO(fifoHandle->private_data, MVLOG_ERROR,
            "fifo handle is corrupt or has been destroyed");

    struct _fifoPrivate_t *f = (struct _fifoPrivate_t *) fifoHandle->private_data;
    if (f->state != NC_FIFO_CREATED && option != NC_RW_FIFO_HOST_TENSOR_DESCRIPTOR) {
        mvLog(MVLOG_ERROR, "cannot set Fifo options after allocation");
        return NC_UNAUTHORIZED;
    }

    switch (option) {
    case NC_RW_FIFO_TYPE:{
            unsigned int size = sizeof(ncFifoType_t);
            if (dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      dataLength, size);
                return NC_INVALID_DATA_LENGTH;
            }
            int tempType = *(ncFifoType_t *) data;
            if (tempType != NC_FIFO_HOST_WO && tempType != NC_FIFO_HOST_RO) {
                 mvLog(MVLOG_ERROR,
                      "Type value set (%d) is invalid!\n",
                      tempType);
                return NC_INVALID_PARAMETERS;
            }
            f->type = tempType;
            break;
        }
    case NC_RW_FIFO_CONSUMER_COUNT:{
            unsigned int size = sizeof(int);
            if (dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      dataLength, size);
                return NC_INVALID_DATA_LENGTH;
            }
            f->consumer_cnt = *(int *) data;
            break;
        }
    case NC_RW_FIFO_DATA_TYPE:{
            unsigned int size = sizeof(ncFifoDataType_t);
            if (dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      dataLength, size);
                return NC_INVALID_DATA_LENGTH;
            }
            int tempDType = *(int *) data;
            if (tempDType != NC_FIFO_FP16 && tempDType != NC_FIFO_FP32) {
                mvLog(MVLOG_ERROR,
                      "dataType value set (%d) is invalid!\n",
                      tempDType);
                return NC_INVALID_PARAMETERS;
            }
            f->host_tensor_desc.dataType = tempDType;
            break;
        }
    case NC_RW_FIFO_HOST_TENSOR_DESCRIPTOR:{
            unsigned int size = sizeof(struct ncTensorDescriptor_t);
            if (dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      dataLength, size);
                return NC_INVALID_DATA_LENGTH;
            }

            int expected_total_size = getTotalSize((struct ncTensorDescriptor_t *) data);
            if (expected_total_size != ((struct ncTensorDescriptor_t *) data)->totalSize) {
                mvLog(MVLOG_ERROR,
                      "totalSize in host tensor descriptor (%d) doesn't match expeected totalSize (%d)!\n",
                      ((struct ncTensorDescriptor_t *) data)->totalSize, expected_total_size);
                return NC_INVALID_PARAMETERS;
            }
            if (f->state == NC_FIFO_ALLOCATED) {
                struct ncTensorDescriptor_t* temp = (struct ncTensorDescriptor_t*) data;
                if (temp->w != f->graph_tensor_desc.w ||
                    temp->h != f->graph_tensor_desc.h ||
                    temp->c != f->graph_tensor_desc.c ||
                    temp->n != f->graph_tensor_desc.n)
                {
                    mvLog(MVLOG_ERROR, "trying to set host tensor decriptor to a shape that doesn't match graph tensor descriptor shape!\n");
                    return NC_INVALID_PARAMETERS;
                }
            }

            f->host_tensor_desc = *(struct ncTensorDescriptor_t *) data;
            f->host_tensor_desc_set = 1;
            f->datasize = getElementSize(f);

            break;
        }
    case NC_RW_FIFO_DONT_BLOCK:
        return NC_UNSUPPORTED_FEATURE;
        break;
    case NC_RO_FIFO_CAPACITY:
    case NC_RO_FIFO_READ_FILL_LEVEL:
    case NC_RO_FIFO_WRITE_FILL_LEVEL:
    case NC_RO_FIFO_GRAPH_TENSOR_DESCRIPTOR:
    case NC_RO_FIFO_STATE:
    case NC_RO_FIFO_ELEMENT_DATA_SIZE:
        return NC_UNAUTHORIZED;
        break;
    default:
        return NC_INVALID_PARAMETERS;
        break;
    }
    return NC_OK;
}

ncStatus_t ncFifoGetOption(struct ncFifoHandle_t * fifoHandle, int option,
                           void *data, unsigned int *dataLength)
{
    CHECK_HANDLE_CORRECT(fifoHandle);
    CHECK_HANDLE_CORRECT_WINFO(fifoHandle->private_data, MVLOG_ERROR,
            "Fifo is corrupt or has been destroyed")

    if (!dataLength || (*dataLength != 0 && !data)) {
        mvLog(MVLOG_ERROR, "Some of the parameters are NULL");
        return NC_INVALID_PARAMETERS;
    }

    if (fifoHandle->private_data->state == NC_FIFO_CREATED &&
        option != NC_RO_FIFO_STATE && option != NC_RW_FIFO_DATA_TYPE &&
        option != NC_RW_FIFO_DONT_BLOCK && option != NC_RW_FIFO_CONSUMER_COUNT
        && option != NC_RO_FIFO_NAME && option != NC_RW_FIFO_HOST_TENSOR_DESCRIPTOR) {
        mvLog(MVLOG_ERROR,
              "Fifo hasn't been allocated, cannot read those options");
        return NC_NOT_ALLOCATED;
    }
    switch (option) {
    case NC_RW_FIFO_CONSUMER_COUNT:
    case NC_RO_FIFO_CAPACITY:
    case NC_RO_FIFO_READ_FILL_LEVEL:
    case NC_RO_FIFO_WRITE_FILL_LEVEL:
    case NC_RO_FIFO_STATE:
    case NC_RO_FIFO_ELEMENT_DATA_SIZE:
        {
            unsigned int size = sizeof(int);
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            break;
        }
    default:
        break;
    }

    switch (option) {
    case NC_RW_FIFO_TYPE:{
            unsigned int size = sizeof(ncFifoType_t);
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            *(ncFifoType_t *) data = fifoHandle->private_data->type;
            *dataLength = sizeof(fifoHandle->private_data->type);
            break;
        }
    case NC_RW_FIFO_CONSUMER_COUNT:
        *(int *) data = fifoHandle->private_data->consumer_cnt;
        *dataLength = sizeof(fifoHandle->private_data->consumer_cnt);
        break;
    case NC_RO_FIFO_ELEMENT_DATA_SIZE:
        *(int *) data = getElementSize(fifoHandle->private_data);
        *dataLength = sizeof(fifoHandle->private_data->datasize);
        break;
    case NC_RW_FIFO_DATA_TYPE:
        {
            unsigned int size = sizeof(ncFifoDataType_t);
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            *(int *) data = fifoHandle->private_data->host_tensor_desc.dataType;
            *dataLength = sizeof(fifoHandle->private_data->host_tensor_desc.dataType);
            break;
        }
    case NC_RO_FIFO_CAPACITY:
        *(int *) data = fifoHandle->private_data->num_elements;
        *dataLength = sizeof(fifoHandle->private_data->num_elements);
        break;
    case NC_RO_FIFO_GRAPH_TENSOR_DESCRIPTOR:
        {
            unsigned int size = sizeof(struct ncTensorDescriptor_t);
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            if (fifoHandle->private_data->state != NC_FIFO_ALLOCATED)
                return NC_UNAUTHORIZED; // before allocation, tensor_desc is NULL
            *(struct ncTensorDescriptor_t *) data =
                fifoHandle->private_data->graph_tensor_desc;
            *dataLength = sizeof(fifoHandle->private_data->graph_tensor_desc);
            break;
        }
    case NC_RW_FIFO_HOST_TENSOR_DESCRIPTOR:
        {
            unsigned int size = sizeof(struct ncTensorDescriptor_t);
            if (*dataLength < size) {
                mvLog(MVLOG_ERROR,
                      "data length of output buffer (%d) is smaller that required (%d)!\n",
                      *dataLength, size);
                *dataLength = size;
                return NC_INVALID_DATA_LENGTH;
            }
            if (fifoHandle->private_data->state != NC_FIFO_ALLOCATED &&
                fifoHandle->private_data->host_tensor_desc_set == 0) {
                mvLog(MVLOG_ERROR,
                      "option NC_RW_FIFO_HOST_TENSOR_DESCRIPTOR cannot be read before it has been set or before Fifo has been allocated");
                return NC_UNAUTHORIZED;
            }
            *(struct ncTensorDescriptor_t *) data =
                fifoHandle->private_data->host_tensor_desc;
            *dataLength = sizeof(fifoHandle->private_data->host_tensor_desc);
            break;
        }
    case NC_RO_FIFO_READ_FILL_LEVEL:
        {
            struct _fifoPrivate_t *fi = fifoHandle->private_data;
            if (!fifoReadAccess(fi))
                return NC_UNAUTHORIZED;

            *dataLength = sizeof(int);
            if (fi->state != NC_FIFO_ALLOCATED) {
                *(int *) data = 0;
                break;
            }
            int fillLevel;
            if (XLinkGetFillLevel(fi->streamId, 0, &fillLevel) == X_LINK_SUCCESS) {
                *(int *) data = (fillLevel / fi->graph_tensor_desc.totalSize);
            } else {
                return NC_UNAUTHORIZED;
            }

            break;
        }
    case NC_RO_FIFO_WRITE_FILL_LEVEL:
        {
            struct _fifoPrivate_t *fi = fifoHandle->private_data;
            if (!fifoWriteAccess(fi))
                return NC_UNAUTHORIZED;

            *dataLength = sizeof(int);
            if (fi->state != NC_FIFO_ALLOCATED) {
                *(int *) data = 0;
                break;
            }
            int fillLevel;
            if (XLinkGetFillLevel(fi->streamId, 1, &fillLevel) == X_LINK_SUCCESS) {
                *(int *) data = (fillLevel / fi->graph_tensor_desc.totalSize);
            } else {
                return NC_ERROR;
            }

            break;
        }
    case NC_RW_FIFO_DONT_BLOCK:
        return NC_UNSUPPORTED_FEATURE; //TODO: XLink support for this (fill level may be enough for it)
        break;
    case NC_RO_FIFO_STATE:
        *(int *) data = fifoHandle->private_data->state;
        *dataLength = sizeof(int);
        break;
    case NC_RO_FIFO_NAME:
        if (*dataLength < strlen(fifoHandle->private_data->name) + 1) {
            mvLog(MVLOG_ERROR,
                  "data length of output buffer (%d) is smaller that required (%zu)!\n",
                  *dataLength, strlen(fifoHandle->private_data->name) + 1);
            *dataLength = strlen(fifoHandle->private_data->name) + 1;
            return NC_INVALID_DATA_LENGTH;
        }
        *dataLength = strlen(fifoHandle->private_data->name) + 1;
        mv_strncpy((char *) data, *dataLength, fifoHandle->private_data->name, *dataLength - 1);
        break;
    default:
        return NC_INVALID_PARAMETERS;
        break;
    }
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

    graphMonCommand_t cmd;
    cmd.cmdClass = GRAPH_MON_CLASS_GRAPH_CMD;
    cmd.cmd.graphCmd.type = GRAPH_TRIGGER_CMD;
    cmd.cmd.graphCmd.id = g->id;
    cmd.cmd.graphCmd.buffId1 = fi->id;
    cmd.cmd.graphCmd.buffId2 = fo->id;

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

    if(sendGraphMonitorRequest(g->dev->graph_monitor_stream_id, &cmd)) {
        mvLog(MVLOG_ERROR, "Can't send trigger request");
        g->dev->state = NC_DEVICE_FAILED;
        CHECK_MUTEX_SUCCESS(pthread_mutex_unlock(&g->dev->graph_stream_m));
        return NC_ERROR;
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
