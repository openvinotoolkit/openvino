// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __NC_H_INCLUDED__
#define __NC_H_INCLUDED__

#ifdef __cplusplus
extern "C"
{
#endif

#include "watchdog/watchdog.h"

#define NC_THERMAL_BUFFER_SIZE 100
#define NC_DEBUG_BUFFER_SIZE   120
#define NC_MAX_DEVICES         (32)
#define NC_MAX_NAME_SIZE       (64)

#ifndef NOMINMAX
#define NOMINMAX
#endif
#define MVNC_EXPORT_API

typedef enum {
    NC_OK = 0,
    NC_BUSY = -1,                     // Device is busy, retry later
    NC_ERROR = -2,                    // Error communicating with the device
    NC_OUT_OF_MEMORY = -3,            // Out of memory
    NC_DEVICE_NOT_FOUND = -4,         // No device at the given index or name
    NC_INVALID_PARAMETERS = -5,       // At least one of the given parameters is wrong
    NC_TIMEOUT = -6,                  // Timeout in the communication with the device
    NC_MVCMD_NOT_FOUND = -7,          // The file to boot Myriad was not found
    NC_NOT_ALLOCATED = -8,            // The graph or device has been closed during the operation
    NC_UNAUTHORIZED = -9,             // Unauthorized operation
    NC_UNSUPPORTED_GRAPH_FILE = -10,  // The graph file version is not supported
    NC_UNSUPPORTED_CONFIGURATION_FILE = -11, // The configuration file version is not supported
    NC_UNSUPPORTED_FEATURE = -12,     // Not supported by this FW version
    NC_MYRIAD_ERROR = -13,            // An error has been reported by the device
                                      // use  NC_DEVICE_DEBUG_INFO or NC_GRAPH_DEBUG_INFO
    NC_INVALID_DATA_LENGTH = -14,      // invalid data length has been passed when get/set option
    NC_INVALID_HANDLE = -15           // handle to object that is invalid
} ncStatus_t;

typedef enum {
    NC_LOG_DEBUG = 0,   // debug and above (full verbosity)
    NC_LOG_INFO,        // info and above
    NC_LOG_WARN,        // warning and above
    NC_LOG_ERROR,       // errors and above
    NC_LOG_FATAL,       // fatal only
} ncLogLevel_t;

typedef enum {
    NC_RW_LOG_LEVEL = 0,    // Log level, int, default NC_LOG_WARN
    NC_RO_API_VERSION = 1,  // retruns API Version. array of unsigned int of size 4
                            //major.minor.hotfix.rc
    NC_RW_COMMON_TIMEOUT_MSEC = 2,
    NC_RW_DEVICE_OPEN_TIMEOUT_MSEC = 3,
    NC_RW_RESET_ALL = 9000,     // resetAll on initialize
} ncGlobalOption_t;

typedef enum {
    NC_RO_GRAPH_TIME_TAKEN = 1001,      // Return time taken for last inference (float *)
    NC_RO_GRAPH_INPUT_COUNT = 1002,     // Returns number of inputs, size of array returned
                                        // by NC_RO_INPUT_TENSOR_DESCRIPTORS, int
    NC_RO_GRAPH_OUTPUT_COUNT = 1003,    // Returns number of outputs, size of array returned
                                        // by NC_RO_OUTPUT_TENSOR_DESCRIPTORS,int
    NC_RO_GRAPH_INPUT_TENSOR_DESCRIPTORS = 1004,  // Return a tensorDescriptor pointer array
                                            // which describes the graph inputs in order.
                                            // Can be used for fifo creation.
                                            // The length of the array can be retrieved
                                            // using the NC_RO_INPUT_COUNT option

    NC_RO_GRAPH_OUTPUT_TENSOR_DESCRIPTORS = 1005, // Return a tensorDescriptor pointer
                                            // array which describes the graph
                                            // outputs in order. Can be used for
                                            // fifo creation. The length of the
                                            // array can be retrieved using the
                                            // NC_RO_OUTPUT_COUNT option

    NC_RO_GRAPH_DEBUG_INFO = 1006,          // Return debug info, string
    NC_RO_GRAPH_VERSION = 1009,             // returns graph version, string
    NC_RO_GRAPH_TIME_TAKEN_ARRAY_SIZE = 1011, // Return size of array for time taken option, int
    NC_RW_GRAPH_EXECUTORS_NUM = 1110,
} ncGraphOption_t;

typedef enum {
    NC_DEVICE_OPENED = 0,
    NC_DEVICE_CLOSED = 1,
    NC_DEVICE_FAILED = 2,
    NC_DEVICE_RESETED = 3,
} ncDeviceState_t;

typedef enum {
    NC_GRAPH_CREATED = 0,
    NC_GRAPH_ALLOCATED = 1,
    NC_GRAPH_WAITING_FOR_BUFFERS = 2,
    NC_GRAPH_RUNNING = 3,
    NC_GRAPH_DEALLOCATED = 4,
} ncGraphState_t;

typedef enum {
    NC_FIFO_CREATED = 0,
    NC_FIFO_ALLOCATED = 1,
    NC_FIFO_DESTROYED = 2,
    NC_FIFO_FAILED = 3,
    NC_FIFO_DEALLOCATED = 4
} ncFifoState_t;

typedef enum {
    NC_MA2450 = 0,
    NC_MA2480 = 1,
} ncDeviceHwVersion_t;

typedef enum {
    NC_RO_DEVICE_THERMAL_STATS = 2000,          // Return temperatures, float *, not for general use
    NC_RO_DEVICE_THERMAL_THROTTLING_LEVEL = 2001,   // 1=TEMP_LIM_LOWER reached, 2=TEMP_LIM_HIGHER reached
    NC_RO_DEVICE_CURRENT_MEMORY_USED = 2003,    // Returns current device memory usage
    NC_RO_DEVICE_MEMORY_SIZE = 2004,            // Returns device memory size
    NC_RO_DEVICE_MAX_GRAPH_NUM = 2007,          // return the maximum number of graphs supported
    NC_RO_DEVICE_NAME = 2013,                   // returns device name as generated internally
    NC_RO_DEVICE_PLATFORM = 2017,               // returns device platform (MyriadX, Myriad2)
    NC_RO_DEVICE_PROTOCOL = 2018,               // returns device protocol (USB, PCIe)
    NC_RW_DEVICE_POWER_CONFIG = 2100,           // writes config for the power manager to device
    NC_RW_DEVICE_POWER_CONFIG_RESET = 2101,     // resets power manager config on device
    NC_RW_ENABLE_ASYNC_DMA = 2102               // enable/disable asynchronous DMA on device
} ncDeviceOption_t;

typedef enum {
    NC_ANY_PROTOCOL = 0,
    NC_USB,
    NC_PCIE,
} ncDeviceProtocol_t;

typedef struct _devicePrivate_t devicePrivate_t;
typedef struct _graphPrivate_t graphPrivate_t;
typedef struct _fifoPrivate_t fifoPrivate_t;
typedef struct _ncTensorDescriptorPrivate_t ncTensorDescriptorPrivate_t;

struct ncFifoHandle_t {
    // keep place for public data here
    fifoPrivate_t* private_data;
};

struct ncGraphHandle_t {
    // keep place for public data here
    graphPrivate_t* private_data;
};

struct ncDeviceHandle_t {
    // keep place for public data here
    devicePrivate_t* private_data;
};

struct ncDeviceDescr_t {
    ncDeviceProtocol_t protocol;
    char name[NC_MAX_NAME_SIZE];
};

typedef struct ncDeviceOpenParams {
    WatchdogHndl_t* watchdogHndl;
    int watchdogInterval;
    char memoryType;
    const char* customFirmwareDirectory;
} ncDeviceOpenParams_t;

typedef enum {
    NC_FIFO_HOST_RO = 0, // fifo can be read through the API but can not be
                         // written ( graphs can read and write data )
    NC_FIFO_HOST_WO = 1, // fifo can be written through the API but can not be
                         // read (graphs can read but can not write)
} ncFifoType_t;

struct ncTensorDescriptor_t {
    unsigned int n;         // batch size, currently only 1 is supported
    unsigned int c;         // number of channels
    unsigned int w;         // width
    unsigned int h;         // height
    unsigned int totalSize; // Total size of the data in tensor = largest stride* dim size
};

// Global
MVNC_EXPORT_API ncStatus_t ncGlobalSetOption(ncGlobalOption_t option, const void *data,
                             unsigned int dataLength);
MVNC_EXPORT_API ncStatus_t ncGlobalGetOption(ncGlobalOption_t option, void *data, unsigned int *dataLength);

// Device
MVNC_EXPORT_API ncStatus_t ncDeviceSetOption(struct ncDeviceHandle_t *deviceHandle,
                             ncDeviceOption_t option, const void *data,
                             unsigned int dataLength);
MVNC_EXPORT_API ncStatus_t ncDeviceGetOption(struct ncDeviceHandle_t *deviceHandle,
                             ncDeviceOption_t option, void *data, unsigned int *dataLength);

/**
 * @brief Sets wait time for successful connection to the device
 * @param deviceConnectTimeoutSec timeout for new connections in seconds.
 *                                  Should be non-negative.
 */
MVNC_EXPORT_API ncStatus_t ncSetDeviceConnectTimeout(int deviceConnectTimeoutSec);

/**
 * @brief Create handle and open any free device
 * @param in_ncDeviceDesc a set of parameters that the device must comply with
 * @param watchdogInterval Time interval to ping device in milliseconds. 0 to disable watchdog.
 * @param customFirmwareDirectory Custom path to directory with firmware.
 *          If NULL or empty, default path searching behavior will be used.
 */
MVNC_EXPORT_API ncStatus_t ncDeviceOpen(struct ncDeviceHandle_t **deviceHandlePtr,
    struct ncDeviceDescr_t in_ncDeviceDesc, ncDeviceOpenParams_t deviceOpenParams);

/**
 * @brief Returns a description of all available devices in the system
 * @param deviceDescrPtr is pre-allocated array to get names of available devices
 * @param maxDevices size of deviceDescrPtr
 * @param out_countDevices count of available devices
 */
MVNC_EXPORT_API ncStatus_t ncAvailableDevices(struct ncDeviceDescr_t *deviceDescrPtr,
                                              int maxDevices, int* out_countDevices);

/**
 * @brief Close device and destroy handler
 */
MVNC_EXPORT_API ncStatus_t ncDeviceClose(struct ncDeviceHandle_t **deviceHandle, WatchdogHndl_t* watchdogHndl);

// Graph
MVNC_EXPORT_API ncStatus_t ncGraphCreate(const char* name, struct ncGraphHandle_t **graphHandle);
MVNC_EXPORT_API ncStatus_t ncGraphAllocate(struct ncDeviceHandle_t *deviceHandle,
                           struct ncGraphHandle_t *graphHandle,
                           const void *graphBuffer, unsigned int graphBufferLength,
                           const void *graphHeader, unsigned int graphHeaderLength);
MVNC_EXPORT_API ncStatus_t ncGraphDestroy(struct ncGraphHandle_t **graphHandle);
MVNC_EXPORT_API ncStatus_t ncGraphSetOption(struct ncGraphHandle_t *graphHandle,
                                            ncGraphOption_t option, const void *data, unsigned int dataLength);
MVNC_EXPORT_API ncStatus_t ncGraphGetOption(struct ncGraphHandle_t *graphHandle,
                                            ncGraphOption_t option, void *data, unsigned int *dataLength);
MVNC_EXPORT_API ncStatus_t ncGraphQueueInference(struct ncGraphHandle_t *graphHandle,
                            struct ncFifoHandle_t** fifoIn, unsigned int inFifoCount,
                            struct ncFifoHandle_t** fifoOut, unsigned int outFifoCount);

// Fifo
MVNC_EXPORT_API ncStatus_t ncFifoCreate(const char *name, ncFifoType_t type,
                        struct ncFifoHandle_t **fifoHandle);
MVNC_EXPORT_API ncStatus_t ncFifoAllocate(struct ncFifoHandle_t* fifoHandle,
                        struct ncDeviceHandle_t* device,
                        struct ncTensorDescriptor_t* tensorDesc,
                        unsigned int numElem);

MVNC_EXPORT_API ncStatus_t ncFifoDestroy(struct ncFifoHandle_t** fifoHandle);
MVNC_EXPORT_API ncStatus_t ncFifoWriteElem(struct ncFifoHandle_t* fifoHandle, const void *inputTensor,
                        unsigned int * inputTensorLength, void *userParam);
MVNC_EXPORT_API ncStatus_t ncFifoReadElem(struct ncFifoHandle_t* fifoHandle, void *outputData,
                        unsigned int* outputDataLen, void **userParam);

//Helper functions
MVNC_EXPORT_API ncStatus_t ncGraphQueueInferenceWithFifoElem(struct ncGraphHandle_t *graphHandle,
                                                             struct ncFifoHandle_t* fifoIn,
                                                             struct ncFifoHandle_t* fifoOut, const void *inputTensor,
                                                             unsigned int * inputTensorLength, void *userParam);

#ifdef __cplusplus
}
#endif

#endif
