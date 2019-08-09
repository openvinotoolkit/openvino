// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef __NC_H_INCLUDED__
#define __NC_H_INCLUDED__

#ifdef __cplusplus
extern "C"
{
#endif

#define NC_THERMAL_BUFFER_SIZE 100
#define NC_DEBUG_BUFFER_SIZE   120
#define NC_MAX_DEVICES         (32)
#define NC_MAX_NAME_SIZE       (28)

#define NOMINMAX
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
    NC_RW_ALLOC_GRAPH_TIMEOUT_MSEC = 4,
    NC_RW_RESET_ALL = 9000,     // resetAll on initialize
} ncGlobalOption_t;

typedef enum {
    NC_RO_GRAPH_STATE = 1000,           // Returns graph state: CREATED, ALLOCATED, WAITING_FOR_BUFFERS, RUNNING, DESTROYED
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
    NC_RO_GRAPH_NAME = 1007,                // Returns name of the graph, string
    NC_RO_GRAPH_OPTION_CLASS_LIMIT = 1008,  // return the highest option class supported
    NC_RO_GRAPH_VERSION = 1009,             // returns graph version, string
    NC_RO_GRAPH_TIME_TAKEN_ARRAY_SIZE = 1011, // Return size of array for time taken option, int
    NC_RO_GRAPH_BATCH_SIZE = 1012,           // returns batch size of loaded graph
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
    NC_RO_DEVICE_STATE = 2002,                  // Returns device state: CREATED, OPENED, CLOSED, DESTROYED
    NC_RO_DEVICE_CURRENT_MEMORY_USED = 2003,    // Returns current device memory usage
    NC_RO_DEVICE_MEMORY_SIZE = 2004,            // Returns device memory size
    NC_RO_DEVICE_MAX_FIFO_NUM = 2005,           // return the maximum number of fifos supported
    NC_RO_DEVICE_ALLOCATED_FIFO_NUM = 2006,     // return the number of currently allocated fifos
    NC_RO_DEVICE_MAX_GRAPH_NUM = 2007,          // return the maximum number of graphs supported
    NC_RO_DEVICE_ALLOCATED_GRAPH_NUM = 2008,    //  return the number of currently allocated graphs
    NC_RO_DEVICE_OPTION_CLASS_LIMIT = 2009,     //  return the highest option class supported
    NC_RO_DEVICE_FW_VERSION = 2010,             // return device firmware version, array of unsigned int of size 4
                                                //major.minor.hwtype.buildnumber
    NC_RO_DEVICE_DEBUG_INFO = 2011,             // Return debug info, string, not supported yet
    NC_RO_DEVICE_MVTENSOR_VERSION = 2012,       // returns mv tensor version, array of unsigned int of size 2
                                                //major.minor
    NC_RO_DEVICE_NAME = 2013,                   // returns device name as generated internally
    NC_RO_DEVICE_MAX_EXECUTORS_NUM = 2014,      //Maximum number of executers per graph
    NC_RO_DEVICE_HW_VERSION = 2015,             //returns HW Version, enum
    NC_RO_DEVICE_ID = 2016,                     // returns device id
    NC_RO_DEVICE_PLATFORM = 2017,               // returns device platform (MyriadX, Myriad2)
    NC_RO_DEVICE_PROTOCOL = 2018,               // returns device protocol (USB, PCIe)
} ncDeviceOption_t;

typedef enum {
    NC_ANY_PLATFORM = 0,
    NC_MYRIAD_2 = 2450,
    NC_MYRIAD_X = 2480,
} ncDevicePlatform_t;

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
    ncDevicePlatform_t platform;
    char name[NC_MAX_NAME_SIZE];
};

typedef enum {
    NC_FIFO_HOST_RO = 0, // fifo can be read through the API but can not be
                         // written ( graphs can read and write data )
    NC_FIFO_HOST_WO = 1, // fifo can be written through the API but can not be
                         // read (graphs can read but can not write)
} ncFifoType_t;

typedef enum {
    NC_FIFO_FP16 = 0,
    NC_FIFO_FP32 = 1,
} ncFifoDataType_t;

struct ncTensorDescriptor_t {
    unsigned int n;         // batch size, currently only 1 is supported
    unsigned int c;         // number of channels
    unsigned int w;         // width
    unsigned int h;         // height
    unsigned int totalSize; // Total size of the data in tensor = largest stride* dim size
    unsigned int cStride;   // Stride in the channels' dimension
    unsigned int wStride;   // Stride in the horizontal dimension
    unsigned int hStride;   // Stride in the vertical dimension
    ncFifoDataType_t dataType;  // data type of the tensor, FP32 or FP16
};

typedef enum {
    NC_RW_FIFO_TYPE = 0,            // configure the fifo type to one type from ncFifoType_t
    NC_RW_FIFO_CONSUMER_COUNT = 1,  // The number of consumers of elements
                                    // (the number of times data must be read by
                                    // a graph or host before the element is removed.
                                    // Defaults to 1. Host can read only once always.
    NC_RW_FIFO_DATA_TYPE = 2,       // 0 for fp16, 1 for fp32. If configured to fp32,
                                    // the API will convert the data to the internal
                                    // fp16 format automatically
    NC_RW_FIFO_DONT_BLOCK = 3,      // WriteTensor will return NC_OUT_OF_MEMORY instead
                                    // of blocking, GetResult will return NO_DATA, not supported yet
    NC_RO_FIFO_CAPACITY = 4,        // return number of maximum elements in the buffer
    NC_RO_FIFO_READ_FILL_LEVEL = 5,     // return number of tensors in the read buffer
    NC_RO_FIFO_WRITE_FILL_LEVEL = 6,    // return number of tensors in a write buffer
    NC_RO_FIFO_GRAPH_TENSOR_DESCRIPTOR = 7,   // return the tensor descriptor of the FIFO
    NC_RO_FIFO_TENSOR_DESCRIPTOR = NC_RO_FIFO_GRAPH_TENSOR_DESCRIPTOR,   // deprecated
    NC_RO_FIFO_STATE = 8,               // return the fifo state, returns CREATED, ALLOCATED,DESTROYED
    NC_RO_FIFO_NAME = 9,                // return fifo name
    NC_RO_FIFO_ELEMENT_DATA_SIZE = 10,  //element data size in bytes, int
    NC_RW_FIFO_HOST_TENSOR_DESCRIPTOR = 11,  // App's tensor descriptor, defaults to none strided channel minor
} ncFifoOption_t;

typedef enum {
    NC_DEBUG_INFO_SIZE = 0,
    NC_TIMETAKEN_SIZE = 1,
    NC_THERMAL_SIZE = 2,
    NC_NINTPUT_SIZE = 3,
    NC_NOUTPUT_SIZE = 4,
    NC_BATCH_SIZE = 5,
} ncUserGetInfo_t;



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
 * @brief Create handle and open any free device
 * @param in_ncDeviceDesc a set of parameters that the device must comply with
 * @param watchdogInterval Time interval to ping device in milliseconds. 0 to disable watchdog.
 * @param customFirmwareDirectory Custom path to directory with firmware.
 *          If NULL or empty, default path searching behavior will be used.
 */
MVNC_EXPORT_API ncStatus_t ncDeviceOpen(struct ncDeviceHandle_t **deviceHandlePtr,
    struct ncDeviceDescr_t in_ncDeviceDesc, int watchdogInterval, const char* customFirmwareDirectory);

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
MVNC_EXPORT_API ncStatus_t ncDeviceClose(struct ncDeviceHandle_t **deviceHandle);

// Graph
MVNC_EXPORT_API ncStatus_t ncGraphCreate(const char* name, struct ncGraphHandle_t **graphHandle);
MVNC_EXPORT_API ncStatus_t ncGraphAllocate(struct ncDeviceHandle_t *deviceHandle,
                           struct ncGraphHandle_t *graphHandle,
                           const void *graphBuffer, unsigned int graphBufferLength,
                           const void *graphHeader, unsigned int graphHeaderLength);
MVNC_EXPORT_API ncStatus_t ncGraphDestroy(struct ncGraphHandle_t **graphHandle);
MVNC_EXPORT_API ncStatus_t ncGraphSetOption(struct ncGraphHandle_t *graphHandle,
                            int option, const void *data, unsigned int dataLength);
MVNC_EXPORT_API ncStatus_t ncGraphGetOption(struct ncGraphHandle_t *graphHandle,
                            int option, void *data,
                            unsigned int *dataLength);
MVNC_EXPORT_API ncStatus_t ncGraphQueueInference(struct ncGraphHandle_t *graphHandle,
                            struct ncFifoHandle_t** fifoIn, unsigned int inFifoCount,
                            struct ncFifoHandle_t** fifoOut, unsigned int outFifoCount);

//Helper functions
MVNC_EXPORT_API ncStatus_t ncGraphQueueInferenceWithFifoElem(struct ncGraphHandle_t *graphHandle,
                        struct ncFifoHandle_t* fifoIn,
                        struct ncFifoHandle_t* fifoOut, const void *inputTensor,
                        unsigned int * inputTensorLength, void *userParam);
MVNC_EXPORT_API ncStatus_t ncGraphAllocateWithFifos(struct ncDeviceHandle_t* deviceHandle,
                        struct ncGraphHandle_t* graphHandle,
                        const void *graphBuffer, unsigned int graphBufferLength,
                        const void *graphHeader, unsigned int graphHeaderLength,
                        struct ncFifoHandle_t ** inFifoHandle,
                        struct ncFifoHandle_t ** outFifoHandle);

/*
 * @outNumElem A unused param, we get output size from the graph
 */
MVNC_EXPORT_API ncStatus_t ncGraphAllocateWithFifosEx(struct ncDeviceHandle_t* deviceHandle,
    struct ncGraphHandle_t* graphHandle,
    const void *graphBuffer, unsigned int graphBufferLength,
    const void *graphHeader, unsigned int graphHeaderLength,
    struct ncFifoHandle_t ** inFifoHandle, ncFifoType_t inFifoType,
    unsigned int inNumElem, ncFifoDataType_t inDataType,
    struct ncFifoHandle_t ** outFifoHandle,  ncFifoType_t outFifoType,
    unsigned int outNumElem, ncFifoDataType_t outDataType);
// Fifo
MVNC_EXPORT_API ncStatus_t ncFifoCreate(const char *name, ncFifoType_t type,
                        struct ncFifoHandle_t **fifoHandle);
MVNC_EXPORT_API ncStatus_t ncFifoAllocate(struct ncFifoHandle_t* fifoHandle,
                        struct ncDeviceHandle_t* device,
                        struct ncTensorDescriptor_t* tensorDesc,
                        unsigned int numElem);
MVNC_EXPORT_API ncStatus_t ncFifoSetOption(struct ncFifoHandle_t* fifoHandle, int option,
                        const void *data, unsigned int dataLength);
MVNC_EXPORT_API ncStatus_t ncFifoGetOption(struct ncFifoHandle_t* fifoHandle, int option,
                           void *data, unsigned int *dataLength);


MVNC_EXPORT_API ncStatus_t ncFifoDestroy(struct ncFifoHandle_t** fifoHandle);
MVNC_EXPORT_API ncStatus_t ncFifoWriteElem(struct ncFifoHandle_t* fifoHandle, const void *inputTensor,
                        unsigned int * inputTensorLength, void *userParam);
MVNC_EXPORT_API ncStatus_t ncFifoReadElem(struct ncFifoHandle_t* fifoHandle, void *outputData,
                        unsigned int* outputDataLen, void **userParam);
MVNC_EXPORT_API ncStatus_t ncFifoRemoveElem(struct ncFifoHandle_t* fifoHandle); //not supported yet

#ifdef __cplusplus
}
#endif

#endif
