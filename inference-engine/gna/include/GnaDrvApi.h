/*
 INTEL CONFIDENTIAL
 Copyright 2019 Intel Corporation.

 The source code contained or described herein and all documents related
 to the source code ("Material") are owned by Intel Corporation or its suppliers
 or licensors. Title to the Material remains with Intel Corporation or its suppliers
 and licensors. The Material may contain trade secrets and proprietary
 and confidential information of Intel Corporation and its suppliers and licensors,
 and is protected by worldwide copyright and trade secret laws and treaty provisions.
 No part of the Material may be used, copied, reproduced, modified, published,
 uploaded, posted, transmitted, distributed, or disclosed in any way without Intel's
 prior express written permission.

 No license under any patent, copyright, trade secret or other intellectual
 property right is granted to or conferred upon you by disclosure or delivery
 of the Materials, either expressly, by implication, inducement, estoppel
 or otherwise. Any license under such intellectual property rights must
 be express and approved by Intel in writing.

 Unless otherwise agreed by Intel in writing, you may not remove or alter this notice
 or any other notice embedded in Materials by Intel or Intel's suppliers or licensors
 in any way.
*/

#pragma once

#include <initguid.h>

/******************************************************************************

 Driver interface

 *****************************************************************************/

/**
 Interface Guid
 {8113B324-9F9B-4B9F-BF55-1342A58593DC}
 */
DEFINE_GUID(GUID_DEVINTERFACE_GNA_DRV,
    0x8113b324, 0x9f9b, 0x4b9f, 0xbf, 0x55, 0x13, 0x42, 0xa5, 0x85, 0x93, 0xdc);

#define FILE_DEVICE_PCI_GNA 0x8000

#define GNA_IOCTL_MEM_MAP2   CTL_CODE(FILE_DEVICE_PCI_GNA, 0xA00, METHOD_IN_DIRECT, FILE_ANY_ACCESS)
#define GNA_IOCTL_MEM_UNMAP2 CTL_CODE(FILE_DEVICE_PCI_GNA, 0xA01, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_GET_PARAM  CTL_CODE(FILE_DEVICE_PCI_GNA, 0xA02, METHOD_BUFFERED, FILE_ANY_ACCESS)
#define GNA_IOCTL_NOTIFY     CTL_CODE(FILE_DEVICE_PCI_GNA, 0xA03, METHOD_BUFFERED, FILE_ANY_ACCESS)

/* Status register flags */
#define STS_SATURATION_FLAG 0x20000 // WARNING: score has reached the saturation, MUST CLEAR
#define STS_OUTBUFFULL_FLAG 0x10000 // WARNING: hw output buffer is currently full, MUST CLEAR
#define STS_PARAM_OOR_FLAG  0x100   // ERROR: hw parameter out of range
#define STS_VA_OOR_FLAG     0x80    // ERROR: VA out of range
#define STS_UNEXPCOMPL_FLAG 0x40    // ERROR: PCIe error: unexpected completion
#define STS_DMAREQERR_FLAG  0x20    // ERROR: PCIe error: DMA req
#define STS_MMUREQERR_FLAG  0x10    // ERROR: PCIe error: MMU req
#define STS_STATVALID_FLAG  0x08    // compute statistics valid
#define STS_SDTPASUE_FLAG   0x04    // suspended due to pause
#define STS_BPPASUE_FLAG    0x02    // suspended breakpoint match
#define STS_COMPLETED_FLAG  0x01    // scoring completed flag

/* GNA device/driver parameters */
#define GNA_PARAM_DEVICE_ID        1
#define GNA_PARAM_RECOVERY_TIMEOUT 2
#define GNA_PARAM_DEVICE_TYPE      3
#define GNA_PARAM_INPUT_BUFFER_S   4
#define GNA_PARAM_CE_NUM           5
#define GNA_PARAM_PLE_NUM          6
#define GNA_PARAM_AFE_NUM          7
#define GNA_PARAM_HAS_MMU          8
#define GNA_PARAM_HW_VER           9

/**
 Default time in seconds after which driver will try to auto recover
 from hardware hang
*/
#define DRV_RECOVERY_TIMEOUT 60

/******************************************************************************

 Driver IOCTL's input-output data structures

 ******************************************************************************
 NOTE: all below IOCTL in/out data type structures have to be 8 B padded
          as this is required for x86-x64 spaces cooperation
 *****************************************************************************/

# pragma pack (1) // set structure packaging to 1 to ensure alignment and size

// disable zero-sized array in struct/union warning
#pragma warning(disable:4200)

// disables anonymous struct/unions warning, useful to flatten structs
#pragma warning(disable:4201)

/**
 Device types for GNA_PARAM_DEVICE_TYPE
 */

// No supported device available
#define GNA_HW_NO_DEVICE 0x00

// GMM Device
#define GNA_HW_GMM 0x01

// GNA 0.9 Device, no CNN support
#define GNA_HW_0_9 0x09

// GNA 1.0 Device, full featured GNA 1.0
#define GNA_HW_1_0 0x10

// GNA 2.0 Device, full featured GNA 2.0
#define GNA_HW_2_0 0x20

// GNA 3.0 Device, full featured GNA 3.0
#define GNA_HW_3_0 0x30

/**
 Calculate Control flags
 */
typedef struct
{
    // active list mode (0:disabled, 1:enabled)
    UINT32 activeListOn : 1;

    // GNA operation mode (0:GMM, 1:xNN)
    UINT32 gnaMode : 2;
    UINT32 ddiVersion : 21;
    UINT32 hwPerfEncoding : 8;
    union
    {
    // backward compatibility: size of layer descriptors sent
    UINT32 xnnLyrDscSize;
    UINT32 layerCount;
    };

} CTRL_FLAGS;

static_assert(8 == sizeof(CTRL_FLAGS), "Invalid size of CTRL_FLAGS");

/**
 Accelerator (hardware level) scoring request performance results
 */
typedef struct
{
    // # of total cycles spent on scoring in hw
    UINT64 total;

    // # of stall cycles spent in hw (since scoring)
    UINT64 stall;

} GNA_PERF_HW;

static_assert(16 == sizeof(GNA_PERF_HW), "Invalid size of GNA_PERF_HW");

/**
 Accelerator (driver level) scoring request performance results
 */
typedef struct
{
    // time of setting up and issuing HW scoring
    UINT64 startHW;

    // time between HW scoring start and scoring complete interrupt
    UINT64 scoreHW;

    // time of processing scoring complete interrupt
    UINT64 intProc;

} GNA_PERF_DRV;

static_assert(24 == sizeof(GNA_PERF_DRV), "Invalid size of GNA_PERF_DRV");

/**
 Size of GNA (GMM/xNN) configuration data in bytes
 */
#define CFG_SIZE 256

/**
 Legacy CALCULATE request data with output information.
 Size:    312 B
 NOTE: always include performance results
 this allow to use PROFILED library with NON-PROFILED driver and vice versa
 */
typedef struct _GNA_CALC_IN
{
    CTRL_FLAGS          ctrlFlags;  // scoring mode
    UINT8               config[CFG_SIZE];// configuration data for GMM or xNN
    GNA_PERF_DRV        drvPerf;    // driver level performance profiling results
    GNA_PERF_HW         hwPerf;     // hardware level performance results
    UINT32              status;     // status of scoring
    UINT32              __res;      // 4 B padding to multiple 8 B size

} GNA_CALC_IN, *PGNA_CALC_IN;       // CALCULATE IOCTL - Input data

static_assert(312 == sizeof(GNA_CALC_IN), "Invalid size of GNA_CALC_IN");

/**
 Driver instrumentation results
 */
typedef struct
{
    /**
     Request preprocessing start
     */
    UINT64 Preprocessing;

    /**
     Request processing started by hardware
     */
    UINT64 Processing;

    /**
     Request completed interrupt triggered by hardware
     */
    UINT64 DeviceRequestCompleted;

    /**
     Driver completed interrupt and request handling.
     */
    UINT64 Completion;

} GNA_DRIVER_INSTRUMENTATION;

static_assert(32 == sizeof(GNA_DRIVER_INSTRUMENTATION), "Invalid size of GNA_DRIVER_INSTRUMENTATION");

typedef struct _INFERENCE_CONFIG_IN
{
    // scoring mode
    // @note This must be first field as it's required to be compatible with legacy GNA_CALC_IN
    CTRL_FLAGS ctrlFlags;

    // layer base / offset to gmm descriptor
    UINT32 configBase;

    UINT32 _reserved;

    // number of buffers lying outside this structure
    UINT64 bufferCount;

    // memory buffers with patches
    UINT8 buffers[];

} GNA_INFERENCE_CONFIG_IN, * PGNA_INFERENCE_CONFIG_IN;

static_assert(24 == sizeof(GNA_INFERENCE_CONFIG_IN), "Invalid size of GNA_INFERENCE_CONFIG_IN");

typedef struct _INFERENCE_CONFIG_OUT
{
    GNA_DRIVER_INSTRUMENTATION driverInstrumentation;

    GNA_PERF_HW hardwareInstrumentation;

    UINT32 _reserved;

    UINT32 status;

} GNA_INFERENCE_CONFIG_OUT, *PGNA_INFERENCE_CONFIG_OUT;

static_assert(56 == sizeof(GNA_INFERENCE_CONFIG_OUT), "Invalid size of GNA_INFERENCE_CONFIG_OUT");

/**
 Inference config
 */
typedef union
{
    GNA_INFERENCE_CONFIG_IN input;

    GNA_INFERENCE_CONFIG_OUT output;

} GNA_INFERENCE_CONFIG, * PGNA_INFERENCE_CONFIG;

static_assert(56 == sizeof(GNA_INFERENCE_CONFIG), "Invalid size of GNA_INFERENCE_CONFIG");

/**
 User buffer identified by memory id
 List of such buffers are received in WRITE request to driver
 Each buffer is added to MMU according to it's offset and size
 Each buffer may contain patches that driver will apply to the memory before starting GNA
 */
typedef struct
{
    UINT64 memoryId;
    UINT64 offset;
    UINT64 size;
    UINT64 patchCount;

} GNA_MEMORY_BUFFER, *PGNA_MEMORY_BUFFER;

static_assert(32 == sizeof(GNA_MEMORY_BUFFER), "Invalid size of GNA_MEMORY_BUFFER");

/**
 Patch structure describes memory location that has to be patched before request
 Memory is patched according to provided data and it's size
 List of such patches are received in WRITE request to driver
 Each patch is linked to memory described by GNA_MEMORY_BUFFER
 */
typedef struct
{
    UINT64 offset;
    UINT64 size;
    UINT8 data[];

} GNA_MEMORY_PATCH, *PGNA_MEMORY_PATCH;

static_assert(16 == sizeof(GNA_MEMORY_PATCH), "Invalid size of GNA_MEMORY_PATCH");

#pragma pack ()
