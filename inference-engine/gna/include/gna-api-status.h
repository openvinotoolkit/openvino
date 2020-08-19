/*
    Copyright 2018 Intel Corporation.
    This software and the related documents are Intel copyrighted materials,
    and your use of them is governed by the express license under which they
    were provided to you (Intel OBL Software License Agreement (OEM/IHV/ISV
    Distribution & Single User) (v. 11.2.2017) ). Unless the License provides
    otherwise, you may not use, modify, copy, publish, distribute, disclose or
    transmit this software or the related documents without Intel's prior
    written permission.
    This software and the related documents are provided as is, with no
    express or implied warranties, other than those that are expressly
    stated in the License.
*/

/******************************************************************************
 *
 * GNA 2.0 API
 *
 * GMM Scoring and Neural Network Accelerator Module
 * API Status codes definition
 *
 *****************************************************************************/

#ifndef __GNA_API_STATUS_H
#define __GNA_API_STATUS_H

#if !defined(_WIN32)
#include <assert.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

/** Library API import/export macros */
#if !defined(GNAAPI)
#    if 1 == _WIN32
#       if 1 == INTEL_GNA_DLLEXPORT
#           define GNAAPI __declspec(dllexport)
#       else
#           define GNAAPI __declspec(dllimport)
#       endif
#    else
#        if __GNUC__ >= 4
#           define GNAAPI __attribute__ ((visibility ("default")))
#        else
#           define GNAAPI
#        endif
#    endif
#endif

/** GNA API Status codes */
typedef enum _gna_status_t
{
    GNA_INVALID_STATUS = -1, // Workaround for gcc -Wconversion error
    GNA_SUCCESS,             // Success: Operation successful, no errors or warnings
    GNA_DEVICEBUSY,          // Warning: Device busy - accelerator is still running, can not enqueue more requests
    GNA_SSATURATE,           // Warning: Scoring saturation - an arithmetic operation has resulted in saturation
    GNA_UNKNOWN_ERROR,       // Error: Unknown error occurred
    GNA_ERR_QUEUE,           // Error: Queue can not create or enqueue more requests
    GNA_READFAULT,           // Error: Scoring data: invalid input
    GNA_WRITEFAULT,          // Error: Scoring data: invalid output buffer
    GNA_BADFEATWIDTH,        // Error: Feature vector: width not supported
    GNA_BADFEATLENGTH,       // Error: Feature vector: length not supported
    GNA_BADFEATOFFSET,       // Error: Feature vector: offset not supported
    GNA_BADFEATALIGN,        // Error: Feature vector: invalid memory alignment

    GNA_BADFEATNUM,          // Error: Feature vector: number of feature vectors not supported
    GNA_INVALIDINDICES,      // Error: Scoring data: number of active indices not supported
    GNA_DEVNOTFOUND,         // Error: Device: not available
    GNA_ERR_INVALID_THREAD_COUNT,// Error: Device failed to open, thread count is invalid.
    GNA_INVALIDHANDLE,       // Error: Device: invalid handle
    GNA_CPUTYPENOTSUPPORTED, // Error: Device: processor type not supported
    GNA_PARAMETEROUTOFRANGE, // Error: Device: GMM Parameter out of Range error occurred
    GNA_VAOUTOFRANGE,        // Error: Device: Virtual Address out of range on DMA ch.
    GNA_UNEXPCOMPL,          // Error: Device: Unexpected completion during PCIe operation
    GNA_DMAREQERR,           // Error: Device: DMA error during PCIe operation

    GNA_MMUREQERR,           // Error: Device: MMU error during PCIe operation
    GNA_BREAKPOINTPAUSE,     // Error: Device: GMM accelerator paused on breakpoint
    GNA_BADMEMALIGN,         // Error: Device: invalid memory alignment
    GNA_INVALIDMEMSIZE,      // Error: Device: requested memory size not supported
    GNA_MODELSIZEEXCEEDED,   // Error: Device: request's model configuration exceeded supported GNA_HW mode limits
    GNA_BADREQID,            // Error: Device: invalid scoring request identifier
    GNA_WAITFAULT,           // Error: Device: wait failed
    GNA_IOCTLRESERR,         // Error: Device: IOCTL result retrieval failed
    GNA_IOCTLSENDERR,        // Error: Device: sending IOCTL failed
    GNA_NULLARGNOTALLOWED,   // Error: NULL argument not allowed
    GNA_INVALID_MODEL,       // Error: Given model is invalid
    GNA_INVALID_REQUEST_CONFIGURATION,       // Error: Given request configuration is invalid

    GNA_NULLARGREQUIRED,     // Error: NULL argument is required
    GNA_ERR_MEM_ALLOC1,      // Error: Memory: Already allocated, only single allocation per device is allowed
    GNA_ERR_RESOURCES,       // Error: Unable to create new resources
    GNA_ERR_NOT_MULTIPLE,    // Error: Value is not multiple of required value
    GNA_ERR_DEV_FAILURE,     // Error: Critical device error occurred, device has been reset

    GMM_BADMEANWIDTH,        // Error: Mean vector: width not supported
    GMM_BADMEANOFFSET,       // Error: Mean vector: offset not supported
    GMM_BADMEANSETOFF,       // Error: Mean vector: set offset not supported
    GMM_BADMEANALIGN,        // Error: Mean vector: invalid memory alignment
    GMM_BADVARWIDTH,         // Error: Variance vector: width not supported
    GMM_BADVAROFFSET,        // Error: Variance vector: offset not supported
    GMM_BADVARSETOFF,        // Error: Variance vector: set offset not supported
    GMM_BADVARSALIGN,        // Error: Variance vector: invalid memory alignment
    GMM_BADGCONSTOFFSET,     // Error: Gconst: set offset not supported
    GMM_BADGCONSTALIGN,      // Error: Gconst: invalid memory alignment
    GMM_BADMIXCNUM,          // Error: Scoring data: number of mixture components not supported
    GMM_BADNUMGMM,           // Error: Scoring data: number of GMMs not supported
    GMM_BADMODE,             // Error: Scoring data: GMM scoring mode not supported
    GMM_CFG_INVALID_LAYOUT,  // Error: GMM Data layout is invalid

    XNN_ERR_NET_LYR_NO,      // Error: XNN: Not supported number of layers
    XNN_ERR_NETWORK_INPUTS,  // Error: XNN: Network is invalid - input buffers number differs from input layers number
    XNN_ERR_NETWORK_OUTPUTS, // Error: XNN: Network is invalid - output buffers number differs from output layers number
    XNN_ERR_LYR_OPERATION,   // Error: XNN: Not supported layer operation
    XNN_ERR_LYR_CFG,         // Error: XNN: Layer configuration for given device(s) and operation is invalid
    XNN_ERR_LYR_INVALID_TENSOR_ORDER, // Error: XNN: Order of data tensor used in layer is invalid
    XNN_ERR_LYR_INVALID_TENSOR_DIMENSIONS, // Error: XNN: Dimensions of data tensor used in layer are invalid
    XNN_ERR_INVALID_BUFFER,  // Error: XNN: Buffer outside allocated memory
    XNN_ERR_NO_FEEDBACK,     // Error: XNN: No RNN feedback buffer specified
    XNN_ERR_NO_LAYERS,       // Error: XNN: At least one layer must be specified
    XNN_ERR_GROUPING,        // Error: XNN: Invalid grouping factor
    XNN_ERR_INPUT_BYTES,     // Error: XNN: Invalid number of bytes per input
    XNN_ERR_INPUT_VOLUME,    // Error: XNN: Invalid input volume dimensions
    XNN_ERR_OUTPUT_VOLUME,   // Error: XNN: Invalid output volume dimensions
    XNN_ERR_INT_OUTPUT_BYTES,// Error: XNN: Invalid number of bytes per intermediate output
    XNN_ERR_OUTPUT_BYTES,    // Error: XNN: Invalid number of bytes per output
    XNN_ERR_WEIGHT_BYTES,    // Error: XNN: Invalid number of bytes per weight
    XNN_ERR_WEIGHT_VOLUME,   // Error: XNN: Invalid weight/filter/GMM volume dimensions
    XNN_ERR_BIAS_BYTES,      // Error: XNN: Invalid number of bytes per bias
    XNN_ERR_BIAS_VOLUME,     // Error: XNN: Invalid bias volume dimensions
    XNN_ERR_BIAS_MODE,       // Error: XNN: Invalid bias operation mode (gna_bias_mode)
    XNN_ERR_BIAS_MULTIPLIER, // Error: XNN: Multiplier larger than 255
    XNN_ERR_BIAS_INDEX,      // Error: XNN: Bias Vector index larger than grouping factor
    XNN_ERR_PWL_SEGMENTS,    // Error: XNN: Activation function segment count is invalid, valid values: <2,128>
    XNN_ERR_PWL_DATA,        // Error: XNN: Activation function enabled but segment data not set
    XNN_ERR_CONV_FLT_BYTES,  // Error: CNN Layer: invalid number of bytes per convolution filter element
    CNN_ERR_CONV_FLT_COUNT,  // Error: CNN Layer: invalid number of convolution filters
    CNN_ERR_CONV_FLT_VOLUME, // Error: CNN Layer: Invalid convolution filter volume dimensions
    CNN_ERR_CONV_FLT_STRIDE, // Error: CNN Layer: invalid convolution filter stride
    CNN_ERR_CONV_FLT_PADDING, // Error: CNN Layer: invalid convolution padding
    CNN_ERR_POOL_STRIDE,     // Error: CNN Layer: invalid pool stride
    CNN_ERR_POOL_SIZE,       // Error: CNN Layer: invalid pooling window dimensions
    CNN_ERR_POOL_TYPE,       // Error: CNN Layer: invalid pooling function type

    XNN_ERR_MM_INVALID_IN,   // Error: XNN: Invalid input data or configuration in matrix mul. op.

    GNA_ERR_MEMORY_NOT_ALLOCATED, // Error: Memory is not yet allocated. Allocate memory first.
    GNA_ERR_MEMORY_ALREADY_MAPPED, // Error: Memory is already mapped, cannot map again. Release memory first.
    GNA_ERR_MEMORY_ALREADY_UNMAPPED, // Error: Memory is already unmapped, cannot unmap again.
    GNA_ERR_MEMORY_NOT_MAPPED,       // Error: Memory is not mapped.

    GNA_ERR_INVALID_API_VERSION,    // Error: Api version value is invalid
    GNA_ERR_INVALID_DEVICE_VERSION,    // Error: Device version value is invalid


    GNA_ERR_INVALID_DATA_MODE,  // Error: Data mode value is invalid
    GNA_ERR_NOT_IMPLEMENTED,    // Error: Functionality not implemented yet

    NUMGNASTATUS
} intel_gna_status_t, gna_status_t;       // GNA API Status codes

static_assert(4 == sizeof(intel_gna_status_t), "Invalid size of intel_gna_status_t");

/** Constant indicating that feature is disabled. */
#define GNA_DISABLED (-1)

/** Constant indicating that value is default. */
#define GNA_DEFAULT (0)

/** Constant indicating that feature is not available. */
#define GNA_NOT_SUPPORTED (0x80000000)

/**
 * Gets printable status name with the description as a c-string
 *
 * @param status        A status to translate.
 * @return A c-string status with the description.
 */
GNAAPI char const * GnaStatusToString(
    intel_gna_status_t status);

#ifdef __cplusplus
}
#endif

#endif //infndef __GNA_API_STATUS_H
