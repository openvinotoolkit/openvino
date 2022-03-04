// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Includes
// ----------------------------------------------------------------------------
#ifndef _MVNC_COMM_H_
#define _MVNC_COMM_H_

#include <stdint.h>
#include <mvnc.h>

#include "XLinkPublicDefines.h"

#ifdef __cplusplus
extern "C"
{
#endif

struct tensorDescriptor_t {
    uint32_t n;
    uint32_t c;
    uint32_t w;
    uint32_t h;
    uint32_t totalSize;
};

typedef enum {
    MVNCI_SUCCESS,
    MVNCI_NULL_PARAM,
    MVNCI_MASK_NOTCONTINUOUS,
    MVNCI_UNSUPPORTED_NETWORK_ELEMENT,
    MVNCI_INVALID_HANDLE,
    MVNCI_OUT_OF_RESOURCES,
    MVNCI_NOT_IMPLEMENTED,
    MVNCI_SHAVES_SLICES_MISMATCH,
    MVNCI_TIMEOUT,
    MVNCI_INTERNAL_ERROR,
    MVNCI_OUT_OF_MEMORY,
} ncMvNCIErrorCode_t;

//---------------------------------------------------------
//----- Graph monitor's communication types. Begin. -------
//---------------------------------------------------------

typedef enum {
    GRAPH_WAITING_FOR_BUFFERS,
    GRAPH_RUNNING
} graphState_t;

typedef enum {
    GRAPH_ALLOCATE_CMD          = 0,
    GRAPH_DEALLOCATE_CMD        = 1,
    GRAPH_TRIGGER_CMD           = 2,
    GRAPH_VERIFY_CMD            = 3,
    GRAPH_ALLOCATION_VERIFY_CMD = 4,
    GRAPH_BUFFER_ALLOCATE_CMD   = 5,
    GRAPH_BUFFER_DEALLOCATE_CMD = 6,
    GRAPH_GET_TIMING_DATA       = 7,
    GRAPH_GET_DEBUG_DATA        = 8,
    GRAPH_COMMAND_LAST          = 10,
} graphCommandType_t;

typedef struct {
    graphCommandType_t type;
    uint32_t id;
    char streamName[MAX_STREAM_NAME_LENGTH];
    uint32_t buffId1;
    uint32_t buffId2;
    uint32_t executors_number;
} graphCMDCommand_t;

typedef struct {
    graphCommandType_t type;
    uint32_t id;
    char name[MAX_STREAM_NAME_LENGTH];
    uint32_t elemCnt;
    struct tensorDescriptor_t desc;
    uint8_t readChannel;
    uint8_t writeChannel;
} bufferAllocateCommand_t;

typedef struct {
    graphCommandType_t type;
    uint32_t id;
} graphCommonCommand_t;

//-------------------------------------------------------
//----- Graph monitor's communication types. End. -------
//-------------------------------------------------------



//---------------------------------------------------------
//----- Device monitor's communication types. Begin. ------
//---------------------------------------------------------

typedef struct {
    uint32_t max_graphs;
    uint32_t max_fifos;
    uint32_t max_memory;
    uint32_t max_executors;
    uint32_t fw_version[4];
    uint32_t mv_tensor_version[2];
} deviceCapabilities_t;

typedef enum {
    DEVICE_GET_THERMAL_STATS        = 0,
    DEVICE_GET_CAPABILITIES         = 1,
    DEVICE_GET_USED_MEMORY          = 2,
    DEVICE_GET_DEVICE_ID            = 3,
    DEVICE_WATCHDOG_PING            = 4,
    DEVICE_SET_STDIO_REDIRECT_XLINK = 5,
    DEVICE_SET_POWER_CONFIG         = 6,
    DEVICE_RESET_POWER_CONFIG       = 7,
    DEVICE_ENABLE_ASYNC_DMA         = 8,
    DEVICE_COMMAND_LAST             = 9
} deviceCommandType_t;

typedef struct {
    deviceCommandType_t type;
    uint32_t arg;
} deviceCommand_t;

//---------------------------------------------------------
//----- Device monitor's communication types. End. --------
//---------------------------------------------------------

ncStatus_t getFirmwarePath(char *firmware_file_path, const int firmware_file_length,
                           const deviceDesc_t deviceDesc);

#ifdef __cplusplus
}
#endif

#endif
