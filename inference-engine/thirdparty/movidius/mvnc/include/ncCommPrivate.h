// Copyright (C) 2018-2019 Intel Corporation
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

// TODO: #-17902
struct tensorDescriptor_t {
    uint32_t n;
    uint32_t c;
    uint32_t w;
    uint32_t h;
    uint32_t totalSize;
    uint32_t widthStride;
    uint32_t heightStride;
    uint32_t channelsStride;
};
typedef enum {
    NC_GRAPH_OK,
    NC_GRAPH_WRONG_INPUT_FORMAT,
    NC_GRAPH_MYRIAD_ERROR
} ncGraphError_t;

typedef enum {
    GRAPH_WAITING_FOR_BUFFERS,
    GRAPH_RUNNING
} graphState_t;

typedef enum {
    GRAPH_MON_CLASS_GRAPH_CMD = 0,
    GRAPH_MON_CLASS_BUFFER_CMD = 1,
    GRAPH_MON_CLASS_GET_CLASS0 = 2,
    GRAPH_MON_CLASS_GET_CLASS1 = 3,
    GRAPH_MON_CLASS_GET_CLASS2 = 4,
    GRAPH_MON_CLASS_GET_CLASS3 = 5,
    GRAPH_MON_CLASS_SET_CLASS0 = 6,
    GRAPH_MON_CLASS_SET_CLASS1 = 7,
    GRAPH_MON_CLASS_SET_CLASS2 = 8,
    GRAPH_MON_CLASS_SET_CLASS3 = 9,
} graphMonClass_t;

typedef enum {
    GRAPH_VERIFY_CMD = 3,
    GRAPH_ALLOCATE_CMD = 0,
    GRAPH_DEALLOCATE_CMD = 1,
    GRAPH_TRIGGER_CMD = 2,
} graphCommandType_t;

typedef enum {
    CLASS0_TIMING_DATA = 0,
    CLASS0_DEBUG_DATA = 1,
    CLASS0_STATE = 2,
} graphOptionClass0_t;
typedef enum {
    CLASS1_GR_NI = 0,
} graphOptionClass1_t;
typedef enum {
    CLASS2_GR_NI = 0,
} graphOptionClass2_t;
typedef enum {
    CLASS3_GR_NI = 0,
} graphOptionClass3_t;

typedef enum {
    BUFFER_ALLOCATE_CMD = 0,
    BUFFER_DEALLOCATE_CMD = 1,
} bufferCommandType_t;

typedef struct {
    graphCommandType_t type;
    uint32_t id;
    char streamName[MAX_STREAM_NAME_LENGTH];
    uint32_t buffId1;
    uint32_t buffId2;
    uint32_t executors_number;
    uint8_t laterUse[24];
} graphCommand_t;

typedef struct {
    bufferCommandType_t type;
    char name[MAX_STREAM_NAME_LENGTH];
    uint32_t id;
    uint32_t elemCnt;
    struct tensorDescriptor_t desc;
    uint8_t readChannel;
    uint8_t writeChannel;
    uint8_t laterUse[10];
} bufferCommand_t;

typedef struct {
    union {
        graphOptionClass0_t c0;
        graphOptionClass1_t c1;
        graphOptionClass2_t c2;
        graphOptionClass3_t c3;
    } type;
    uint32_t id;
} graphOptionSet_t;

typedef struct {
    graphMonClass_t cmdClass;
    union {
        graphCommand_t graphCmd;
        bufferCommand_t buffCmd;
        graphOptionSet_t optionCmd;
    } cmd;
} graphMonCommand_t;

typedef enum {
    CLASS0_THERMAL_STATS = 1,
    CLASS0_DEVICE_CAPABILITIES = 2,
    CLASS0_DEVICE_USED_MEMORY = 3,
    CLASS0_DEVICE_ID = 4,
    /* constants for internal profiling below */
    CLASS0_DEVICE_PROFILING_DATA = 10001,
    CLASS0_DEVICE_QUERY_CLOCKS,
} deviceOptionClass0;
typedef enum {
    CLASS1_WATCHDOG_PING = 0,
} deviceOptionClass1;
typedef enum {
    CLASS2_GET_TEMP_LIM_LOWER = 0,
    CLASS2_SET_TEMP_LIM_LOWER,
    CLASS2_GET_TEMP_LIM_HIGHER,
    CLASS2_SET_TEMP_LIM_HIGHER,
    CLASS2_GET_BACKOFF_TIME_NORMAL,
    CLASS2_SET_BACKOFF_TIME_NORMAL,
    CLASS2_GET_BACKOFF_TIME_HIGH,
    CLASS2_SET_BACKOFF_TIME_HIGH,
    CLASS2_GET_BACKOFF_TIME_CRITICAL,
    CLASS2_SET_BACKOFF_TIME_CRITICAL,
    CLASS2_GET_TEMPERATURE_DEBUG,
    CLASS2_SET_TEMPERATURE_DEBUG,
    CLASS2_SET_STDIO_REDIRECT_XLINK,
    CLASS2_OPT_LIST,
} deviceOptionClass2;

typedef enum {
    CLASS3_START_SHELL = 0,
    CLASS3_SET_LOG_LEVEL_GLOBAL,
    CLASS3_SET_LOG_LEVEL_FATHOM,
    CLASS3_SET_LOG_LEVEL_XLINK,
} deviceOptionClass3;

typedef struct {
    union {
        deviceOptionClass0 c0;
        deviceOptionClass1 c1;
        deviceOptionClass2 c2;
        deviceOptionClass3 c3;
    } type;
    uint32_t optionClass;
    uint32_t data;
} deviceCommand_t;

typedef struct {
    uint32_t max_graphs;
    uint32_t max_fifos;
    uint32_t max_memory;
    uint32_t max_device_opt_class;
    uint32_t max_graph_opt_class;
    uint32_t max_executors;
    uint32_t fw_version[4];
    uint32_t mv_tensor_version[2];
} deviceCapabilities_t;

ncStatus_t getFirmwarePath(char* mv_cmd_file_path, const char* dev_addr);

#ifdef __cplusplus
}
#endif

#endif
