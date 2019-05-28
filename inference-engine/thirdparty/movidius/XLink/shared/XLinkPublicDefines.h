// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///
#ifndef _XLINKPUBLICDEFINES_H
#define _XLINKPUBLICDEFINES_H
#include <stdint.h>
#ifdef __cplusplus
extern "C"
{
#endif

#define XLINK_MAX_STREAMS 32
#define XLINK_MAX_PACKETS_PER_STREAM 64

typedef enum{
    X_LINK_SUCCESS = 0,
    X_LINK_ALREADY_OPEN,
    X_LINK_COMMUNICATION_NOT_OPEN,
    X_LINK_COMMUNICATION_FAIL,
    X_LINK_COMMUNICATION_UNKNOWN_ERROR,
    X_LINK_DEVICE_NOT_FOUND,
    X_LINK_TIMEOUT,
    X_LINK_ERROR,
    X_LINK_OUT_OF_MEMORY
} XLinkError_t;

typedef enum{
    USB_VSC = 0,
    USB_CDC,
    PCIE,
    IPC,
    NMB_OF_PROTOCOLS
} XLinkProtocol_t;

#define USB_LINK_INVALID_FD  (-314)

#define INVALID_STREAM_ID 0xDEADDEAD
#define INVALID_STREAM_ID_OUT_OF_MEMORY 0xDEADFFFF
#define INVALID_LINK_ID   0xFF

typedef uint32_t streamId_t;
typedef uint8_t linkId_t;


typedef struct streamPacketDesc_t
{
    uint8_t* data;
    uint32_t length;

} streamPacketDesc_t;

typedef struct XLinkProf_t
{
    float totalReadTime;
    float totalWriteTime;
    unsigned long totalReadBytes;
    unsigned long totalWriteBytes;
    unsigned long totalBootCount;
    float totalBootTime;
} XLinkProf_t;

typedef struct XLinkGlobalHandler_t
{
    int loglevel;
    int profEnable;
    XLinkProtocol_t protocol;
    XLinkProf_t profilingData;
} XLinkGlobalHandler_t;

typedef struct
{
    char* devicePath;
    char* devicePath2;
    linkId_t linkId;
} XLinkHandler_t;

#ifdef __cplusplus
}
#endif

#endif

/* end of include file */
