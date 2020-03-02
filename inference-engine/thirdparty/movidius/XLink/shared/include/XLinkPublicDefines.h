// Copyright (C) 2018-2020 Intel Corporation
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

#define XLINK_MAX_NAME_SIZE 28
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
    X_LINK_OUT_OF_MEMORY,
    X_LINK_NOT_IMPLEMENTED
} XLinkError_t;

typedef enum{
    X_LINK_USB_VSC = 0,
    X_LINK_USB_CDC,
    X_LINK_PCIE,
    X_LINK_IPC,
    X_LINK_NMB_OF_PROTOCOLS,
    X_LINK_ANY_PROTOCOL
} XLinkProtocol_t;

typedef enum{
    X_LINK_ANY_PLATFORM = 0,
    X_LINK_MYRIAD_2 = 2450,
    X_LINK_MYRIAD_X = 2480,
} XLinkPlatform_t;

typedef enum{
    X_LINK_ANY_STATE = 0,
    X_LINK_BOOTED,
    X_LINK_UNBOOTED,
} XLinkDeviceState_t;

typedef enum{
    X_LINK_PCIE_UNKNOWN_BOOTLOADER = 0,
    X_LINK_PCIE_SIMPLIFIED_BOOTLOADER = 1,
    X_LINK_PCIE_UNIFIED_BOOTLOADER = 2
} XLinkPCIEBootloader;

#define INVALID_STREAM_ID 0xDEADDEAD
#define INVALID_STREAM_ID_OUT_OF_MEMORY 0xDEADFFFF
#define INVALID_LINK_ID   0xFF
#define MAX_STREAM_NAME_LENGTH 64

typedef uint32_t streamId_t;
typedef uint8_t linkId_t;

typedef struct {
    XLinkProtocol_t protocol;
    XLinkPlatform_t platform;
    char name[XLINK_MAX_NAME_SIZE];
} deviceDesc_t;

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
    int profEnable;
    XLinkProf_t profilingData;

    //Deprecated fields. Begin.
    int loglevel;
    int protocol;
    //Deprecated fields. End.
} XLinkGlobalHandler_t;

typedef struct
{
    char* devicePath;
    char* devicePath2;
    linkId_t linkId;
    XLinkProtocol_t protocol;
} XLinkHandler_t;

const char* XLinkErrorToStr(XLinkError_t rc);

//Deprecated defines. Begin.

typedef enum{
    USB_VSC = 0,
    USB_CDC,
    PCIE,
    IPC,
    NMB_OF_PROTOCOLS
} XLinkProtocol_deprecated_t;

//Deprecated defines. End.

#ifdef __cplusplus
}
#endif

#endif

/* end of include file */
