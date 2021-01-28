// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///
#ifndef OPENVINO_XLINK_PRIVATEDEFINES_H
#define OPENVINO_XLINK_PRIVATEDEFINES_H

#include "XLinkPublicDefines.h"

#if !defined(XLINK_ALIGN_TO_BOUNDARY)
# if defined(_WIN32) && !defined(__GNUC__)
#  define XLINK_ALIGN_TO_BOUNDARY(_n) __declspec(align(_n))
# else
#  define XLINK_ALIGN_TO_BOUNDARY(_n) __attribute__((aligned(_n)))
# endif
#endif  // XLINK_ALIGN_TO_BOUNDARY

#ifdef __cplusplus
extern "C"
{
#endif

#define MAXIMUM_SEMAPHORES 32
#define __CACHE_LINE_SIZE 64

typedef int32_t eventId_t;

/**
 * @brief Device description
 */
typedef struct xLinkDeviceHandle_t {
    XLinkProtocol_t protocol;
    void* xLinkFD;
} xLinkDeviceHandle_t;

//events which are coming from remote
typedef enum
{
    /*USB-X_LINK_PCIE related events*/
    XLINK_WRITE_REQ,
    XLINK_READ_REQ,
    XLINK_READ_REL_REQ,
    XLINK_CREATE_STREAM_REQ,
    XLINK_CLOSE_STREAM_REQ,
    XLINK_PING_REQ,
    XLINK_RESET_REQ,
    XLINK_REQUEST_LAST,
    //note that is important to separate request and response
    XLINK_WRITE_RESP,
    XLINK_READ_RESP,
    XLINK_READ_REL_RESP,
    XLINK_CREATE_STREAM_RESP,
    XLINK_CLOSE_STREAM_RESP,
    XLINK_PING_RESP,
    XLINK_RESET_RESP,
    XLINK_RESP_LAST,

    /*X_LINK_IPC related events*/
    IPC_WRITE_REQ,
    IPC_READ_REQ,
    IPC_CREATE_STREAM_REQ,
    IPC_CLOSE_STREAM_REQ,
    //
    IPC_WRITE_RESP,
    IPC_READ_RESP,
    IPC_CREATE_STREAM_RESP,
    IPC_CLOSE_STREAM_RESP,
} xLinkEventType_t;


typedef enum
{
    EVENT_LOCAL,
    EVENT_REMOTE,
} xLinkEventOrigin_t;

#ifdef __PC__
#define MAX_LINKS 32
#else
#define MAX_LINKS 1
#endif

#define XLINK_MAX_DEVICES MAX_LINKS

#ifdef __cplusplus
}
#endif

#endif  // OPENVINO_XLINK_PRIVATEDEFINES_H

/* end of include file */
