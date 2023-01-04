// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///
/// @file
///
/// @brief     Application configuration Leon header
///
#ifndef _XLINKPRIVATEDEFINES_H
#define _XLINKPRIVATEDEFINES_H

#include "XLinkStream.h"

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

#ifdef XLINK_MAX_STREAM_RES
#define MAXIMUM_SEMAPHORES XLINK_MAX_STREAM_RES
#else
#define MAXIMUM_SEMAPHORES 32
#endif
#define __CACHE_LINE_SIZE 64

typedef int32_t eventId_t;

/**
 * @brief State for xLinkDesc_t
 */
typedef enum {
    XLINK_NOT_INIT,
    XLINK_UP,
    XLINK_DOWN,
}xLinkState_t;

/**
 * @brief Device description
 */
typedef struct xLinkDeviceHandle_t {
    XLinkProtocol_t protocol;
    void* xLinkFD;
} xLinkDeviceHandle_t;

/**
 * @brief XLink primitive for each device
 */
typedef struct xLinkDesc_t {
    // Incremental number, doesn't get decremented.
    uint32_t nextUniqueStreamId;
    streamDesc_t availableStreams[XLINK_MAX_STREAMS];
    xLinkState_t peerState;
    xLinkDeviceHandle_t deviceHandle;
    linkId_t id;
    XLink_sem_t dispatcherClosedSem;

    //Deprecated fields. Begin.
    int hostClosedFD;
    //Deprecated fields. End.

} xLinkDesc_t;

streamId_t XLinkAddOrUpdateStream(void *fd, const char *name,
                                  uint32_t writeSize, uint32_t readSize, streamId_t forcedId);

//events which are coming from remote
typedef enum
{
    /*USB-X_LINK_PCIE related events*/
    XLINK_WRITE_REQ,
    XLINK_READ_REQ,
    XLINK_READ_REL_REQ,
    XLINK_READ_REL_SPEC_REQ,
    XLINK_CREATE_STREAM_REQ,
    XLINK_CLOSE_STREAM_REQ,
    XLINK_PING_REQ,
    XLINK_RESET_REQ,
    XLINK_DROP_REQ,
    XLINK_REQUEST_LAST,
    //note that is important to separate request and response
    XLINK_WRITE_RESP,
    XLINK_READ_RESP,
    XLINK_READ_REL_RESP,
    XLINK_READ_REL_SPEC_RESP,
    XLINK_CREATE_STREAM_RESP,
    XLINK_CLOSE_STREAM_RESP,
    XLINK_PING_RESP,
    XLINK_RESET_RESP,
    XLINK_DROP_RESP,
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

#define MAX_EVENTS 64
#define MAX_SCHEDULERS MAX_LINKS
#define XLINK_MAX_DEVICES MAX_LINKS

typedef struct xLinkEventHeader_t{
    eventId_t           id;
    xLinkEventType_t    type;
    char                streamName[MAX_STREAM_NAME_LENGTH];
    streamId_t          streamId;
    uint32_t            size;
    uint32_t            dropped;
    uint32_t            canBeServed;
    union{
        uint32_t raw;
        struct{
            uint32_t ack : 1;
            uint32_t nack : 1;
            uint32_t block : 1;
            uint32_t localServe : 1;
            uint32_t terminate : 1;
            uint32_t bufferFull : 1;
            uint32_t sizeTooBig : 1;
            uint32_t noSuchStream : 1;
        }bitField;
    }flags;
}xLinkEventHeader_t;

typedef struct xLinkEvent_t {
    XLINK_ALIGN_TO_BOUNDARY(64) xLinkEventHeader_t header;
    xLinkDeviceHandle_t deviceHandle;
    void* data;
}xLinkEvent_t;

#define XLINK_INIT_EVENT(event, in_streamId, in_type, in_size, in_data, in_deviceHandle) do { \
    (event).header.streamId = (in_streamId); \
    (event).header.type = (in_type); \
    (event).header.size = (in_size); \
    (event).header.dropped = 0; \
    (event).header.canBeServed = 1; \
    (event).data = (in_data); \
    (event).deviceHandle = (in_deviceHandle); \
} while(0)

#define XLINK_EVENT_ACKNOWLEDGE(event) do { \
    (event)->header.flags.bitField.ack = 1; \
    (event)->header.flags.bitField.nack = 0; \
} while(0)

#define XLINK_EVENT_NOT_ACKNOWLEDGE(event) do { \
    (event)->header.flags.bitField.ack = 0; \
    (event)->header.flags.bitField.nack = 1; \
} while(0)

#define XLINK_SET_EVENT_FAILED_AND_SERVE(event) do { \
    XLINK_EVENT_NOT_ACKNOWLEDGE(event); \
    (event)->header.flags.bitField.localServe = 1; \
} while(0)

#ifdef __cplusplus
}
#endif

#endif

/* end of include file */
