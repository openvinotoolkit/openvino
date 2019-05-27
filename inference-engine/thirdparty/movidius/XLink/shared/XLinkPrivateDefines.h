/*
* Copyright 2018-2019 Intel Corporation.
* The source code, information and material ("Material") contained herein is
* owned by Intel Corporation or its suppliers or licensors, and title to such
* Material remains with Intel Corporation or its suppliers or licensors.
* The Material contains proprietary information of Intel or its suppliers and
* licensors. The Material is protected by worldwide copyright laws and treaty
* provisions.
* No part of the Material may be used, copied, reproduced, modified, published,
* uploaded, posted, transmitted, distributed or disclosed in any way without
* Intel's prior express written permission. No license under any patent,
* copyright or other intellectual property rights in the Material is granted to
* or conferred upon you, either expressly, by implication, inducement, estoppel
* or otherwise.
* Any license under such intellectual property rights must be express and
* approved by Intel in writing.
*/

///
/// @file
///
/// @brief     Application configuration Leon header
///
#ifndef _XLINKPRIVATEDEFINES_H
#define _XLINKPRIVATEDEFINES_H

#ifdef _XLINK_ENABLE_PRIVATE_INCLUDE_

#include <stdint.h>
#if (defined(_WIN32) || defined(_WIN64))
#include "win_semaphore.h"
#else
#include <semaphore.h>
#endif
#include <XLinkPublicDefines.h>
#include "XLinkPlatform.h"

#ifdef __cplusplus
extern "C"
{
#endif
#define MAX_NAME_LENGTH 16

#ifdef USE_USB_VSC
#define HEADER_SIZE (64-12 -8)
#else
#define HEADER_SIZE (64-12 -8)
#endif

#define MAXIMUM_SEMAPHORES 32
#define __CACHE_LINE_SIZE 64

#ifdef NDEBUG  // Release configuration
    #ifndef __PC__
        #define ASSERT_X_LINK(x)   if(!(x)) { exit(EXIT_FAILURE); }
        #define ASSERT_X_LINK_R(x, r) ASSERT_X_LINK(x)
    #else
        #define ASSERT_X_LINK(x)   if(!(x)) { return X_LINK_ERROR; }
        #define ASSERT_X_LINK_R(x, r)   if(!(x)) { return r; }
    #endif
#else   // Debug configuration
    #ifndef __PC__
        #define ASSERT_X_LINK(x)   if(!(x)) { fprintf(stderr, "%s:%d:\n Assertion Failed: %s\n", __FILE__, __LINE__, #x); exit(EXIT_FAILURE); }
        #define ASSERT_X_LINK_R(x, r) ASSERT_X_LINK(x)
    #else
        #define ASSERT_X_LINK(x)   if(!(x)) { fprintf(stderr, "%s:%d:\n Assertion Failed: %s\n", __FILE__, __LINE__, #x); return X_LINK_ERROR; }
        #define ASSERT_X_LINK_R(x, r)   if(!(x)) { fprintf(stderr, "%s:%d:\n Assertion Failed: %s\n", __FILE__, __LINE__, #x); return r; }
    #endif
#endif //  NDEBUG

#ifndef CHECK_MUTEX_SUCCESS
#define CHECK_MUTEX_SUCCESS(call)  {                                \
    int error;                                                      \
    if ((error = (call))) {                                         \
      mvLog(MVLOG_ERROR, "%s failed with error: %d", #call, error); \
    }                                                               \
}
#endif  // CHECK_MUTEX_SUCCESS

#ifndef CHECK_MUTEX_SUCCESS_RC
#define CHECK_MUTEX_SUCCESS_RC(call, rc)  {                         \
    int error;                                                      \
    if ((error = (call))) {                                         \
      mvLog(MVLOG_ERROR, "%s failed with error: %d", #call, error); \
      return rc;                                                    \
    }                                                               \
}
#endif  // CHECK_MUTEX_SUCCESS_RC

typedef int32_t eventId_t;

/**
 * @brief State for xLinkDesc_t
 */
typedef enum {
    XLINK_NOT_INIT,
    XLINK_UP,
    XLINK_DOWN,
} xLinkState_t;

/**
 * @brief Streams opened to device
 */
typedef struct{
    char name[MAX_NAME_LENGTH];
    streamId_t id;
    void* fd;
    uint32_t writeSize;
    uint32_t readSize;  /*No need of read buffer. It's on remote,
    will read it directly to the requested buffer*/
    streamPacketDesc_t packets[XLINK_MAX_PACKETS_PER_STREAM];
    uint32_t availablePackets;
    uint32_t blockedPackets;

    uint32_t firstPacket;
    uint32_t firstPacketUnused;
    uint32_t firstPacketFree;
    uint32_t remoteFillLevel;
    uint32_t localFillLevel;
    uint32_t remoteFillPacketLevel;

    uint32_t closeStreamInitiated;

    sem_t sem;
} streamDesc_t;

/**
 * @brief XLink primitive for each device
 */
typedef struct xLinkDesc_t {
    // Incremental number, doesn't get decremented.
    int nextUniqueStreamId;
    streamDesc_t availableStreams[XLINK_MAX_STREAMS];
    xLinkState_t peerState;
    void* fd;
    linkId_t id;
} xLinkDesc_t;


//events which are coming from remote
typedef enum
{
    /*USB-PCIE related events*/
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

    /*IPC related events*/
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

#define MAX_EVENTS 64

#define MAX_SCHEDULERS MAX_LINKS

typedef struct xLinkEventHeader_t{
    eventId_t           id;
    xLinkEventType_t    type;
    char                streamName[MAX_NAME_LENGTH];
    streamId_t          streamId;
    uint32_t            size;
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
    xLinkEventHeader_t header;
    void* xLinkFD;
    void* data;
}xLinkEvent_t;

int XLinkWaitSem(sem_t* sem);

int XLinkWaitSemUserMode(sem_t* sem, unsigned int timeout);

const char* XLinkErrorToStr(XLinkError_t rc);

#ifdef __cplusplus
}
#endif

#endif  /*_XLINK_ENABLE_PRIVATE_INCLUDE_ end*/
#endif

/* end of include file */
