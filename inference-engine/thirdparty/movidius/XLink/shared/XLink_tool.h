// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINK_TOOL_H
#define _XLINK_TOOL_H

#ifdef __cplusplus
extern "C"
{
#endif
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
#define ASSERT_X_LINK(x)        if(!(x)) {  \
            mvLog(MVLOG_ERROR, "%s:%d\n\t Assertion Failed: %s", __FILE__, __LINE__, #x);   \
            return X_LINK_ERROR;    \
        }
#define ASSERT_X_LINK_R(x, r)   if(!(x)) {  \
            mvLog(MVLOG_ERROR, "%s:%d\n\t Assertion Failed: %s", __FILE__, __LINE__, #x);   \
            return r;               \
        }
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

#define CIRCULAR_INCREMENT(x, maxVal) \
        { \
             x++; \
             if (x == maxVal) \
                 x = 0; \
        }

//avoid problems with unsigned. first compare and then give the nuw value
#define CIRCULAR_DECREMENT(x, maxVal) \
    { \
        if (x == 0) \
            x = maxVal; \
        else \
            x--; \
    }

#define CIRCULAR_INCREMENT_BASE(x, maxVal, base) \
        { \
            x++; \
            if (x == maxVal) \
                x = base; \
        }
//avoid problems with unsigned. first compare and then give the nuw value
#define CIRCULAR_DECREMENT_BASE(x, maxVal, base) \
    { \
        if (x == base) \
            x = maxVal - 1; \
        else \
            x--; \
    }

#define EXTRACT_IDS(streamId, linkId) \
    { \
        linkId = (streamId >> 24) & 0XFF; \
        streamId = streamId & 0xFFFFFF; \
    }

#define COMBIN_IDS(streamId, linkid) \
         streamId = streamId | ((linkid & 0xFF) << 24);

#ifdef __cplusplus
}
#endif

#endif //_XLINK_TOOL_H
