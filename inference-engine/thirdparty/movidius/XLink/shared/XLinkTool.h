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
        #define ASSERT_X_LINK_R(x, r)   ASSERT_X_LINK(x)
    #else
    #define ASSERT_X_LINK(x) \
        if(!(x)) {  \
            mvLog(MVLOG_ERROR, "%s:%d\n\t Assertion Failed: %s", __FILE__, __LINE__, #x); \
            return X_LINK_ERROR; \
        }
    #define ASSERT_X_LINK_R(x, r) \
        if(!(x)) { \
            mvLog(MVLOG_ERROR, "%s:%d\n\t Assertion Failed: %s", __FILE__, __LINE__, #x); \
            return r; \
        }
    #endif

#endif //  NDEBUG

#ifndef XLINK_CHECK_CALL
#define XLINK_CHECK_CALL(call)  { \
        int error; \
        if ((error = (call))) { \
          mvLog(MVLOG_ERROR, "%s failed with error: %d", #call, error); \
        } \
    }
#endif  // XLINK_CHECK_CALL

#ifndef XLINK_RET_IF
#define XLINK_RET_IF(call)  { \
        int error; \
        if ((error = (call))) { \
          mvLog(MVLOG_ERROR, "%s failed with error: %d", #call, error); \
          return error; \
        } \
    }
#endif  // XLINK_RET_IF

#ifndef XLINK_RET_IF_RC
#define XLINK_RET_IF_RC(call, rc)  { \
        if ((call)) { \
          mvLog(MVLOG_ERROR, "%s expression failed", #call); \
          return rc; \
        } \
    }
#endif  // XLINK_RET_IF_RC

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
