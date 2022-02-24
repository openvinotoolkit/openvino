// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINK_TOOL_H
#define _XLINK_TOOL_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef NDEBUG  // Release configuration

    #ifndef ASSERT_XLINK
    #define ASSERT_XLINK(condition) do { \
            if(!(condition)) { \
                mvLog(MVLOG_ERROR, "Assertion Failed: %s \n", #condition); \
                return X_LINK_ERROR; \
            } \
        } while(0)
    #endif  // ASSERT_XLINK

#else   // Debug configuration

    #ifndef ASSERT_XLINK
    #define ASSERT_XLINK(condition) do { \
            if(!(condition)) { \
                mvLog(MVLOG_ERROR, "Assertion Failed: %s \n", #condition); \
                exit(EXIT_FAILURE); \
            } \
        } while(0)
    #endif // ASSERT_XLINK

#endif //  NDEBUG


//--------------------------------------------------------------------
//----- Check an expression and return error if needed. Begin. -------
//--------------------------------------------------------------------

//----------------- Check any condition. ---------------------

#ifndef XLINK_RET_ERR_IF
#define XLINK_RET_ERR_IF(condition, err) do { \
        if ((condition)) { \
            mvLog(MVLOG_ERROR, "Condition failed: %s", #condition);\
            return (err); \
        } \
    } while(0)
#endif  // XLINK_RET_ERR_IF

#ifndef XLINK_RET_IF
#define XLINK_RET_IF(condition) do { \
        \
        XLINK_RET_ERR_IF((condition), X_LINK_ERROR);\
        \
    } while(0)
#endif  // XLINK_RET_IF

//------------ Check method's return value. ------------------

#ifndef XLINK_RET_IF_FAIL
#define XLINK_RET_IF_FAIL(call) do { \
        int rc; \
        if ((rc = (call))) { \
            mvLog(MVLOG_ERROR, " %s method call failed with an error: %d", #call, rc); \
            return rc; \
        } \
    } while(0)
#endif  // XLINK_RET_IF_FAIL


//-------------------------------------------------------------
//------- Check an expression and goto out if needed. ---------
//-------------------------------------------------------------

#ifndef XLINK_OUT_WITH_LOG_IF
#define XLINK_OUT_WITH_LOG_IF(condition, log) do { \
        if ((condition)) { \
            (log); \
            goto XLINK_OUT; \
        } \
    } while(0)
#endif  // XLINK_OUT_WITH_LOG_IF

#ifndef XLINK_OUT_IF
#define XLINK_OUT_IF(condition) do { \
        \
        XLINK_OUT_WITH_LOG_IF((condition), \
            mvLog(MVLOG_ERROR, "Condition failed: %s \n", #condition));\
        \
    } while(0)
#endif  // XLINK_OUT_IF

#ifdef __cplusplus
}
#endif

#endif //_XLINK_TOOL_H
