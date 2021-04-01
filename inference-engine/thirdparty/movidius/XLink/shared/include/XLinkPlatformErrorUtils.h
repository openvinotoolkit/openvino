// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _XLINKPLATFORM_TOOL_H
#define _XLINKPLATFORM_TOOL_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef NDEBUG  // Release configuration

    #ifndef ASSERT_XLINK_PLATFORM_R
    #define ASSERT_XLINK_PLATFORM_R(condition, err) do { \
            if(!(condition)) { \
                mvLog(MVLOG_ERROR, "Assertion Failed: %s \n", #condition); \
                return (err); \
            } \
        } while(0)
    #endif  // ASSERT_XLINK_PLATFORM_R

    #ifndef ASSERT_XLINK_PLATFORM
    #define ASSERT_XLINK_PLATFORM(condition) do { \
            \
            ASSERT_XLINK_PLATFORM_R((condition), X_LINK_PLATFORM_ERROR);\
            \
        } while(0)
    #endif  // ASSERT_XLINK_PLATFORM

#else   // Debug configuration

    #ifndef ASSERT_XLINK_PLATFORM_R
    #define ASSERT_XLINK_PLATFORM_R(condition, err) do { \
                if(!(condition)) { \
                    mvLog(MVLOG_ERROR, "Assertion Failed: %s \n", #condition); \
                    exit(EXIT_FAILURE); \
                } \
            } while(0)
    #endif  // ASSERT_XLINK_PLATFORM_R

    #ifndef ASSERT_XLINK_PLATFORM
    #define ASSERT_XLINK_PLATFORM(condition) do { \
            \
            ASSERT_XLINK_PLATFORM_R((condition), X_LINK_PLATFORM_ERROR);\
            \
        } while(0)
    #endif // ASSERT_XLINK_PLATFORM

#endif //  NDEBUG

#ifdef __cplusplus
}
#endif

#endif //_XLINKPLATFORM_TOOL_H
