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
    #ifndef __PC__
        #define ASSERT_X_LINK_PLATFORM(x)   if(!(x)) { exit(EXIT_FAILURE); }
        #define ASSERT_X_LINK_PLATFORM_R(x, r) ASSERT_X_LINK_PLATFORM(x)
    #else
        #define ASSERT_X_LINK_PLATFORM(x)   if(!(x)) { return X_LINK_PLATFORM_ERROR; }
        #define ASSERT_X_LINK_PLATFORM_R(x, r)   if(!(x)) { return r; }
    #endif
#else   // Debug configuration
    #ifndef __PC__
        #define ASSERT_X_LINK_PLATFORM(x)   if(!(x)) { fprintf(stderr, "%s:%d:\n Assertion Failed: %s\n", __FILE__, __LINE__, #x); exit(EXIT_FAILURE); }
        #define ASSERT_X_LINK_PLATFORM_R(x, r) ASSERT_X_LINK_PLATFORM(x)
    #else
        #define ASSERT_X_LINK_PLATFORM(x) \
        if(!(x)) { \
            mvLog(MVLOG_ERROR, "%s:%d\n\t Assertion Failed: %s", __FILE__, __LINE__, #x); \
            return X_LINK_PLATFORM_ERROR; \
        }
        #define ASSERT_X_LINK_PLATFORM_R(x, r) \
        if(!(x)) { \
            mvLog(MVLOG_ERROR, "%s:%d\n\t Assertion Failed: %s", __FILE__, __LINE__, #x); \
            return r; \
        }
    #endif
#endif //  NDEBUG

#ifdef __cplusplus
}
#endif

#endif //_XLINKPLATFORM_TOOL_H
