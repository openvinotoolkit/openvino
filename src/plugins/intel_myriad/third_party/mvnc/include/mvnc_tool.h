// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#ifndef _MVNC_TOOL_H
#define _MVNC_TOOL_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef CHECK_HANDLE_CORRECT
#define CHECK_HANDLE_CORRECT(handle)                                \
do {                                                                \
    if (!handle) {                                                  \
      mvLog(MVLOG_ERROR, "%s is NULL", #handle);                    \
      return NC_INVALID_HANDLE;                                     \
    }                                                               \
} while (0)
#endif  // CHECK_HANDLE_CORRECT


#ifndef CHECK_MUTEX_SUCCESS
#define CHECK_MUTEX_SUCCESS(call)                                   \
do {                                                                \
    int error;                                                      \
    if ((error = (call))) {                                         \
      mvLog(MVLOG_ERROR, "%s failed with error: %d", #call, error); \
    }                                                               \
} while (0)
#endif  // CHECK_MUTEX_SUCCESS

#ifndef CHECK_MUTEX_SUCCESS_RC
#define CHECK_MUTEX_SUCCESS_RC(call, rc)                            \
do {                                                                \
    int error;                                                      \
    if ((error = (call))) {                                         \
      mvLog(MVLOG_ERROR, "%s failed with error: %d", #call, error); \
      return rc;                                                    \
    }                                                               \
} while (0)
#endif  // CHECK_MUTEX_SUCCESS_RC

#ifndef CHECK_HANDLE_CORRECT_RC
#define CHECK_HANDLE_CORRECT_RC(handle, rc) \
do {                                                                \
    if (!handle) {                                                  \
      mvLog(MVLOG_ERROR, "%s is NULL", #handle);                    \
      return rc;                                                    \
    }                                                               \
} while (0)
#endif  // CHECK_HANDLE_CORRECT

#ifndef CHECK_HANDLE_CORRECT_WINFO
#define CHECK_HANDLE_CORRECT_WINFO(handle, logLevel, printMessage)  \
do {                                                                \
    if (!handle) {                                                  \
      mvLog(logLevel, "%s", printMessage);                          \
      return NC_INVALID_HANDLE;                                     \
    }                                                               \
} while (0)
#endif  // CHECK_HANDLE_CORRECT_WINFO

#ifdef __cplusplus
}
#endif

#endif //_MVNC_TOOL_H
