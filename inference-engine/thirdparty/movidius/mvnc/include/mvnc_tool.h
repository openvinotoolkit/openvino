#ifndef _MVNC_TOOL_H
#define _MVNC_TOOL_H

#ifdef __cplusplus
extern "C"
{
#endif

#ifndef CHECK_HANDLE_CORRECT
#define CHECK_HANDLE_CORRECT(handle)  {                             \
    if (!handle) {                                                  \
      mvLog(MVLOG_ERROR, "%s is NULL", #handle);                    \
      return NC_INVALID_HANDLE;                                     \
    }                                                               \
}
#endif  // CHECK_HANDLE_CORRECT


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

#ifndef CHECK_HANDLE_CORRECT_RC
#define CHECK_HANDLE_CORRECT_RC(handle, rc)  {                      \
    if (!handle) {                                                  \
      mvLog(MVLOG_ERROR, "%s is NULL", #handle);                    \
      return rc;                                                    \
    }                                                               \
}
#endif  // CHECK_HANDLE_CORRECT

#ifndef CHECK_HANDLE_CORRECT_WINFO
#define CHECK_HANDLE_CORRECT_WINFO(handle, logLevel, printMessage) {\
    if (!handle) {                                                  \
      mvLog(logLevel, "%s", printMessage);                          \
      return NC_INVALID_HANDLE;                                     \
    }                                                               \
}
#endif  // CHECK_HANDLE_CORRECT_WINFO

#ifdef __cplusplus
}
#endif

#endif //_MVNC_TOOL_H
