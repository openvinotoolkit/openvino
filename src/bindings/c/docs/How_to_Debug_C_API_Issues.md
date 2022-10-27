# How to Debug C API Issues

C API provides exception handling, here is all possible return values of the functions:
```ruby
typedef enum {
    OK = 0,
    /*
     * @brief map exception to C++ interface
     */
    GENERAL_ERROR = -1,
    NOT_IMPLEMENTED = -2,
    NETWORK_NOT_LOADED = -3,
    PARAMETER_MISMATCH = -4,
    NOT_FOUND = -5,
    OUT_OF_BOUNDS = -6,
    /*
     * @brief exception not of std::exception derived type was thrown
     */
    UNEXPECTED = -7,
    REQUEST_BUSY = -8,
    RESULT_NOT_READY = -9,
    NOT_ALLOCATED = -10,
    INFER_NOT_STARTED = -11,
    NETWORK_NOT_READ = -12,
    INFER_CANCELLED = -13,
    /*
     * @brief exception in C wrapper
     */
    INVALID_C_PARAM = -14,
    UNKNOWN_C_ERROR = -15,
    NOT_IMPLEMENT_C_METHOD = -16,
    UNKNOW_EXCEPTION = -17,
} ov_status_e;
```

As known, C API is a wrapping for C++ API, the main issues we had met can be classified into two kinds. Issue from input parameters checking and from C++ call exception.
* for parameter checking issue: return value is -14, please check the input parameters
* for C++ call exception issue: C interface just returns the status value, no more detail info message. If you want details please print it in exception macro.

> **NOTE**: The exception from C interface is not the same as C++ exception, do not use the C status value in C++ debuging.


