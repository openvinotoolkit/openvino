// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_npu/utils/zero/zero_result.hpp"

#include <ze_api.h>

#include <string>

namespace intel_npu {

const std::string ze_result_to_string(const ze_result_t result) {
    std::string as_string = {};

    switch (result) {
    case ZE_RESULT_SUCCESS:
        as_string = "ZE_RESULT_SUCCESS";
        break;
    case ZE_RESULT_NOT_READY:
        as_string = "ZE_RESULT_NOT_READY";
        break;
    case ZE_RESULT_ERROR_DEVICE_LOST:
        as_string = "ZE_RESULT_ERROR_DEVICE_LOST";
        break;
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        as_string = "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
        break;
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        as_string = "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
        break;
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
        as_string = "ZE_RESULT_ERROR_MODULE_BUILD_FAILURE";
        break;
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
        as_string = "ZE_RESULT_ERROR_MODULE_LINK_FAILURE";
        break;
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        as_string = "ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET";
        break;
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        as_string = "ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE";
        break;
    case ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX:
        as_string = "ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX";
        break;
    case ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE:
        as_string = "ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE";
        break;
    case ZE_RESULT_EXP_ERROR_REMOTE_DEVICE:
        as_string = "ZE_RESULT_EXP_ERROR_REMOTE_DEVICE";
        break;
    // *** Defination not found in enum _ze_result_t level_zero/ze_api.h ***
    // case ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE:
    //     as_string = "ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE";
    //     break;
    // case ZE_RESULT_EXP_RTAS_BUILD_RETRY:
    //     as_string = "ZE_RESULT_EXP_RTAS_BUILD_RETRY";
    //     break;
    // case ZE_RESULT_EXP_RTAS_BUILD_DEFERRED:
    //     as_string = "ZE_RESULT_EXP_RTAS_BUILD_DEFERRED";
    //     break;
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
        as_string = "ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS";
        break;
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
        as_string = "ZE_RESULT_ERROR_NOT_AVAILABLE";
        break;
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
        as_string = "ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE";
        break;
    case ZE_RESULT_WARNING_DROPPED_DATA:
        as_string = "ZE_RESULT_WARNING_DROPPED_DATA";
        break;
    case ZE_RESULT_ERROR_UNINITIALIZED:
        as_string = "ZE_RESULT_ERROR_UNINITIALIZED";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
        break;
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        as_string = "ZE_RESULT_ERROR_INVALID_ARGUMENT";
        break;
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
        as_string = "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
        break;
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        as_string = "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
        break;
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
        as_string = "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
        break;
    case ZE_RESULT_ERROR_INVALID_SIZE:
        as_string = "ZE_RESULT_ERROR_INVALID_SIZE";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT";
        break;
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
        as_string = "ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT";
        break;
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
        as_string = "ZE_RESULT_ERROR_INVALID_ENUMERATION";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        as_string = "ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT";
        break;
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
        as_string = "ZE_RESULT_ERROR_INVALID_NATIVE_BINARY";
        break;
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
        as_string = "ZE_RESULT_ERROR_INVALID_GLOBAL_NAME";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_NAME";
        break;
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
        as_string = "ZE_RESULT_ERROR_INVALID_FUNCTION_NAME";
        break;
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        as_string = "ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION";
        break;
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        as_string = "ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        as_string = "ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE";
        break;
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
        as_string = "ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED";
        break;
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
        as_string = "ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE";
        break;
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
        as_string = "ZE_RESULT_ERROR_OVERLAPPING_REGIONS";
        break;
    case ZE_RESULT_WARNING_ACTION_REQUIRED:
        as_string = "ZE_RESULT_WARNING_ACTION_REQUIRED";
        break;
    case ZE_RESULT_ERROR_UNKNOWN:
        as_string = "ZE_RESULT_ERROR_UNKNOWN";
        break;
    case ZE_RESULT_FORCE_UINT32:
        as_string = "ZE_RESULT_FORCE_UINT32";
        break;
    default:
        as_string = "ze_result_t Unrecognized";
        break;
    }

    return as_string;
}

const std::string ze_result_to_description(const ze_result_t result) {
    std::string as_string = {};

    switch (result) {
    case ZE_RESULT_SUCCESS:
        as_string = "success";
        break;
    case ZE_RESULT_NOT_READY:
        as_string = "synchronization primitive not signaled";
        break;
    case ZE_RESULT_ERROR_DEVICE_LOST:
        as_string = "device hung, reset, was removed, or driver update occurred";
        break;
    case ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
        as_string = "insufficient host memory to satisfy call";
        break;
    case ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
        as_string = "insufficient device memory to satisfy call";
        break;
    case ZE_RESULT_ERROR_MODULE_BUILD_FAILURE:
        as_string = "error occurred when building module, see build log for details";
        break;
    case ZE_RESULT_ERROR_MODULE_LINK_FAILURE:
        as_string = "error occurred when linking modules, see build log for details";
        break;
    case ZE_RESULT_ERROR_DEVICE_REQUIRES_RESET:
        as_string = "device requires a reset";
        break;
    case ZE_RESULT_ERROR_DEVICE_IN_LOW_POWER_STATE:
        as_string = "device currently in low power state";
        break;
    case ZE_RESULT_EXP_ERROR_DEVICE_IS_NOT_VERTEX:
        as_string = "device is not represented by a fabric vertex";
        break;
    case ZE_RESULT_EXP_ERROR_VERTEX_IS_NOT_DEVICE:
        as_string = "fabric vertex does not represent a device";
        break;
    case ZE_RESULT_EXP_ERROR_REMOTE_DEVICE:
        as_string = "fabric vertex represents a remote device or subdevice";
        break;
    // *** Defination not found in enum _ze_result_t level_zero/ze_api.h ***
    // case ZE_RESULT_EXP_ERROR_OPERANDS_INCOMPATIBLE:
    //     as_string = "operands of comparison are not compatible";
    //     break;
    // case ZE_RESULT_EXP_RTAS_BUILD_RETRY:
    //     as_string = "ray tracing acceleration structure build operation failed due to insufficient resources, retry "
    //                 "with a larger acceleration structure buffer allocation ";
    //     break;
    // case ZE_RESULT_EXP_RTAS_BUILD_DEFERRED:
    //     as_string = "ray tracing acceleration structure build operation deferred to parallel operation join ";
    //     break;
    case ZE_RESULT_ERROR_INSUFFICIENT_PERMISSIONS:
        as_string = "access denied due to permission level";
        break;
    case ZE_RESULT_ERROR_NOT_AVAILABLE:
        as_string = "resource already in use and simultaneous access not allowed or resource was removed";
        break;
    case ZE_RESULT_ERROR_DEPENDENCY_UNAVAILABLE:
        as_string = "external required dependency is unavailable or missing";
        break;
    case ZE_RESULT_WARNING_DROPPED_DATA:
        as_string = "data may have been dropped";
        break;
    case ZE_RESULT_ERROR_UNINITIALIZED:
        as_string = "driver is not initialized";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
        as_string = "generic error code for unsupported versions";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
        as_string = "generic error code for unsupported features";
        break;
    case ZE_RESULT_ERROR_INVALID_ARGUMENT:
        as_string = "generic error code for invalid arguments";
        break;
    case ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
        as_string = "handle argument is not valid";
        break;
    case ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
        as_string = "object pointed to by handle still in-use by device";
        break;
    case ZE_RESULT_ERROR_INVALID_NULL_POINTER:
        as_string = "pointer argument may not be nullptr";
        break;
    case ZE_RESULT_ERROR_INVALID_SIZE:
        as_string = "size argument is invalid (e.g., must not be zero)";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
        as_string = "size argument is not supported by the device (e.g., too large)";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_ALIGNMENT:
        as_string = "alignment argument is not supported by the device (e.g., too small)";
        break;
    case ZE_RESULT_ERROR_INVALID_SYNCHRONIZATION_OBJECT:
        as_string = "synchronization object in invalid state";
        break;
    case ZE_RESULT_ERROR_INVALID_ENUMERATION:
        as_string = "enumerator argument is not valid";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_ENUMERATION:
        as_string = "enumerator argument is not supported by the device";
        break;
    case ZE_RESULT_ERROR_UNSUPPORTED_IMAGE_FORMAT:
        as_string = "image format is not supported by the device";
        break;
    case ZE_RESULT_ERROR_INVALID_NATIVE_BINARY:
        as_string = "native binary is not supported by the device";
        break;
    case ZE_RESULT_ERROR_INVALID_GLOBAL_NAME:
        as_string = "global variable is not found in the module";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_NAME:
        as_string = "kernel name is not found in the module";
        break;
    case ZE_RESULT_ERROR_INVALID_FUNCTION_NAME:
        as_string = "function name is not found in the module";
        break;
    case ZE_RESULT_ERROR_INVALID_GROUP_SIZE_DIMENSION:
        as_string = "group size dimension is not valid for the kernel or device";
        break;
    case ZE_RESULT_ERROR_INVALID_GLOBAL_WIDTH_DIMENSION:
        as_string = "global width dimension is not valid for the kernel or device";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_INDEX:
        as_string = "kernel argument index is not valid for kernel";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ARGUMENT_SIZE:
        as_string = "kernel argument size does not match kernel";
        break;
    case ZE_RESULT_ERROR_INVALID_KERNEL_ATTRIBUTE_VALUE:
        as_string = "value of kernel attribute is not valid for the kernel or device";
        break;
    case ZE_RESULT_ERROR_INVALID_MODULE_UNLINKED:
        as_string = "module with imports needs to be linked before kernels can be created from it";
        break;
    case ZE_RESULT_ERROR_INVALID_COMMAND_LIST_TYPE:
        as_string = "command list type does not match command queue type";
        break;
    case ZE_RESULT_ERROR_OVERLAPPING_REGIONS:
        as_string = "copy operations do not support overlapping regions of memory";
        break;
    case ZE_RESULT_WARNING_ACTION_REQUIRED:
        as_string = "an action is required to complete the desired operation";
        break;
    case ZE_RESULT_ERROR_UNKNOWN:
        as_string = "an action is required to complete the desired operation";
        break;
    case ZE_RESULT_FORCE_UINT32:
        as_string = "FORCE UINT32 (error converting type to uint32)";
        break;
    default:
        as_string = "ze_result_t Unrecognized";
        break;
    }

    return as_string;
}

}  // namespace intel_npu
