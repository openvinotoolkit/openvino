// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for the OpenVINO Runtime common aliases and data types.
 *
 * @file openvino/runtime/common.hpp
 */
#pragma once

#include <map>
#include <string>

#include "openvino/core/any.hpp"
#include "openvino/core/visibility.hpp"

#if defined(OPENVINO_STATIC_LIBRARY) || defined(USE_STATIC_IE)
#    define OPENVINO_RUNTIME_API_C(...) OPENVINO_EXTERN_C __VA_ARGS__
#    define OPENVINO_RUNTIME_API
#else
#    ifdef IMPLEMENT_OPENVINO_RUNTIME_API  // defined if we are building the OpenVINO runtime DLL (instead of using it)
#        define OPENVINO_RUNTIME_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS __VA_ARGS__ OPENVINO_CDECL
#        define OPENVINO_RUNTIME_API        OPENVINO_CORE_EXPORTS
#    else
#        define OPENVINO_RUNTIME_API_C(...) OPENVINO_EXTERN_C OPENVINO_CORE_IMPORTS __VA_ARGS__ OPENVINO_CDECL
#        define OPENVINO_RUNTIME_API        OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_RUNTIME_API
#endif      // OPENVINO_STATIC_LIBRARY || USE_STATIC_IE

/**
 * @def OPENVINO_PLUGIN_API
 * @brief Defines the OpenVINO Runtime Plugin API method.
 */

#if defined(IMPLEMENT_OPENVINO_RUNTIME_PLUGIN)
#    define OPENVINO_PLUGIN_API OPENVINO_EXTERN_C OPENVINO_CORE_EXPORTS
#else
#    define OPENVINO_PLUGIN_API OPENVINO_EXTERN_C
#endif

namespace ov {

/**
 * @brief This type of map is used for result of Core::query_model
 * @ingroup ov_runtime_cpp_api
 *   - `key` means operation name
 *   - `value` means device name supporting this operation
 */
using SupportedOpsMap = std::map<std::string, std::string>;

#ifdef _WIN32
// Windows uses HANDLE (void*) for file handles
using FileHandle = void*;
#else
// Linux/Unix uses int for file descriptors
using FileHandle = int;
#endif

/**
 * @brief Type definition for file handle provider callback (cross-platform).
 * Function that takes no arguments and returns a platform-specific file handle.
 * The callback implementation must release ownership, caller should close the FileHandle.
 * On Linux/Unix: returns int (file descriptor)
 * On Windows: returns void* (HANDLE cast to void*)
 * This is useful for scenarios where file access needs to be controlled externally,
 * such as Android content providers or Windows restricted file access scenarios.
 * @ingroup ov_runtime_cpp_api
 */
using FileHandleProvider = std::function<FileHandle()>;

}  // namespace ov

#if defined(_WIN32) && !defined(__GNUC__)
#    define __PRETTY_FUNCTION__ __FUNCSIG__
#else
#    define __PRETTY_FUNCTION__ __PRETTY_FUNCTION__
#endif
