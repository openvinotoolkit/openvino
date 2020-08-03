// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file with functions related to filesystem operations
 * 
 * @file os_filesystem.hpp
 */
#pragma once

#include <string>

#include "ie_api.h"

namespace InferenceEngine {
namespace details {

template <typename C>
using enableIfSupportedChar = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type;

#ifdef ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Conversion from wide character string to a single-byte chain.
 * @param wstr A wide-char string
 * @return A multi-byte string
 */
INFERENCE_ENGINE_API_CPP(std::string) wStringtoMBCSstringChar(const std::wstring& wstr);

/**
 * @brief Conversion from single-byte chain to wide character string.
 * @param str A null-terminated string
 * @return A wide-char string
 */
INFERENCE_ENGINE_API_CPP(std::wstring) multiByteCharToWString(const char* str);

#endif  // ENABLE_UNICODE_PATH_SUPPORT

}  // namespace details
}  // namespace InferenceEngine
