// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file with functions related to filesystem operations
 * 
 * @file os_filesystem.hpp
 */
#pragma once

#include "ie_api.h"

#ifdef _WIN32
# ifndef NOMINMAX
#  define NOMINMAX
# endif
# ifndef _WINSOCKAPI_
#  define _WINSOCKAPI_
# endif
# include <Windows.h>
#elif defined ENABLE_UNICODE_PATH_SUPPORT
# include <codecvt>
#endif

#include <locale>
#include <string>

namespace InferenceEngine {
namespace details {

template<typename C>
using enableIfSupportedChar = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type;

#ifdef ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Conversion from wide character string to a single-byte chain.
 */
inline const std::string wStringtoMBCSstringChar(const std::wstring& wstr) {
#ifdef _WIN32
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);  // NOLINT
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);  // NOLINT
    return strTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
#endif
}

/**
 * @brief Conversion from single-byte chain to wide character string.
 */
inline const std::wstring multiByteCharToWString(const char* str) {
#ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#endif
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

}  // namespace details
}  // namespace InferenceEngine
