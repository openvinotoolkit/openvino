// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file with functions related to filesystem operations
 * 
 * @file os_filesystem.hpp
 */
#pragma once

#include <ie_api.h>

#ifdef ENABLE_UNICODE_PATH_SUPPORT
#include <codecvt>
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
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
}

/**
 * @brief Conversion from single-byte chain to wide character string.
 */
inline const std::wstring multiByteCharToWString(const char* str) {
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
}

#endif  // ENABLE_UNICODE_PATH_SUPPORT

}  // namespace details
}  // namespace InferenceEngine
