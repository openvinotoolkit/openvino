// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file with functions related to filesystem operations.
 * @file os_filesystem.h
 */
#pragma once

#ifdef ENABLE_UNICODE_PATH_SUPPORT
#include <string>
#include <codecvt>
#include <locale>

namespace InferenceEngine {
namespace details {

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
    std::wstring_convert<std::codecvt_utf8 <wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
}

}  // namespace details
}  // namespace InferenceEngine

#endif
