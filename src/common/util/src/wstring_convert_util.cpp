// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/wstring_convert_util.hpp"

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

#    include <codecvt>
#    include <locale>

#    ifdef _WIN32
#        include <windows.h>
#    endif

#    ifdef __clang__
#        pragma clang diagnostic push
#        pragma clang diagnostic ignored "-Wdeprecated-declarations"
#    endif

std::string ov::util::wstring_to_string(const std::wstring& wstr) {
#    ifdef _WIN32
    int size_needed = WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), NULL, 0, NULL, NULL);
    std::string strTo(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, &wstr[0], (int)wstr.size(), &strTo[0], size_needed, NULL, NULL);
    return strTo;
#    else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_decoder;
    return wstring_decoder.to_bytes(wstr);
#    endif
}

std::wstring ov::util::string_to_wstring(const std::string& string) {
    const char* str = string.c_str();
#    ifdef _WIN32
    int strSize = static_cast<int>(std::strlen(str));
    int size_needed = MultiByteToWideChar(CP_UTF8, 0, str, strSize, NULL, 0);
    std::wstring wstrTo(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, strSize, &wstrTo[0], size_needed);
    return wstrTo;
#    else
    std::wstring_convert<std::codecvt_utf8<wchar_t>> wstring_encoder;
    std::wstring result = wstring_encoder.from_bytes(str);
    return result;
#    endif
}

#    ifdef __clang__
#        pragma clang diagnostic pop
#    endif

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
