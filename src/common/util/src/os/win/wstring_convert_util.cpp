// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/wstring_convert_util.hpp"

#include <cstdint>
#include <windows.h>

namespace ov::util {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
std::string wstring_to_string(const std::wstring_view wstr) {
    const auto wstr_size = static_cast<int>(wstr.size());
    const auto size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), wstr_size, NULL, 0, NULL, NULL);
    std::string result(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), wstr_size, result.data(), size_needed, NULL, NULL);
    return result;
}

std::wstring string_to_wstring(const std::string_view string) {
    const char* str = string.data();
    const auto str_size = static_cast<int>(string.size());
    const auto size_needed = MultiByteToWideChar(CP_UTF8, 0, str, str_size, NULL, 0);
    std::wstring result(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, str_size, result.data(), size_needed);
    return result;
}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
}  // namespace ov::util
