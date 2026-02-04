// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/wstring_convert_util.hpp"

#include <cstdint>
#include <stdexcept>

#ifdef _WIN32
#    include <windows.h>
#endif

namespace ov::util {
#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

constexpr auto value_mask = 0x3FU;
constexpr auto codepoint_2nd_shift = 6U;
constexpr auto codepoint_3rd_shift = 12U;
constexpr auto codepoint_4th_shift = 18U;

std::string wstring_to_string(const std::wstring_view wstr) {
#    ifdef _WIN32
    const auto wstr_size = static_cast<int>(wstr.size());
    const auto size_needed = WideCharToMultiByte(CP_UTF8, 0, wstr.data(), wstr_size, NULL, 0, NULL, NULL);
    std::string result(size_needed, 0);
    WideCharToMultiByte(CP_UTF8, 0, wstr.data(), wstr_size, result.data(), size_needed, NULL, NULL);
#    else
    std::string result;
    result.reserve(wstr.size() * (sizeof(wchar_t) >= 4 ? 4 : 3));  // Worst case for UTF-8

    for (const auto& wc : wstr) {
        uint32_t codepoint = static_cast<uint32_t>(wc);

        if (codepoint <= 0x7FU) {
            // 1-byte sequence (ASCII)
            result.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7FFU) {
            // 2-byte sequence
            result.push_back(static_cast<char>(0xC0U | ((codepoint >> codepoint_2nd_shift) & 0x1FU)));
            result.push_back(static_cast<char>(0x80U | (codepoint & value_mask)));
        } else if (codepoint <= 0xFFFFU) {
            // 3-byte sequence
            result.push_back(static_cast<char>(0xE0U | ((codepoint >> codepoint_3rd_shift) & 0x0FU)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> codepoint_2nd_shift) & value_mask)));
            result.push_back(static_cast<char>(0x80U | (codepoint & value_mask)));
        } else if (codepoint <= 0x10FFFFU) {
            // 4-byte sequence
            result.push_back(static_cast<char>(0xF0U | ((codepoint >> codepoint_4th_shift) & 0x07U)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> codepoint_3rd_shift) & value_mask)));
            result.push_back(static_cast<char>(0x80U | ((codepoint >> codepoint_2nd_shift) & value_mask)));
            result.push_back(static_cast<char>(0x80U | (codepoint & value_mask)));
        } else {
            throw std::runtime_error("Invalid Unicode codepoint");
        }
    }
    result.shrink_to_fit();
#    endif
    return result;
}

std::wstring string_to_wstring(const std::string_view string) {
    const char* str = string.data();
#    ifdef _WIN32
    const auto str_size = static_cast<int>(string.size());
    const auto size_needed = MultiByteToWideChar(CP_UTF8, 0, str, str_size, NULL, 0);
    std::wstring result(size_needed, 0);
    MultiByteToWideChar(CP_UTF8, 0, str, str_size, result.data(), size_needed);
#    else

    const auto check_utf8_seq_size = [](const char* first, const char* last, const std::ptrdiff_t seq_size) {
        if (seq_size > std::distance(first, last)) {
            throw std::runtime_error("Invalid UTF-8 sequence");
        }
    };

    std::wstring result;
    result.reserve(string.size());
    for (const auto last = str + string.size(); str < last;) {
        auto codepoint = static_cast<uint32_t>(*str++);
        if (codepoint <= 0x7FU) {
            // 1-byte sequence, nothing to do
        } else if ((codepoint & 0xE0U) == 0xC0U) {
            // 2-byte sequence
            check_utf8_seq_size(str, last, 1);
            codepoint = (codepoint & 0x1FU) << codepoint_2nd_shift;
            codepoint |= (static_cast<unsigned char>(*str++) & value_mask);
        } else if ((codepoint & 0xF0U) == 0xE0U) {
            // 3-byte sequence
            check_utf8_seq_size(str, last, 2);
            codepoint = (codepoint & 0x0FU) << codepoint_3rd_shift;
            codepoint |= (static_cast<unsigned char>(*str++) & value_mask) << codepoint_2nd_shift;
            codepoint |= (static_cast<unsigned char>(*str++) & value_mask);
        } else if ((codepoint & 0xF8U) == 0xF0U) {
            // 4-byte sequence
            check_utf8_seq_size(str, last, 3);
            codepoint = (codepoint & 0x07U) << codepoint_4th_shift;
            codepoint |= (static_cast<unsigned char>(*str++) & value_mask) << codepoint_3rd_shift;
            codepoint |= (static_cast<unsigned char>(*str++) & value_mask) << codepoint_2nd_shift;
            codepoint |= (static_cast<unsigned char>(*str++) & value_mask);
        } else {
            throw std::runtime_error("Invalid UTF-8 byte");
        }

        result.push_back(static_cast<wchar_t>(codepoint));
    }

#    endif
    return result;
}
#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
}  // namespace ov::util
