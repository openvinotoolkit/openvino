// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/core_visibility.hpp"

namespace ov {
namespace utils {

template <class T>
struct OPENVINO_API FileTraits;
#ifdef _WIN32
template <>
struct ov::utils::FileTraits<char> {
    constexpr static const auto file_separator = '\\';
    constexpr static const auto dot_symbol = '.';
    static std::string plugin_library_prefix() {
        return {};
    }
    static std::string plugin_library_ext() {
        return {"dll"};
    }
};
template <>
struct ov::utils::FileTraits<wchar_t> {
    constexpr static const auto file_separator = L'\\';
    constexpr static const auto dot_symbol = L'.';
    static std::wstring plugin_library_prefix() {
        return {};
    }
    static std::wstring plugin_library_ext() {
        return {L"dll"};
    }
};
#elif defined __APPLE__
/// @brief File path separator
template <>
struct ov::utils::FileTraits<char> {
    constexpr static const auto file_separator = '/';
    constexpr static const auto dot_symbol = '.';
    static std::string plugin_library_prefix() {
        return {"lib"};
    }
    static std::string plugin_library_ext() {
        return {"so"};
    }
};
template <>
struct ov::utils::FileTraits<wchar_t> {
    constexpr static const auto file_separator = L'/';
    constexpr static const auto dot_symbol = L'.';
    static std::wstring plugin_library_prefix() {
        return {L"lib"};
    }
    static std::wstring plugin_library_ext() {
        return {L"so"};
    }
};
#else
/// @brief File path separator
template <>
struct ov::utils::FileTraits<char> {
    constexpr static const auto file_separator = '/';
    constexpr static const auto dot_symbol = '.';
    static std::string plugin_library_prefix() {
        return {"lib"};
    }
    static std::string plugin_library_ext() {
        return {"so"};
    }
};
template <>
struct ov::utils::FileTraits<wchar_t> {
    constexpr static const auto file_separator = L'/';
    constexpr static const auto dot_symbol = L'.';
    static std::wstring plugin_library_prefix() {
        return {L"lib"};
    }
    static std::wstring plugin_library_ext() {
        return {L"so"};
    }
};
#endif

/// \brief Conversion from wide character string to a single-byte chain.
/// \param wstr A wide-char string
/// \return A multi-byte string
OPENVINO_API std::string wstring_to_string(const std::wstring& wstr);

/// \brief Conversion from single-byte chain to wide character string.
/// \param str A null-terminated string
/// \return A wide-char string
OPENVINO_API std::wstring multi_byte_char_to_wstring(const char* str);

template <typename C,
          typename = typename std::enable_if<(std::is_same<C, char>::value || std::is_same<C, wchar_t>::value)>::type>
inline std::basic_string<C> make_plugin_library_name(const std::basic_string<C>& path,
                                                     const std::basic_string<C>& input) {
    std::basic_string<C> separator(1, FileTraits<C>::file_separator);
    if (path.empty())
        separator = {};
    return path + separator + FileTraits<C>::plugin_library_prefix() + input + FileTraits<C>::dot_symbol +
           FileTraits<C>::plugin_library_ext();
}

}  // namespace utils
}  // namespace ov

