// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/core/core_visibility.hpp"

namespace ov {
namespace utils {

template <class C>
struct OPENVINO_API FileTraits;

template <>
struct FileTraits<char> {
    static constexpr const auto file_separator =
#ifdef _WIN32
        '\\';
#else
        '/';
#endif
    static constexpr const auto dot_symbol = '.';
    static std::string library_ext() {
#ifdef _WIN32
        return {"dll"};
#else
        return {"so"};
#endif
    }
    static std::string library_prefix() {
#ifdef _WIN32
        return {""};
#else
        return {"lib"};
#endif
    }
};

template <>
struct FileTraits<wchar_t> {
    static constexpr const auto file_separator =
#ifdef _WIN32
        L'\\';
#else
        L'/';
#endif
    static constexpr const auto dot_symbol = L'.';
    static std::wstring library_ext() {
#ifdef _WIN32
        return {L"dll"};
#else
        return {L"so"};
#endif
    }
    static std::wstring library_prefix() {
#ifdef _WIN32
        return {L""};
#else
        return {L"lib"};
#endif
    }
};

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
    return path + separator + FileTraits<C>::library_prefix() + input + FileTraits<C>::dot_symbol +
           FileTraits<C>::library_ext();
}

}  // namespace utils
}  // namespace ov

