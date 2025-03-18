// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>

#include "openvino/util/filesystem.hpp"

namespace ov::util {

#if defined(OPENVINO_HAS_FILESYSTEM)
// There are known issues with usage of std::filesystem::path unicode represenataion:
// * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
// * https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath
// Working compiler versions has been designated with godbolt.
using Path = std::filesystem::path;
#elif defined(OPENVINO_HAS_EXP_FILESYSTEM)
// Known issues:
// * error C2280: 'std::u32string std::experimental::filesystem::v1::path::u32string(void) const': attempting to
// * filesystem error: Cannot convert character sequence: Invalid in or incomplete multibyte or wide character

///
/// @typedef Path
/// @brief Alias for std::experimental::filesystem::path.
///
/// This alias is used to simplify the usage of filesystem paths.
///
/// @note The experimental version of std::filesystem::path may not support all features correctly.
/// It is recommended to use this alias with caution and consider upgrading to C++17 or higher
/// for full support of std::filesystem::path.
///
using Path = std::experimental::filesystem::path;
#endif

#if !defined(__GNUC__) || (__GNUC__ > 12 || __GNUC__ == 12 && __GNUC_MINOR__ >= 3)
#    define GCC_NOT_USED_OR_VER_AT_LEAST_12_3
#endif

#if !defined(__clang__) || defined(__clang__) && __clang_major__ >= 17
#    define CLANG_NOT_USED_OR_VER_AT_LEAST_17
#endif

#if defined(__GNUC__) && (__GNUC__ < 12 || __GNUC__ == 12 && __GNUC_MINOR__ < 3)
#    define GCC_VER_LESS_THAN_12_3
#endif

#if defined(__clang__) && __clang_major__ < 17
#    define CLANG_VER_LESS_THAN_17
#endif

}  // namespace ov::util

#if defined(GCC_VER_LESS_THAN_12_3) || defined(CLANG_VER_LESS_THAN_17)

template <>
ov::util::Path::path(const std::wstring& source, ov::util::Path::format fmt);

#endif
