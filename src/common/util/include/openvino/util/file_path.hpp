// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>
#include <filesystem>

namespace ov::util {

// There are known issues with usage of std::filesystem::path unicode represenataion:
// * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=95048
// * https://stackoverflow.com/questions/58521857/cross-platform-way-to-handle-stdstring-stdwstring-with-stdfilesystempath
// Working compiler versions has been designated with godbolt.
using Path = std::filesystem::path;

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
