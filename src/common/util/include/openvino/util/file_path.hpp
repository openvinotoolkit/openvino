// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>

#include "openvino/util/filesystem.hpp"
#include "openvino/util/wstring_cast_util.hpp"

namespace ov {
namespace util {

#if defined(OPENVINO_HAS_FILESYSTEM)
// There are known issues related with usage of std::filesystem::path unocode represenataion:
// https://jira.devtools.intel.com/browse/CVS-160477
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

#if defined(GCC_VER_LESS_THEN_12_3) || defined(CLANG_VER_LESS_THEN_17)
inline ov::util::Path WPath(const std::wstring& wpath) {
    return {ov::util::wstring_to_string(wpath)};
}
#else
inline ov::util::Path WPath(const std::wstring& wpath) {
    return {wpath};
}
#endif

}  // namespace util
}  // namespace ov
