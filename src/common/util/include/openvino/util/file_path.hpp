// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>

#include "openvino/util/filesystem.hpp"
namespace ov {
namespace util {

#if defined(OPENVINO_HAS_FILESYSTEM)
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

}  // namespace util
}  // namespace ov
