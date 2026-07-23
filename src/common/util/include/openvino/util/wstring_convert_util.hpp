// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <string_view>

#include "openvino/util/util.hpp"

/// @brief Widens a narrow string literal to a native path literal.
/// On Windows (where std::filesystem::path::value_type is wchar_t) this prepends L.
/// On other platforms it is a no-op.
#ifdef _WIN32
#    define OV_WSTR_IMPL(s) L##s
#    define OV_WSTR(s)      OV_WSTR_IMPL(s)
#else
#    define OV_WSTR(s) s
#endif

namespace ov::util {

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Conversion from wide character string to a single-byte chain.
 * @param wstr A wide-char string
 * @return A multi-byte string
 */
std::string wstring_to_string(const std::wstring_view wstr);

/**
 * @brief Conversion from single-byte chain to wide character string.
 * @param str A multi-byte string
 * @return A wide-char string
 */
std::wstring string_to_wstring(const std::string_view str);

#endif

}  // namespace ov::util
