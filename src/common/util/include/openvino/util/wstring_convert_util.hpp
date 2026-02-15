// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/util/util.hpp"

namespace ov::util {

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
/**
 * @brief Conversion from wide character string to a single-byte chain.
 * @param wstr A wide-char string
 * @return A multi-byte string
 */
std::string wstring_to_string(const std::wstring& wstr);
/**
 * @brief Conversion from single-byte chain to wide character string.
 * @param str A null-terminated string
 * @return A wide-char string
 */
std::wstring string_to_wstring(const std::string& str);

#endif

}  // namespace ov::util
