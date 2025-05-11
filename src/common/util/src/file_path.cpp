// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_path.hpp"

#include "openvino/util/wstring_convert_util.hpp"

#if defined(GCC_VER_LESS_THAN_12_3) || defined(CLANG_VER_LESS_THAN_17)

template <>
ov::util::Path::path(const std::wstring& source, ov::util::Path::format fmt)
    : path(ov::util::wstring_to_string(source), fmt) {}

#endif
