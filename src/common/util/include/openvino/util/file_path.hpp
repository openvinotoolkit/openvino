// Copyright (C) 2018-2024 Intel Corporation
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
using Path = std::experimental::filesystem::path;
#endif

}  // namespace util
}  // namespace ov
