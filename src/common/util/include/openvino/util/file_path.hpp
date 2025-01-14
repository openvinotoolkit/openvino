// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdio>

#include "openvino/util/ov_filesystem.hpp"
//#include "openvino/util/util.hpp"

namespace ov {
namespace util {

namespace fs = std_fs;
using Path = fs::path;

// auto deleter = [](std::FILE* file) {
//     std::fclose(file);
// };

// std::unique_ptr<std::FILE, decltype(deleter)> File(std::FILE* file){
//     return {file, deleter};
// }

}  // namespace util
}  // namespace ov
