// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "ov_filesystem.hpp"

namespace ov {
namespace util {

namespace fs = std_fs;
using Path = fs::path;


auto File = [](std::FILE* file) {
    auto deleter = [](std::FILE* file) {
        std::fclose(file);
    };
    return std::unique_ptr<std::FILE, decltype(deleter)>{file, deleter};
};

}  // namespace util
}  // namespace ov
