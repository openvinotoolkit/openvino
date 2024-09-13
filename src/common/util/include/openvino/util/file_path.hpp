// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <experimental/filesystem>
#include <memory>

namespace ov {
namespace util {

namespace fs = std::experimental::filesystem;
using Path = fs::path;


auto File = [](std::FILE* file) {
    auto deleter = [](std::FILE* file) {
        std::fclose(file);
    };
    return std::unique_ptr<std::FILE, decltype(deleter)>{file, deleter};
};

}  // namespace util
}  // namespace ov
