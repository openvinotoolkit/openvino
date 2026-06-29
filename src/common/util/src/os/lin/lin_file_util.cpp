// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fcntl.h>
#include <unistd.h>

#include "openvino/util/file_util.hpp"

namespace ov::util {

FileHandle open_file(const std::filesystem::path& path, FileMode mode) {
    // CVS-189123
    int flags = O_CLOEXEC;
    if (has_flag(mode, FileMode::READ)) {
        flags |= O_RDONLY;
    }
    if (has_flag(mode, FileMode::DIRECT)) {
        flags |= O_DIRECT;
    }
    return ::open(path.c_str(), flags);
}

void close_file(FileHandle handle) {
    // CVS-189123
    if (handle != -1) {
        ::close(handle);
    }
}

}  // namespace ov::util
