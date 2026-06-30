// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fcntl.h>
#include <unistd.h>
#ifdef __APPLE__
#    include <sys/fcntl.h>
#endif

#include "openvino/util/file_util.hpp"

namespace ov::util {

FileHandle open_file(const std::filesystem::path& path, FileMode mode) {
    // CVS-189123
    int flags = O_CLOEXEC;
    if (has_flag(mode, FileMode::READ)) {
        flags |= O_RDONLY;
    }
#if defined(O_DIRECT)
    if (has_flag(mode, FileMode::DIRECT)) {
        flags |= O_DIRECT;
    }
#endif
    FileHandle fd = ::open(path.c_str(), flags);
#if defined(__APPLE__) && defined(F_NOCACHE)
    // macOS has no O_DIRECT; use F_NOCACHE to disable the unified buffer cache.
    if (fd != invalid_handle && has_flag(mode, FileMode::DIRECT)) {
        ::fcntl(fd, F_NOCACHE, 1);
    }
#endif
    return fd;
}

void close_file(FileHandle handle) {
    // CVS-189123
    if (handle != invalid_handle) {
        ::close(handle);
    }
}

}  // namespace ov::util
