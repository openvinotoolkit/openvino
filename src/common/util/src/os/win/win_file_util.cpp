// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/file_util.hpp"

// clang-format off
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>
// clang-format on

namespace ov::util {

FileHandle open_file(const std::filesystem::path& path, FileMode mode) {
    // CVS-189123
    DWORD desired_access = 0;
    DWORD flags_and_attrs = FILE_ATTRIBUTE_NORMAL;

    if (mode_set(mode, FileMode::READ)) {
        desired_access |= GENERIC_READ;
    }
    if (mode_set(mode, FileMode::DIRECT)) {
        flags_and_attrs |= FILE_FLAG_NO_BUFFERING;
    }

    return CreateFileW(path.native().c_str(),
                       desired_access,
                       FILE_SHARE_READ | FILE_SHARE_WRITE,
                       nullptr,
                       OPEN_EXISTING,
                       flags_and_attrs,
                       nullptr);
}

void close_file(FileHandle handle) {
    // CVS-189123
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
}

}  // namespace ov::util
