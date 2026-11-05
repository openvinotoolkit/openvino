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

std::optional<uint64_t> get_file_position(FileHandle handle) {
    if (handle == INVALID_HANDLE_VALUE || handle == nullptr) {
        return std::nullopt;
    }
    // A zero move relative to FILE_CURRENT reports the position without changing it.
    // Fails for non-seekable handles (pipes, consoles).
    LARGE_INTEGER distance{};
    LARGE_INTEGER position{};
    if (!SetFilePointerEx(handle, distance, &position, FILE_CURRENT) || position.QuadPart < 0) {
        return std::nullopt;
    }
    return static_cast<uint64_t>(position.QuadPart);
}

}  // namespace ov::util
