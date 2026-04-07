// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/parallel_io.hpp"

// clang-format off
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
// clang-format on

#include <cstdint>
#include <stdexcept>

#include "openvino/util/file_util.hpp"

namespace ov::util {

void get_file_handle_and_size(const std::filesystem::path& path,
                              std::streamoff file_offset,
                              FileHandle& out_handle,
                              std::streamoff& out_size) {
    out_handle = CreateFileW(path.c_str(),
                             GENERIC_READ,
                             FILE_SHARE_READ | FILE_SHARE_WRITE,
                             nullptr,
                             OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL,
                             nullptr);
    if (out_handle == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Cannot open file: " + ov::util::path_to_string(path));
    }
    LARGE_INTEGER file_size = {};
    if (!GetFileSizeEx(out_handle, &file_size)) {
        CloseHandle(out_handle);
        out_handle = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Cannot get file size: " + ov::util::path_to_string(path));
    }
    out_size = static_cast<std::streamoff>(file_size.QuadPart);
    if (file_offset < 0 || file_offset > out_size) {
        CloseHandle(out_handle);
        out_handle = INVALID_HANDLE_VALUE;
        throw std::out_of_range("header_offset is out of range for file: " + ov::util::path_to_string(path));
    }
}

void close_file_handle(FileHandle handle) {
    if (handle != INVALID_HANDLE_VALUE) {
        CloseHandle(handle);
    }
}

FileHandle open_file_for_read(const std::filesystem::path& path) {
    return CreateFileW(path.native().c_str(),
                       GENERIC_READ,
                       FILE_SHARE_READ | FILE_SHARE_WRITE,
                       nullptr,
                       OPEN_EXISTING,
                       FILE_ATTRIBUTE_NORMAL,
                       nullptr);
}

bool positional_read(FileHandle handle, char* dst, size_t size, size_t file_offset) {
    char* cur = dst;
    size_t remaining = size;
    size_t cur_offset = file_offset;
    while (remaining > 0) {
        const DWORD to_read = static_cast<DWORD>(std::min(remaining, static_cast<size_t>(UINT_MAX - 1024u)));
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(cur_offset);
        if (!SetFilePointerEx(handle, li, nullptr, FILE_BEGIN)) {
            return false;
        }
        DWORD bytes_read = 0;
        if (!ReadFile(handle, cur, to_read, &bytes_read, nullptr)) {
            return false;
        }
        if (bytes_read == 0) {
            return false;
        }
        cur += bytes_read;
        cur_offset += bytes_read;
        remaining -= bytes_read;
    }
    return true;
}

}  // namespace ov::util