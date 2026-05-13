// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/parallel_io.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "openvino/util/file_util.hpp"

namespace ov::util {

void get_file_handle_and_size(const std::filesystem::path& path,
                              std::streamoff file_offset,
                              FileHandle& out_handle,
                              std::streamoff& out_size) {
    out_handle = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (out_handle == -1) {
        throw std::runtime_error("Cannot open file: " + ov::util::path_to_string(path));
    }
    struct stat st = {};
    if (::fstat(out_handle, &st) != 0) {
        ::close(out_handle);
        out_handle = -1;
        throw std::runtime_error("Cannot stat file: " + ov::util::path_to_string(path));
    }
    out_size = static_cast<std::streamoff>(st.st_size);
    if (file_offset < 0 || file_offset > out_size) {
        ::close(out_handle);
        out_handle = -1;
        throw std::out_of_range("header_offset is out of range for file: " + ov::util::path_to_string(path));
    }
}

void close_file_handle(FileHandle handle) {
    if (handle != -1) {
        ::close(handle);
    }
}

FileHandle open_file_for_read(const std::filesystem::path& path) {
    return ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
}

bool positional_read(FileHandle handle, char* dst, size_t size, size_t file_offset) {
    char* cur = dst;
    size_t remaining = size;
    off_t cur_offset = static_cast<off_t>(file_offset);
    while (remaining > 0) {
        const ssize_t n = ::pread(handle, cur, remaining, cur_offset);
        if (n <= 0) {
            return false;
        }
        cur += n;
        cur_offset += n;
        remaining -= static_cast<size_t>(n);
    }
    return true;
}

}  // namespace ov::util
