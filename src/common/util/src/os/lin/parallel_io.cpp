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

bool get_mmap_file_info(const void* addr, std::filesystem::path& out_path, std::streamoff& out_offset) {
    std::ifstream maps_file("/proc/self/maps");
    if (!maps_file.is_open())
        return false;
    const auto addr_val = reinterpret_cast<uintptr_t>(addr);
    std::string line;
    while (std::getline(maps_file, line)) {
        // Format: start-end perms offset dev inode [pathname]
        std::istringstream iss(line);
        std::string addr_range, perms, offset_str, dev, inode_str;
        if (!(iss >> addr_range >> perms >> offset_str >> dev >> inode_str))
            continue;
        const auto dash = addr_range.find('-');
        if (dash == std::string::npos)
            continue;
        uintptr_t range_start = 0, range_end = 0;
        try {
            range_start = static_cast<uintptr_t>(std::stoull(addr_range.substr(0, dash), nullptr, 16));
            range_end = static_cast<uintptr_t>(std::stoull(addr_range.substr(dash + 1), nullptr, 16));
        } catch (...) {
            continue;
        }
        if (addr_val < range_start || addr_val >= range_end)
            continue;
        std::string path;
        if (!(iss >> path) || path.empty() || path[0] != '/')
            return false;  // anonymous or special region
        out_path = path;
        std::streamoff map_offset = 0;
        try {
            map_offset = static_cast<std::streamoff>(std::stoull(offset_str, nullptr, 16));
        } catch (...) {
            return false;
        }
        out_offset = map_offset + static_cast<std::streamoff>(addr_val - range_start);
        return true;
    }
    return false;
}

void prefetch_memory(const void* addr, size_t size) {
    // madvise(2) requires addr to be page-aligned; round down and extend size.
    const uintptr_t base = reinterpret_cast<uintptr_t>(addr);
    const uintptr_t aligned_base = base & ~uintptr_t{4095};
    madvise(reinterpret_cast<void*>(aligned_base), size + (base - aligned_base), MADV_WILLNEED);
}

}  // namespace ov::util
