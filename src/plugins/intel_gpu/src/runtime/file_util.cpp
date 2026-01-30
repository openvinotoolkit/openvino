// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/file_util.hpp"
#include <stdexcept>
#include <fstream>

namespace ov::intel_gpu {

void save_binary(const std::string &path, const std::vector<uint8_t>& binary) {
    try {
        ov::util::save_binary(ov::util::make_path(path), binary.data(), binary.size());
    } catch (std::runtime_error&) {}
}

std::vector<uint8_t> load_binary(const std::string& path) {
    std::vector<uint8_t> buffer;

    // Check if file exists and get size
    auto file_path = ov::util::make_path(path);
    if (!ov::util::file_exists(file_path)) {
        return buffer;  // Return empty vector if file doesn't exist
    }

    try {
        auto file_size = std::filesystem::file_size(file_path);
        if (file_size == 0) {
            return buffer;  // Return empty vector for empty file
        }

        // Open and read file
        std::ifstream input(file_path, std::ios::binary);
        if (!input.is_open()) {
            return buffer;  // Return empty vector if can't open
        }

        buffer.resize(file_size);
        input.read(reinterpret_cast<char*>(buffer.data()), buffer.size());

        // Validate that we read the expected amount
        if (static_cast<size_t>(input.gcount()) != buffer.size()) {
            buffer.clear();  // Read failed, return empty vector
        }
    } catch (...) {
        buffer.clear();  // Any error, return empty vector
    }

    return buffer;
}

FileLock::FileLock(const std::string& path) : lock_path_(std::filesystem::path(path + ".lock")), locked_(false) {
#ifdef _WIN32
    // Use FILE_SHARE_READ | FILE_SHARE_WRITE to allow other processes to open the file
    // Then use LockFileEx for actual blocking lock
    handle_ = CreateFileW(lock_path_.c_str(), GENERIC_READ | GENERIC_WRITE,
                            FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_ALWAYS,
                            FILE_ATTRIBUTE_NORMAL, NULL);
    if (handle_ != INVALID_HANDLE_VALUE) {
        // Use LockFileEx with LOCKFILE_EXCLUSIVE_LOCK for blocking exclusive lock
        OVERLAPPED overlapped = {0};
        // Lock the first byte of the file (we just need any locked region)
        if (LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK, 0, 1, 0, &overlapped)) {
            locked_ = true;
        }
    }
#else
    fd_ = open(lock_path_.string().c_str(), O_CREAT | O_RDWR, 0666);
    if (fd_ >= 0) {
        if (flock(fd_, LOCK_EX) == 0) {
            locked_ = true;
        }
    }
#endif
}

FileLock::~FileLock() {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) {
        if (locked_) {
            // Unlock the region before closing
            OVERLAPPED overlapped = {0};
            UnlockFileEx(handle_, 0, 1, 0, &overlapped);
        }
        CloseHandle(handle_);
    }
#else
    if (fd_ >= 0) {
        flock(fd_, LOCK_UN);
        close(fd_);
    }
#endif
    // Don't delete the lock file - other processes may be using it
}

}  // namespace ov::intel_gpu
