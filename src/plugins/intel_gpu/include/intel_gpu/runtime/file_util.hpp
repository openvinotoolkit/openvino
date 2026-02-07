// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <filesystem>

#include "openvino/util/file_util.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ov::intel_gpu {

// Version of save_binary that don't trow an exception if attempt to open file fails
void save_binary(const std::string& path, const std::vector<uint8_t>& binary);

// Safe version of load_binary that validates the read operation
// Returns empty vector if file doesn't exist, is empty, or read fails
std::vector<uint8_t> load_binary(const std::string& path);


// File-based lock for cross-process synchronization
class FileLock {
public:
    explicit FileLock(const std::string& path);

    ~FileLock();

    bool is_locked() const { return locked_; }

    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;

private:
    std::filesystem::path lock_path_;
    bool locked_;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};

}  // namespace ov::intel_gpu
