// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <filesystem>

class MemoryFile {
  public:
    /// Create a memory backed file
    MemoryFile(const void *data, size_t size);
    /// Delete memory backed file
    ~MemoryFile();

    /// Get path to a file.
    const char *name() { return m_name; }

  private:
    char *m_name;
};

struct ScopedRemove {
    std::filesystem::path a, b;
    ~ScopedRemove() {
        std::error_code ec;
        if (!a.empty()) std::filesystem::remove(a, ec);
        if (!b.empty()) std::filesystem::remove(b, ec);
    }
};

std::array<std::tuple<const uint8_t*, size_t>, 2> split_data(const uint8_t* data, size_t size, const uint8_t* delim, size_t delim_size);
const std::filesystem::path create_model_file(const uint8_t* data, size_t size, const char* ext);
std::tuple<std::filesystem::path, std::filesystem::path> create_ir_model_files(const uint8_t* data, size_t size, const uint8_t* delim, size_t delim_size);
