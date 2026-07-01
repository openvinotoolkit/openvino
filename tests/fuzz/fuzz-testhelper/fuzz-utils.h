// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <stddef.h>
#include <stdint.h>
#include <array>
#include <string_view>
#include <tuple>
#include <system_error>
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
    std::filesystem::path path;
    ~ScopedRemove() {
        std::error_code ec;
        if (!path.empty())
			std::filesystem::remove(path, ec);
    }
};


std::array<std::string_view, 2> split_data(std::string_view data, std::string_view delim);
std::filesystem::path create_model_file(const uint8_t* data, size_t size, const std::filesystem::path& ext);
std::tuple<std::filesystem::path, std::filesystem::path> create_ir_model_files(const uint8_t* data, size_t size, std::string_view delim);
