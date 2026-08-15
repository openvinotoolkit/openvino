// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ov::util {

// Returns |path| with separators converted to the platform-native form.
inline std::string make_path(const std::string& path) {
    std::string p = path;
#ifdef _WIN32
    for (auto& c : p)
        if (c == '/')
            c = '\\';
#endif
    return p;
}

// Writes |data| (|size| bytes) to |path|, throwing std::runtime_error on failure.
inline void save_binary(const std::string& path, const void* data, size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open())
        throw std::runtime_error("Failed to open file for writing: " + path);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    if (!f)
        throw std::runtime_error("Failed to write file: " + path);
}

// Reads the whole |path| into a byte buffer, throwing std::runtime_error on
// failure. This is the model-file entry point of the core: a serialized FB/PB
// blob on disk is loaded straight into the Vulkan IR without ov::Model.
inline std::vector<std::byte> load_binary(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.is_open())
        throw std::runtime_error("Failed to open file for reading: " + path);
    const auto size = f.tellg();
    if (size < 0)
        throw std::runtime_error("Failed to size file: " + path);
    std::vector<std::byte> data(static_cast<size_t>(size));
    f.seekg(0, std::ios::beg);
    f.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));
    if (!f && !data.empty())
        throw std::runtime_error("Failed to read file: " + path);
    return data;
}

}  // namespace ov::util

namespace ov::intel_gpu {

// Version of save_binary that don't trow an exception if attempt to open file fails
void save_binary(const std::string& path, const std::vector<uint8_t>& binary);

}  // namespace ov::intel_gpu