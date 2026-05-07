// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fuzz-utils.h"

#include <stdexcept>
#include <string>
#include <array>
#include <tuple>
#include <atomic>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <system_error>

#ifndef _WIN32
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#endif


MemoryFile::MemoryFile(const void *data, size_t size) {
#ifdef _WIN32
    throw std::exception("MemoryFile is not implemented for Windows");
#else  // _WIN32
    m_name = strdup("/dev/shm/fuzz-XXXXXX");
    if (!m_name)
        throw std::bad_alloc();
    int fd = mkstemp(m_name);
    if (size) {
        size_t nbytes = write(fd, data, size);
        if (nbytes != size) {
            free(m_name);
            close(fd);
            throw std::runtime_error("Failed to write " + std::to_string(size) +
                                     " bytes to " + m_name);
        }
    }
    close(fd);
#endif  // _WIN32
}

MemoryFile::~MemoryFile() {
#ifndef _WIN32
    unlink(m_name);
    free(m_name);
#endif  // _WIN32
}


namespace fs = std::filesystem;

bool write_file(const fs::path& p, const uint8_t* buf, size_t len) {
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    if (len) out.write(reinterpret_cast<const char*>(buf),
                    static_cast<std::streamsize>(len));
    return out.good();
}

inline uint64_t next_id() {
    static std::atomic<uint64_t> id{0};
    return ++id;
}

fs::path& temp_root() {
    static fs::path root = []{
        fs::path dir = fs::temp_directory_path() / "openvino_read_model_fuzz";
        std::error_code ec;
        fs::create_directories(dir, ec);
        return ec ? fs::temp_directory_path() : dir;
    }();
    return root;
}


const fs::path create_model_file(const uint8_t* data, size_t size, const char* ext) {
    const auto id = next_id();
    const auto base = std::string("model_") + std::to_string(id);
    const fs::path path = temp_root() / (base + ext);
    if (!write_file(path, data, size)) {
        throw std::runtime_error("Write to file failed");
    }
    return path;
}


std::array<std::tuple<const uint8_t*, size_t>, 2> split_data(const uint8_t* data, size_t size, const uint8_t* delim, size_t delim_size) {
    std::array<std::tuple<const uint8_t*, size_t>, 2> result;
    size_t split = SIZE_MAX;
    for (size_t i = 0; i + delim_size <= size; ++i) {
        if (std::memcmp(data + i, delim, delim_size) == 0) {
            split = i;
            break;
        }
    }
    if (split == SIZE_MAX)
        throw std::runtime_error("Problem detected during spliting fuzzer input data");

    const uint8_t* data0 = data;
    size_t data0_size = split;
    result[0] = {data0, data0_size};
    const uint8_t* data1 = data + split + delim_size;
    size_t data1_size = size - (split + delim_size);
    result[1] = {data1, data1_size};
    return result;
}


std::tuple<fs::path, fs::path> create_ir_model_files(const uint8_t* data, size_t size, const uint8_t* delim, size_t delim_size) {

    std::array<std::tuple<const uint8_t*, size_t>, 2> fuzz_data = split_data(data, size, delim, delim_size);

    const uint8_t* xml_data = std::get<0>(fuzz_data[0]);
    size_t xml_size = std::get<1>(fuzz_data[0]);
    const uint8_t* bin_data = std::get<0>(fuzz_data[1]);
    size_t bin_size = std::get<1>(fuzz_data[1]);

    const auto id = next_id();
    const auto base = std::string("model_") + std::to_string(id);
    const fs::path xml_path = temp_root() / (base + ".xml");
    const fs::path bin_path = temp_root() / (base + ".bin");

    if (!write_file(xml_path, xml_data, xml_size)) {
        throw std::runtime_error("Write to xml file failed");
    }

    if (!write_file(bin_path, bin_data, bin_size)) {
        throw std::runtime_error("Write to bin file failed");
    }

    return {xml_path, bin_path};
}
