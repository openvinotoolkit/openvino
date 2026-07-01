// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "fuzz-utils.h"

#include <stdexcept>
#include <string>
#include <atomic>
#include <fstream>

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


bool write_file(const std::filesystem::path& p, std::string_view data) {
    std::ofstream out(p, std::ios::binary);
    if (!out) return false;
    if (!data.empty())
        out.write(data.data(), static_cast<std::streamsize>(data.size()));
    return out.good();
}

uint64_t next_id() {
    static std::atomic<uint64_t> id{0};
    return ++id;
}

std::filesystem::path& temp_root() {
    static std::filesystem::path root = []{
        std::filesystem::path dir = std::filesystem::temp_directory_path() / "openvino_read_model_fuzz";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        return ec ? std::filesystem::temp_directory_path() : dir;
    }();
    return root;
}


std::filesystem::path create_model_file(const uint8_t* data, size_t size, const std::filesystem::path& ext) {
    auto name = std::filesystem::path("model_" + std::to_string(next_id()));
    name += ext;
    const auto path = temp_root() / name;
    if (!write_file(path, std::string_view(reinterpret_cast<const char*>(data), size))) {
        throw std::runtime_error("Write to file failed");
    }
	return path;
}


std::array<std::string_view, 2> split_data(std::string_view data, std::string_view delim) {
    const auto pos = data.find(delim);
    if (pos == std::string_view::npos)
        throw std::runtime_error("Problem detected during splitting fuzzer input data");
    return {data.substr(0, pos), data.substr(pos + delim.size())};
}


std::tuple<std::filesystem::path, std::filesystem::path> create_ir_model_files(const uint8_t* data, size_t size, std::string_view delim) {
    const auto [xml_sv, bin_sv] = split_data({reinterpret_cast<const char*>(data), size}, delim);
    const auto stem = std::string("model_") + std::to_string(next_id());
    const auto xml_path = temp_root() / (stem + ".xml");
    const auto bin_path = temp_root() / (stem + ".bin");
    if (!write_file(xml_path, xml_sv))
        throw std::runtime_error("Write to xml file failed");
    if (!write_file(bin_path, bin_sv))
        throw std::runtime_error("Write to bin file failed");
    return {xml_path, bin_path};
}