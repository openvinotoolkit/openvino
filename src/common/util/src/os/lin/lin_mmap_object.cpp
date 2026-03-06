// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace util {
size_t get_system_page_size() {
    static auto page_size = static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
    return page_size;
}
}  // namespace util

class HandleHolder {
    int m_handle = -1;
    void reset() noexcept {
        if (m_handle != -1) {
            close(m_handle);
            m_handle = -1;
        }
    }

public:
    explicit HandleHolder(int handle = -1) : m_handle(handle) {}

    HandleHolder(const HandleHolder&) = delete;
    HandleHolder& operator=(const HandleHolder&) = delete;

    HandleHolder(HandleHolder&& other) noexcept : m_handle(other.m_handle) {
        other.m_handle = -1;
    }

    HandleHolder& operator=(HandleHolder&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        m_handle = other.m_handle;
        other.m_handle = -1;
        return *this;
    }

    ~HandleHolder() {
        reset();
    }

    int get() const noexcept {
        return m_handle;
    }
};

class MapHolder final : public MappedMemory {
    void* m_data = MAP_FAILED;
    size_t m_size = 0;
    uint64_t m_id = std::numeric_limits<uint64_t>::max();
    HandleHolder m_handle;

public:
    MapHolder() = default;

    void set(const std::filesystem::path& path) {
        int mode = O_RDONLY;
        int fd = open(path.c_str(), mode);
        if (fd == -1) {
            throw std::runtime_error("Can not open file " + util::path_to_string(path) +
                                     " for mapping. Ensure that file exists and has appropriate permissions");
        }
        set_from_fd(fd);
        m_id = std::hash<std::string>{}(path.native());
    }

    void set_from_fd(const int fd) {
        int prot = PROT_READ;
        struct stat sb = {};
        m_handle = HandleHolder(fd);
        if (fstat(fd, &sb) == -1) {
            throw std::runtime_error("Can not get file size for fd=" + std::to_string(fd));
        }
        m_size = sb.st_size;
        if (m_size > 0) {
            m_data = mmap(nullptr, m_size, prot, MAP_SHARED, fd, 0);
            if (m_data == MAP_FAILED) {
                throw std::runtime_error("Can not create file mapping for " + std::to_string(fd) +
                                         ", err=" + std::strerror(errno));
            }
        } else {
            m_data = MAP_FAILED;
        }
    }

    uint64_t get_id() const noexcept override {
        return m_id;
    }

    ~MapHolder() {
        if (m_data != MAP_FAILED) {
            munmap(m_data, m_size);
        }
    }

    char* data() noexcept override {
        return static_cast<char*>(m_data);
    }

    size_t size() const noexcept override {
        return m_size;
    }
};

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::filesystem::path& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return holder;
}

std::shared_ptr<ov::MappedMemory> load_mmap_object_from_handle(FileHandle handle) {
    // On Linux, FileHandle is int (file descriptor)
    auto holder = std::make_shared<MapHolder>();
    holder->set_from_fd(static_cast<int>(handle));
    return holder;
}

class PartialMapHolder final : public MappedMemory {
    void* m_data = MAP_FAILED;
    size_t m_size = 0;
    uint64_t m_id = std::numeric_limits<uint64_t>::max();
    HandleHolder m_handle;

    void setup_mapping(const int fd, size_t pos, size_t size) {
        if (size > 0) {
            m_data = mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, pos);
            if (m_data == MAP_FAILED) {
                throw std::runtime_error("Can not create file mapping for " + std::to_string(fd) +
                                         ", err=" + std::strerror(errno));
            }
            m_handle = HandleHolder(fd);
            m_size = size;
        }
    }

public:
    PartialMapHolder(const std::filesystem::path& path, size_t pos, size_t size) {
        if (int fd = open(path.c_str(), O_RDONLY); fd != -1) {
            setup_mapping(fd, pos, size);
            m_id = std::hash<std::filesystem::path::string_type>{}(path.native()) ^ std::hash<size_t>{}(pos) ^
                   std::hash<size_t>{}(size);
        } else {
            throw std::runtime_error("Can not open file " + util::path_to_string(path) +
                                     " for mapping. Ensure that file exists and has appropriate permissions");
        }
    }
    ~PartialMapHolder() {
        if (m_data != MAP_FAILED) {
            munmap(m_data, m_size);
        }
    }

    char* data() noexcept override {
        return static_cast<char*>(m_data);
    }
    size_t size() const noexcept override {
        return m_size;
    }
    uint64_t get_id() const noexcept override {
        return m_id;
    }
};

std::shared_ptr<MappedMemory> load_mmap_object(const std::filesystem::path& path, size_t pos, size_t size) {
    return std::make_shared<PartialMapHolder>(path, pos, size);
}
}  // namespace ov
