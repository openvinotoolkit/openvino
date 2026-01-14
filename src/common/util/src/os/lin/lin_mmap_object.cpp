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

class MapHolder : public MappedMemory {
    void* m_data = MAP_FAILED;
    size_t m_size = 0;
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

std::shared_ptr<ov::MappedMemory> load_mmap_object(FileHandle handle) {
    // On Linux, FileHandle is int (file descriptor)
    auto holder = std::make_shared<MapHolder>();
    holder->set_from_fd(static_cast<int>(handle));
    return holder;
}

}  // namespace ov
