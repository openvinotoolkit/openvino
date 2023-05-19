// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace util {
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

class MapHolder {
    void* m_data = MAP_FAILED;
    size_t m_size = 0;
    HandleHolder m_handle;

public:
    MapHolder() = default;

    void set(const std::string& path) {
        int prot = PROT_READ;
        int mode = O_RDONLY;
        struct stat sb = {};
        m_handle = HandleHolder(open(path.c_str(), mode));
        if (m_handle.get() == -1) {
            std::stringstream ss;
            ss << "Can not open file  '" << path
               << "' for mapping. Ensure that file exists and has appropriate permissions";
            if (auto error = dlerror()) {
                ss << ": " << error;
            }
            throw std::runtime_error(ss.str());
        }

        if (fstat(m_handle.get(), &sb) == -1) {
            std::stringstream ss;
            ss << "Can not get file size for " << path;
            if (auto error = dlerror()) {
                ss << ": " << error;
            }
            throw std::runtime_error(ss.str());
        }
        m_size = sb.st_size;
        if (m_size > 0) {
            m_data = mmap(nullptr, m_size, prot, MAP_PRIVATE, m_handle.get(), 0);
            if (m_data == MAP_FAILED) {
                std::stringstream ss;
                ss << "Can not create file mapping for " << path;
                if (auto error = dlerror()) {
                    ss << ": " << error;
                }
                throw std::runtime_error(ss.str());
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

    char* data() noexcept {
        return static_cast<char*>(m_data);
    }

    size_t size() const noexcept {
        return m_size;
    }
};

std::shared_ptr<MmapBuffer> load_mmap_object(const std::string& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return std::make_shared<SharedMmapBuffer<std::shared_ptr<MapHolder>>>(holder->data(), holder->size(), holder);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::shared_ptr<MmapBuffer> load_mmap_object(const std::wstring& path) {
    auto _path = ov::util::wstring_to_string(path);
    return load_mmap_object(_path);
}

#endif
}  // namespace util
}  // namespace ov
