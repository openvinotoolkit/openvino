// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>

#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

// clang-format-off
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>
// clang-format-on

namespace ov {

class HandleHolder {
    HANDLE m_handle = INVALID_HANDLE_VALUE;
    void reset() {
        if (m_handle != INVALID_HANDLE_VALUE) {
            ::CloseHandle(m_handle);
            m_handle = INVALID_HANDLE_VALUE;
        }
    }

public:
    explicit HandleHolder(HANDLE handle = INVALID_HANDLE_VALUE) : m_handle(handle) {}
    HandleHolder(const HandleHolder&) = delete;
    HandleHolder(HandleHolder&& other) noexcept : m_handle(other.m_handle) {
        other.m_handle = INVALID_HANDLE_VALUE;
    }
    HandleHolder& operator=(const HandleHolder&) = delete;
    HandleHolder& operator=(HandleHolder&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        reset();
        m_handle = other.m_handle;
        other.m_handle = INVALID_HANDLE_VALUE;
        return *this;
    }

    ~HandleHolder() {
        reset();
    }

    HANDLE get() const noexcept {
        return m_handle;
    }
};

class MapHolder : public ov::MappedMemory {
public:
    MapHolder() = default;

    ~MapHolder() {
        if (m_data) {
            ::UnmapViewOfFile(m_data);
        }
    }

    void set(const std::string& path) {
        // Note that file can't be changed (renamed/deleted) until it's unmapped. FILE_SHARE_DELETE flag allow
        // rename/deletion, but it doesn't work with FAT32 filesystem (works on NTFS)
        auto h = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        map(path, h);
    }

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    void set(const std::wstring& path) {
        // Note that file can't be changed (renamed/deleted) until it's unmapped. FILE_SHARE_DELETE flag allow
        // rename/deletion, but it doesn't work with FAT32 filesystem (works on NTFS)
        auto h = ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        map(ov::util::wstring_to_string(path), h);
    }
#endif

    char* data() noexcept override {
        return static_cast<char*>(m_data);
    }
    size_t size() const noexcept override {
        return m_size;
    }

private:
    void map(const std::string& path, HANDLE h) {
        if (h == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Can not open file " + path +
                                     " for mapping. Ensure that file exists and has appropriate permissions");
        }
        m_handle = HandleHolder(h);

        DWORD map_mode = FILE_MAP_READ;
        DWORD access = PAGE_READONLY;

        LARGE_INTEGER file_size_large;
        if (::GetFileSizeEx(m_handle.get(), &file_size_large) == 0) {
            throw std::runtime_error("Can not get file size for " + path);
        }

        m_size = static_cast<uint64_t>(file_size_large.QuadPart);
        if (m_size > 0) {
            m_mapping =
                HandleHolder(::CreateFileMapping(m_handle.get(), 0, access, m_size >> 32, m_size & 0xffffffff, 0));
            if (m_mapping.get() == INVALID_HANDLE_VALUE) {
                throw std::runtime_error("Can not create file mapping for " + path);
            }

            m_data = ::MapViewOfFile(m_mapping.get(),
                                     map_mode,
                                     0,  // offset_align >> 32,
                                     0,  // offset_align & 0xffffffff,
                                     m_size);
            if (!m_data) {
                throw std::runtime_error("Can not create map view for " + path);
            }
        } else {
            m_data = nullptr;
        }
    }

private:
    void* m_data = nullptr;
    size_t m_size = 0;
    HandleHolder m_handle;
    HandleHolder m_mapping;
};

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::string& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return holder;
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::wstring& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return holder;
}
#endif

}  // namespace ov
