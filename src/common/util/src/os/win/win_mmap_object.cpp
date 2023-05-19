// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <sstream>

#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

// clang-format-off
#include <direct.h>
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>
// clang-format-on

namespace ov {
namespace util {
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

class MapHolder {
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

    char* data() noexcept {
        return static_cast<char*>(m_data);
    }
    size_t size() const noexcept {
        return m_size;
    }

private:
    void map(const std::string& path, HANDLE h) {
        if (h == INVALID_HANDLE_VALUE) {
            std::stringstream ss;
            ss << "Cannot load library " << path
               << " for mapping. Ensure that file exists and has appropriate permissions";
            throw std::runtime_error(ss.str());
        }

        m_handle = HandleHolder(h);
        SYSTEM_INFO SystemInfo;
        GetSystemInfo(&SystemInfo);

        DWORD map_mode = FILE_MAP_READ;
        DWORD access = PAGE_READONLY;

        LARGE_INTEGER file_size_large;
        if (::GetFileSizeEx(m_handle.get(), &file_size_large) <= 0) {
            char cwd[1024];
            std::stringstream ss;
            ss << "Can not get file size for " << path << " : " << GetLastError()
               << " from cwd: " << _getcwd(cwd, sizeof(cwd));
            throw std::runtime_error(ss.str());
        }

        m_size = static_cast<uint64_t>(file_size_large.QuadPart);
        if (m_size > 0) {
            m_mapping =
                HandleHolder(::CreateFileMapping(m_handle.get(), 0, access, m_size >> 32, m_size & 0xffffffff, 0));
            if (m_mapping.get() == INVALID_HANDLE_VALUE) {
                char cwd[1024];
                std::stringstream ss;
                ss << "Can not create file mapping for " << path << " : " << GetLastError()
                   << " from cwd: " << _getcwd(cwd, sizeof(cwd));
                throw std::runtime_error(ss.str());
            }

            m_data = ::MapViewOfFile(m_mapping.get(),
                                     map_mode,
                                     0,  // offset_align >> 32,
                                     0,  // offset_align & 0xffffffff,
                                     m_size);
            if (m_data == nullptr) {
                char cwd[1024];
                std::stringstream ss;
                ss << "Can not create map view for  " << path << " : " << GetLastError()
                   << " from cwd: " << _getcwd(cwd, sizeof(cwd));
                throw std::runtime_error(ss.str());
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

std::shared_ptr<MmapBuffer> load_mmap_object(const std::string& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return std::make_shared<SharedMmapBuffer<std::shared_ptr<MapHolder>>>(holder->data(), holder->size(), holder);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::shared_ptr<MmapBuffer> load_mmap_object(const std::wstring& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return std::make_shared<SharedMmapBuffer<std::shared_ptr<MapHolder>>>(holder->data(), holder->size(), holder);
}

#endif
}  // namespace util
}  // namespace ov
