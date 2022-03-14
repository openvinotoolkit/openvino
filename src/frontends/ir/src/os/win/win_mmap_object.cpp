// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mmap_object.hpp"
#include "ngraph/runtime/shared_buffer.hpp"
#include "openvino/util/file_util.hpp"

// clang-format-off
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

class MapHolder {
public:
    MapHolder() = default;

    ~MapHolder() {
        if (m_data) {
            ::UnmapViewOfFile(m_data);
        }
    }

    void set(const std::string& path) {
        auto h = ::CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        map(path, h);
    }

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
    void set(const std::wstring& path) {
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
        OPENVINO_ASSERT(h != INVALID_HANDLE_VALUE,
                        "Can not open file ",
                        path,
                        " for mapping. Ensure that file exists and has appropriate permissions");
        m_handle = HandleHolder(h);
        SYSTEM_INFO SystemInfo;
        GetSystemInfo(&SystemInfo);
        const int64_t page_size = SystemInfo.dwAllocationGranularity;

        DWORD file_mode = GENERIC_READ;
        DWORD map_mode = FILE_MAP_READ;
        DWORD access = PAGE_READONLY;

        LARGE_INTEGER file_size_large;
        OPENVINO_ASSERT(::GetFileSizeEx(m_handle.get(), &file_size_large) != 0, "Can not get file size for ", path);

        m_size = static_cast<uint64_t>(file_size_large.QuadPart);
        if (m_size > 0) {
            m_mapping =
                HandleHolder(::CreateFileMapping(m_handle.get(), 0, access, m_size >> 32, m_size & 0xffffffff, 0));
            OPENVINO_ASSERT(m_mapping.get() != INVALID_HANDLE_VALUE, "Can not create file mapping for ", path);

            m_data = ::MapViewOfFile(m_mapping.get(),
                                     map_mode,
                                     0,  // offset_align >> 32,
                                     0,  // offset_align & 0xffffffff,
                                     m_size);
            OPENVINO_ASSERT(m_data, "Can not create map view for ", path);
        } else {
            m_data = NULL;
        }
    }

private:
    void* m_data = NULL;
    size_t m_size = 0;
    HandleHolder m_handle;
    HandleHolder m_mapping;
};

std::shared_ptr<ngraph::runtime::AlignedBuffer> load_mmap_object(const std::string& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<MapHolder>>>(holder->data(),
                                                                                       holder->size(),
                                                                                       holder);
}

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::shared_ptr<ngraph::runtime::AlignedBuffer> load_mmap_object(const std::wstring& path) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path);
    return std::make_shared<ngraph::runtime::SharedBuffer<std::shared_ptr<MapHolder>>>(holder->data(),
                                                                                       holder->size(),
                                                                                       holder);
}

#endif

}  // namespace ov
