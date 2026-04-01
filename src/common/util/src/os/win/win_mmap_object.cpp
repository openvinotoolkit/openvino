// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <stdexcept>

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

// clang-format-off
#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>
// clang-format-on

namespace ov {
namespace util {
int64_t get_system_page_size() {
    static auto page_size = []() {
        SYSTEM_INFO sysInfo;
        GetSystemInfo(&sysInfo);
        return static_cast<int64_t>(sysInfo.dwPageSize);
    }();
    return page_size;
}
}  // namespace util

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
        if (m_mapped_view) {
            ::UnmapViewOfFile(m_mapped_view);
        }
    }

    void set(const std::filesystem::path& path, size_t offset, size_t size) {
        // Note that file can't be changed (renamed/deleted) until it's unmapped. FILE_SHARE_DELETE flag allow
        // rename/deletion, but it doesn't work with FAT32 filesystem (works on NTFS)
        const auto h =
            ::CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, 0, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, 0);
        map(path, h, offset, size);
        m_id = util::u64_hash_combine({std::hash<std::filesystem::path::string_type>{}(path.native()), offset, size});
    }

    void set_from_handle(HANDLE h, size_t offset, size_t size) {
        map("<external_handle>", h, offset, size);
        set_id(h, offset, size);
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

private:
    void set_id(const HANDLE h, const size_t offset, const size_t size) {
        if (FILE_ID_INFO info; GetFileInformationByHandleEx(h, FileIdInfo, &info, sizeof(info))) {
            static_assert(sizeof(info.FileId) == 16);
            uint64_t fid_l, fid_r;
            std::memcpy(&fid_l, &info.FileId, sizeof(fid_l));
            std::memcpy(&fid_r, reinterpret_cast<const char*>(&info.FileId) + sizeof(fid_l), sizeof(fid_r));
            m_id = util::u64_hash_combine({offset, size, info.VolumeSerialNumber, fid_l, fid_r});
        } else {
            throw std::runtime_error{"Cannot obtain file id info for handle " +
                                     std::to_string(reinterpret_cast<uint64_t>(h))};
        }
    }

    void map(const std::filesystem::path& path, const HANDLE h, const size_t offset, const size_t size) {
        if (h == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("Can not open file " + util::path_to_string(path) +
                                     " for mapping. Ensure that file exists and has appropriate permissions");
        }
        m_handle = HandleHolder(h);

        LARGE_INTEGER file_size_large;
        if (::GetFileSizeEx(m_handle.get(), &file_size_large) == 0) {
            throw std::runtime_error("Can not get file size for " + util::path_to_string(path) + ". Error " +
                                     std::to_string(::GetLastError()));
        }
        const auto file_size = static_cast<size_t>(file_size_large.QuadPart);
        m_size = (size == auto_size) ? file_size - offset : size;
        if (offset + m_size > file_size || offset + m_size < offset) {
            throw std::runtime_error("Requested mapping range exceeds file size for " + util::path_to_string(path));
        }

        if (m_size > 0) {
            m_mapping = HandleHolder(::CreateFileMapping(m_handle.get(), 0, PAGE_READONLY, 0, 0, 0));
            if (m_mapping.get() == INVALID_HANDLE_VALUE) {
                throw std::runtime_error("Can not create file mapping for " + util::path_to_string(path));
            }

            SYSTEM_INFO system_info;
            ::GetSystemInfo(&system_info);
            const auto aligned_offset =
                (offset / system_info.dwAllocationGranularity) * system_info.dwAllocationGranularity;
            const auto aligned_size = offset + m_size - aligned_offset;
            m_mapped_view = ::MapViewOfFile(m_mapping.get(),
                                            FILE_MAP_READ,
                                            aligned_offset >> 32,
                                            aligned_offset & 0xffffffff,
                                            aligned_size);
            if (!m_mapped_view) {
                throw std::runtime_error("Can not create map view for " + util::path_to_string(path) + ". Error " +
                                         std::to_string(::GetLastError()));
            }
            m_data = reinterpret_cast<char*>(m_mapped_view) + (offset - aligned_offset);
        }
    }

private:
    void* m_mapped_view = nullptr;
    void* m_data = nullptr;
    size_t m_size = 0;
    uint64_t m_id = std::numeric_limits<uint64_t>::max();
    HandleHolder m_handle;
    HandleHolder m_mapping;
};

std::shared_ptr<MappedMemory> load_mmap_object(const std::filesystem::path& path, size_t offset, size_t size) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path, offset, size);
    return holder;
}

std::shared_ptr<ov::MappedMemory> load_mmap_object(FileHandle handle, size_t offset, size_t size) {
    if (handle == INVALID_HANDLE_VALUE || handle == nullptr) {
        throw std::runtime_error("Invalid handle provided to load_mmap_object");
    }
    auto holder = std::make_shared<MapHolder>();
    holder->set_from_handle(handle, offset, size);
    return holder;
}
}  // namespace ov
