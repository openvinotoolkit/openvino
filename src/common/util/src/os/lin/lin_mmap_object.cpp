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

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace util {
int64_t get_system_page_size() {
    static auto page_size = static_cast<int64_t>(sysconf(_SC_PAGE_SIZE));
    return page_size;
}

struct PageAlignedRegion {
    uintptr_t m_address = 0;
    size_t m_length = 0;
    size_t m_gap = 0;
    bool m_valid = false;
};

constexpr PageAlignedRegion align_to_page(uintptr_t base, size_t raw_len, size_t page_size) {
    const auto aligned = (base / page_size) * page_size;
    const auto gap = base - aligned;
    return {aligned, raw_len + gap, gap, true};
}

inline PageAlignedRegion make_mmap_region(size_t file_offset, size_t size) {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    return align_to_page(static_cast<uintptr_t>(file_offset), size, page_size);
}

inline PageAlignedRegion make_madvise_region(const void* data, size_t mapping_size, size_t offset, size_t size) {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    if (data == nullptr || mapping_size == 0 || offset >= mapping_size || size < page_size) {
        return {};

        const auto available = mapping_size - offset;
        const auto raw_len = (size == auto_size) ? available : std::min(size, available);
        return align_to_page(reinterpret_cast<uintptr_t>(data) + offset, raw_len, page_size);
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
    void* m_mapped_view = MAP_FAILED;
    size_t m_mapped_view_size = 0;
    void* m_data = nullptr;
    size_t m_size = 0;
    uint64_t m_id = std::numeric_limits<uint64_t>::max();
    HandleHolder m_handle;

public:
    MapHolder() = default;

    void set(const std::filesystem::path& path, const size_t offset, const size_t size) {
        int mode = O_RDONLY;
        int fd = open(path.c_str(), mode);
        if (fd == -1) {
            throw std::runtime_error("Can not open file " + util::path_to_string(path) +
                                     " for mapping. Ensure that file exists and has appropriate permissions.");
        }
        set_from_fd(fd, offset, size);
        m_id = util::get_id_for_file(path, offset, size);
    }

    void set_from_fd(const int fd, const size_t offset, const size_t size) {
        m_handle = HandleHolder(fd);

        struct stat sb = {};
        if (fstat(fd, &sb) == -1) {
            throw std::runtime_error("Can not get file size for fd=" + std::to_string(fd));
        }
        const auto file_size = static_cast<size_t>(sb.st_size);
        m_size = (size == auto_size) ? file_size - offset : size;
        if (offset + m_size > file_size || offset + m_size < offset) {
            throw std::runtime_error("Requested mapping range exceeds file size for fd=" + std::to_string(fd));
        }

        if (m_size > 0) {
            const auto view = util::make_mmap_region(offset, m_size);
            m_mapped_view_size = view.m_length;
            m_mapped_view = mmap(nullptr, view.m_length, PROT_READ, MAP_SHARED, fd, view.m_address);
            if (m_mapped_view == MAP_FAILED) {
                throw std::runtime_error("Can not create file mapping for " + std::to_string(fd) +
                                         ", err=" + std::strerror(errno));
            }
            m_data = static_cast<char*>(m_mapped_view) + view.m_gap;
        }
        m_id =
            util::u64_hash_combine(static_cast<uint64_t>(sb.st_ino), {static_cast<uint64_t>(sb.st_dev), offset, size});
    }

    uint64_t get_id() const noexcept override {
        return m_id;
    }

    ~MapHolder() {
        if (m_mapped_view != MAP_FAILED) {
            munmap(m_mapped_view, m_mapped_view_size);
        }
    }

    char* data() noexcept override {
        return static_cast<char*>(m_data);
    }

    size_t size() const noexcept override {
        return m_size;
    }

    void hint_evict(size_t offset, size_t size) override {
        if (m_mapped_view != MAP_FAILED) {
            if (const auto region = util::make_madvise_region(m_data, m_size, offset, size); region.m_valid) {
                std::ignore = madvise(reinterpret_cast<void*>(region.m_address), region.m_length, MADV_DONTNEED);
            }
        }
    }
};

std::shared_ptr<MappedMemory> load_mmap_object(const std::filesystem::path& path, size_t offset, size_t size) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path, offset, size);
    return holder;
}

std::shared_ptr<ov::MappedMemory> load_mmap_object(FileHandle handle, size_t offset, size_t size) {
    if (handle == -1) {
        throw std::runtime_error("Invalid file descriptor provided for mapping.");
    }
    auto holder = std::make_shared<MapHolder>();
    holder->set_from_fd(handle, offset, size);
    return holder;
}
}  // namespace ov
