// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <future>
#include <mutex>
#include <thread>
#include <tuple>

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace util {
int64_t get_system_page_size() {
    static auto page_size = static_cast<int64_t>(sysconf(_SC_PAGE_SIZE));
    return page_size;
}

/**
 * @brief Creates a memory region for mmap operations.
 *
 * @param offset The offset within the mmap region.
 * @param size   The size of the region.
 * @return AlignedRegion The aligned memory region.
 */
inline util::AlignedRegion make_mmap_region(size_t offset, size_t size) {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    return util::align_region(static_cast<uintptr_t>(offset), size, page_size);
}

/**
 * @brief Creates a memory region for madvise operations.
 *
 * @param data         The base address of the mapped memory region.
 * @param mapping_size The size of the mapped memory region.
 * @param offset       The offset within the mapped memory region.
 * @param size         The size of the region.
 * @return AlignedRegion The aligned memory region.
 */
inline util::AlignedRegion make_madvise_region(const void* data, size_t mapping_size, size_t offset, size_t size) {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    if (data == nullptr || mapping_size == 0 || offset >= mapping_size || size < page_size) {
        return {};
    } else {
        const auto available = mapping_size - offset;
        const auto raw_len = (size == auto_size) ? available : std::min(size, available);
        return util::align_region(reinterpret_cast<uintptr_t>(data) + offset, raw_len, page_size);
    }
}

/**
 * @brief Plan computed by @ref make_prefetch_plan, shared between the sync and async
 * hint_prefetch() implementations.
 */
struct PrefetchPlan {
    uintptr_t m_address = 0;
    size_t m_aligned_size = 0;
    size_t m_num_threads = 0;
    bool m_should_prefetch = false;
};

/**
 * @brief Computes the aligned region, page-aligned size and thread count to use for a
 * hint_prefetch()/hint_prefetch_async() call.
 *
 * @param data         The base address of the mapped memory region.
 * @param mapping_size The size of the mapped memory region.
 * @param offset       Offset within the mapping where prefetching starts.
 * @param size         Number of bytes to prefetch.
 * @return PrefetchPlan with m_should_prefetch == false when the requested region is below the
 * 4 MiB threshold (spawning threads would not be worth it).
 */
inline PrefetchPlan make_prefetch_plan(const void* data, size_t mapping_size, size_t offset, size_t size) {
    constexpr size_t one_mb = 1024 * 1024;
    // Below 4 MiB the overhead of spawning threads exceeds the benefit; skip.
    if (const auto region = make_madvise_region(data, mapping_size, offset, size); region.m_length > 4 * one_mb) {
        return {region.m_address,
                align_size_up(region.m_length, static_cast<size_t>(get_system_page_size())),
                std::min<size_t>(10, std::thread::hardware_concurrency()),
                true};
    }
    return {};
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
    // Tasks handed off via PrefetchToken::detach(): must be waited on before unmapping, see
    // wait_for_pending_prefetch() / ~MapHolder().
    std::mutex m_pending_prefetch_mutex;
    std::vector<std::future<void>> m_pending_prefetch;

    void adopt_pending_prefetch(std::vector<std::future<void>>&& tasks) {
        std::lock_guard<std::mutex> lock(m_pending_prefetch_mutex);
        m_pending_prefetch.insert(m_pending_prefetch.end(),
                                  std::make_move_iterator(tasks.begin()),
                                  std::make_move_iterator(tasks.end()));
    }

    void wait_for_pending_prefetch() noexcept {
        std::lock_guard<std::mutex> lock(m_pending_prefetch_mutex);
        for (auto& task : m_pending_prefetch) {
            if (task.valid()) {
                task.wait();
            }
        }
        m_pending_prefetch.clear();
    }

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
            const auto& [aligned_offset, length, gap] = util::make_mmap_region(offset, m_size);
            m_mapped_view_size = length;
            m_mapped_view = mmap(nullptr, length, PROT_READ, MAP_SHARED, fd, aligned_offset);
            if (m_mapped_view == MAP_FAILED) {
                throw std::runtime_error("Can not create file mapping for " + std::to_string(fd) +
                                         ", err=" + std::strerror(errno));
            }
            m_data = static_cast<char*>(m_mapped_view) + gap;
        }
        m_id =
            util::u64_hash_combine(static_cast<uint64_t>(sb.st_ino), {static_cast<uint64_t>(sb.st_dev), offset, size});
    }

    uint64_t get_id() const noexcept override {
        return m_id;
    }

    ~MapHolder() {
        // Detached prefetch tasks (see hint_prefetch_async()) may still be touching this
        // mapping's pages; they must complete before the mapping is torn down.
        wait_for_pending_prefetch();
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

    void hint_evict(size_t offset, size_t size) noexcept override {
        if (m_mapped_view != MAP_FAILED) {
            if (const auto region = util::make_madvise_region(m_data, m_size, offset, size); region.m_length > 0) {
                std::ignore = madvise(reinterpret_cast<void*>(region.m_address), region.m_length, MADV_DONTNEED);
            }
        }
    }

    void hint_prefetch(size_t offset, size_t size) override {
        if (const auto plan = util::make_prefetch_plan(m_data, m_size, offset, size); plan.m_should_prefetch) {
            util::vm_prefetch(reinterpret_cast<void*>(plan.m_address), plan.m_aligned_size, plan.m_num_threads);
        }
    }

    util::PrefetchToken hint_prefetch_async(size_t offset, size_t size) override {
        if (const auto plan = util::make_prefetch_plan(m_data, m_size, offset, size); plan.m_should_prefetch) {
            auto token = util::vm_prefetch_async(reinterpret_cast<void*>(plan.m_address),
                                                  plan.m_aligned_size,
                                                  plan.m_num_threads);
            // Allow token.detach() to hand the still-running tasks off to this object, so that
            // they are guaranteed to complete before ~MapHolder() unmaps the memory.
            token.set_detach_sink([this](std::vector<std::future<void>>&& tasks) {
                adopt_pending_prefetch(std::move(tasks));
            });
            return token;
        }
        return {};
    }
};

std::shared_ptr<MappedMemory> load_mmap_object(const std::filesystem::path& path,
                                               size_t offset,
                                               size_t size,
                                               bool /* no_placeholder */) {
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
