// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <chrono>
#include <cstring>
#include <future>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <vector>

#include "memory_prefetch.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/hash_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

// clang-format off
#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>
// clang-format on

namespace ov {
namespace util {

int64_t get_system_page_size() {
    static auto page_size = []() {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        return static_cast<int64_t>(sys_info.dwPageSize);
    }();
    return page_size;
}

size_t get_system_alloc_granularity() {
    static auto alloc_gran = []() {
        SYSTEM_INFO sys_info;
        GetSystemInfo(&sys_info);
        return static_cast<size_t>(sys_info.dwAllocationGranularity);
    }();
    return alloc_gran;
}

}  // namespace util

class HandleHolder {
    HANDLE m_handle = INVALID_HANDLE_VALUE;
    void reset() {
        if (valid()) {
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
        if (this != &other) {
            reset();
            m_handle = other.m_handle;
            other.m_handle = INVALID_HANDLE_VALUE;
        }
        return *this;
    }

    ~HandleHolder() {
        reset();
    }

    constexpr HANDLE get() const noexcept {
        return m_handle;
    }

    constexpr bool valid() const {
        return valid(m_handle);
    }

    static constexpr bool valid(HANDLE h) {
        return h != INVALID_HANDLE_VALUE && h != nullptr;
    }
};

class MapHolder final : public ov::MappedMemory {
public:
    MapHolder() = default;
    ~MapHolder() override;

    void set(const std::filesystem::path& path, size_t offset, size_t size, MmapMode mode = MmapMode::READ);
    void set_from_handle(FileHandle handle, size_t offset, size_t size);

    // ov::MappedMemory interface
    const std::byte* data() const noexcept override final {
        return m_data;
    }
    std::byte* data() noexcept override final {
        return m_data;
    }
    size_t size() const noexcept override final {
        return m_size;
    }

    std::optional<uint64_t> get_id() const noexcept override {
        return m_id;
    }

    void hint_evict(size_t offset, size_t size) noexcept override;

    void hint_prefetch(size_t offset, size_t size) noexcept override;

    util::PrefetchToken hint_prefetch_async(size_t offset, size_t size) noexcept override;

private:
    /**
     * @brief Adopts the handle returned by util::vm_prefetch_async(), reaping already-finished ones
     * so the pending list doesn't grow unbounded across repeated hint_prefetch_async() calls.
     */
    void adopt_pending_prefetch(util::PrefetchToken task) noexcept;

    /** @brief Joins all outstanding background prefetch tasks. Must run before the view is unmapped,
     *  since a detached task may still be touching its pages. */
    void wait_for_pending_prefetch() noexcept;

    void set_id(HANDLE h, size_t offset, size_t size);

    /** @brief Core setup shared by set() and set_from_handle(). */
    void setup(HANDLE file_handle, size_t offset, size_t size, MmapMode mode);

    std::byte* m_data{};  //!< pointer exposed to callers
    size_t m_size{};      //!< user-visible byte count
    std::optional<uint64_t> m_id;

    HandleHolder m_handle{};       //!< section object from CreateFileMappingW
    HandleHolder m_file_handle{};  //!< file HANDLE kept open to block DeleteFile (set-by-path only)
    char* m_view_base{};           //!< MapViewOfFile base address; nullptr when unmapped

    // Handles shared with hint_prefetch_async()'s caller; joined before unmapping (see ~MapHolder).
    std::mutex m_pending_prefetch_mutex;
    std::vector<util::PrefetchToken> m_pending_prefetch;
};

MapHolder::~MapHolder() {
    // Detached prefetch tasks may still be touching this mapping's pages; join them first.
    wait_for_pending_prefetch();
    if (m_view_base) {
        ::UnmapViewOfFile(m_view_base);
    }
}

void MapHolder::set_id(HANDLE h, size_t offset, size_t size) {
    if (FILE_ID_INFO info; GetFileInformationByHandleEx(h, FileIdInfo, &info, sizeof(info))) {
        static_assert(sizeof(info.FileId) == sizeof(uint64_t[2]));
        uint64_t fid[2];
        std::memcpy(fid, &info.FileId, sizeof(fid));
        m_id = util::u64_hash_combine(offset, {size, info.VolumeSerialNumber, fid[0], fid[1]});
    } else if (BY_HANDLE_FILE_INFORMATION info; ::GetFileInformationByHandle(h, &info)) {
        // GetFileInformationByHandleEx/FileIdInfo is unavailable (network FS, ReFS, older FS).
        // Fall back to the legacy NTFS/FAT file index + volume serial, which are stable across separate opens of the
        // same file (unlike a raw HANDLE value).
        const uint64_t file_index = (static_cast<uint64_t>(info.nFileIndexHigh) << 32) | info.nFileIndexLow;
        m_id = util::u64_hash_combine(offset, {size, info.dwVolumeSerialNumber, file_index});
    } else {
        // Last-resort fallback when no stable file identity metadata is available (e.g. exotic virtual filesystems).
        // HANDLE values are process-local and change across opens, so weight sharing may not work in this case.
        m_id = util::u64_hash_combine(offset, {size, std::hash<HANDLE>{}(h)});
    }
}

void MapHolder::setup(HANDLE file_handle, size_t offset, size_t size, MmapMode mode) {
    LARGE_INTEGER file_size_li{};
    if (!::GetFileSizeEx(file_handle, &file_size_li)) {
        throw std::runtime_error{"GetFileSizeEx failed: " + std::to_string(::GetLastError())};
    }
    const auto file_size = static_cast<size_t>(file_size_li.QuadPart);

    m_size = (size == auto_size) ? file_size - offset : size;
    if (offset + m_size > file_size || offset + m_size < offset) {
        throw std::runtime_error{"Requested mapping range exceeds file size"};
    }

    set_id(file_handle, offset, size);
    if (mode == MmapMode::READ_WRITE) {
        // A read-write mapping is not an immutable data source, so it must not be shared through id-based caches.
        m_id = std::nullopt;
    }

    if (m_size == 0) {
        return;
    }

    const DWORD protect = (mode == MmapMode::READ_WRITE) ? PAGE_READWRITE : PAGE_READONLY;
    m_handle = HandleHolder{::CreateFileMappingW(file_handle, nullptr, protect, 0, 0, nullptr)};
    if (!m_handle.valid()) {
        throw std::runtime_error{"CreateFileMappingW failed: " + std::to_string(::GetLastError())};
    }

    // The whole range is mapped as a single view: one AllocationBase, as required by zero-copy imports.
    const auto gran = util::get_system_alloc_granularity();
    const auto& [aligned_offset, aligned_length, head_pad] =
        util::align_region(static_cast<uintptr_t>(offset), m_size, gran);
    const DWORD access = (mode == MmapMode::READ_WRITE) ? FILE_MAP_ALL_ACCESS : FILE_MAP_READ;
    if (auto view = ::MapViewOfFile(m_handle.get(),
                                    access,
                                    static_cast<DWORD>(aligned_offset >> 32),
                                    static_cast<DWORD>(aligned_offset & 0xFFFFFFFF),
                                    head_pad + m_size)) {
        m_view_base = static_cast<char*>(view);
        m_data = reinterpret_cast<std::byte*>(m_view_base + head_pad);
    } else {
        throw std::runtime_error{"MapViewOfFile failed: " + std::to_string(::GetLastError())};
    }
}

void MapHolder::set(const std::filesystem::path& path, size_t offset, size_t size, MmapMode mode) {
    const bool writable = mode == MmapMode::READ_WRITE;
    auto fh = ::CreateFileW(
        path.c_str(),
        writable ? (GENERIC_READ | GENERIC_WRITE) : GENERIC_READ,
        writable ? (FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE) : (FILE_SHARE_READ | FILE_SHARE_DELETE),
        nullptr,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
        nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error{"Cannot open file: " + ov::util::path_to_string(path) +
                                 " error: " + std::to_string(::GetLastError())};
    }

    HandleHolder fh_holder{fh};
    setup(fh, offset, size, mode);
    // Keep the file handle alive so the section object can always resolve page faults
    // back to the original file data, even if the caller deletes or renames the file.
    // FILE_SHARE_DELETE allows std::filesystem::remove() to succeed while the mapping is alive.
    m_file_handle = std::move(fh_holder);
}

void MapHolder::set_from_handle(FileHandle handle, size_t offset, size_t size) {
    if (!HandleHolder::valid(static_cast<HANDLE>(handle))) {
        throw std::runtime_error{"Invalid handle provided to load_mmap_object"};
    }
    // Duplicate the caller's handle so MapHolder independently owns its lifetime.
    // The original handle remains valid for the caller to reuse.
    HANDLE dup = INVALID_HANDLE_VALUE;
    if (!::DuplicateHandle(::GetCurrentProcess(),
                           static_cast<HANDLE>(handle),
                           ::GetCurrentProcess(),
                           &dup,
                           0,
                           FALSE,
                           DUPLICATE_SAME_ACCESS)) {
        throw std::runtime_error{"DuplicateHandle failed: " + std::to_string(::GetLastError())};
    }
    HandleHolder owned{dup};
    setup(owned.get(), offset, size, MmapMode::READ);
    // owned goes out of scope here: file handle closed.
    // m_handle (section object) keeps the file data accessible independently.
}

namespace {

// Clamps [offset, offset + size) to [0, mapping_size) and page-aligns the result.
// Returns an empty region (m_length == 0) for a null/empty mapping, an offset at or past the end,
// or a sub-page request.
util::AlignedRegion clamp_align_region(const void* data, size_t mapping_size, size_t offset, size_t size) noexcept {
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    if (data == nullptr || mapping_size == 0 || offset >= mapping_size || size < page_size) {
        return {};
    }
    const auto available = mapping_size - offset;
    const auto raw_len = (size == auto_size) ? available : std::min(size, available);
    return util::align_region(reinterpret_cast<uintptr_t>(data) + offset, raw_len, page_size);
}

}  // namespace

void MapHolder::adopt_pending_prefetch(util::PrefetchToken task) noexcept {
    if (task.valid()) {
        try {
            std::lock_guard<std::mutex> lock(m_pending_prefetch_mutex);
            // Reap already-finished handles so the vector doesn't grow without bound across repeated
            // hint_prefetch_async() calls over this mapping's lifetime.
            m_pending_prefetch.erase(std::remove_if(m_pending_prefetch.begin(),
                                                    m_pending_prefetch.end(),
                                                    [](util::PrefetchToken& pending) {
                                                        return !pending.valid() ||
                                                               pending.wait_for(std::chrono::seconds(0)) ==
                                                                   std::future_status::ready;
                                                    }),
                                     m_pending_prefetch.end());
            m_pending_prefetch.push_back(std::move(task));
        } catch (const std::exception&) {
            // Bookkeeping failed (e.g. allocation failure): the prefetch already started in the
            // background, so block here instead of returning with it untracked, or ~MapHolder could
            // unmap while it is still touching these pages.
            task.wait();
        }
    }
}

void MapHolder::wait_for_pending_prefetch() noexcept {
    std::lock_guard<std::mutex> lock(m_pending_prefetch_mutex);
    for (auto& task : m_pending_prefetch) {
        if (task.valid()) {
            task.wait();
        }
    }
    m_pending_prefetch.clear();
}

void MapHolder::hint_prefetch(size_t offset, size_t size) noexcept {
    // Below 4 MiB the overhead of spawning threads exceeds the benefit; skip.
    if (const auto region = clamp_align_region(m_data, m_size, offset, size); region.m_length > 4 * util::one_mib) {
        const auto num_threads = std::min<size_t>(10, std::thread::hardware_concurrency());
        const auto aligned_size =
            util::align_size_up(region.m_length, static_cast<size_t>(util::get_system_page_size()));
        util::vm_prefetch(reinterpret_cast<void*>(region.m_address), aligned_size, num_threads);
    }
}

util::PrefetchToken MapHolder::hint_prefetch_async(size_t offset, size_t size) noexcept {
    if (const auto region = util::clamp_align_region(m_data, m_size, offset, size);
        region.m_length > util::default_parallel_io_threshold) {
        auto future = util::vm_prefetch_async(reinterpret_cast<void*>(region.m_address),
                                              region.m_length,
                                              util::prefetch_thread_count(region.m_length));
        adopt_pending_prefetch(future);
        return future;
    }
    return {};
}

void MapHolder::hint_evict(size_t offset, size_t size) noexcept {
    // VirtualUnlock on unlocked pages only drops them from the working set; the view stays intact and the pages
    // are soft-faulted back on the next access. Unmapping the view in parts would tear down the pages prefetched
    // ahead of the reader and costs seconds on large models.
    if (const auto region = util::clamp_align_region(m_data, m_size, offset, size); region.m_length != 0) {
        ::VirtualUnlock(reinterpret_cast<void*>(region.m_address), region.m_length);
    }
}

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::filesystem::path& path,
                                                   size_t offset,
                                                   size_t size,
                                                   MmapMode mode) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path, offset, size, mode);
    return holder;
}

std::shared_ptr<ov::MappedMemory> load_mmap_object(FileHandle handle, size_t offset, size_t size) {
    if (!HandleHolder::valid(static_cast<HANDLE>(handle))) {
        throw std::runtime_error("Invalid handle provided to load_mmap_object");
    }
    auto holder = std::make_shared<MapHolder>();
    holder->set_from_handle(handle, offset, size);
    return holder;
}
}  // namespace ov
