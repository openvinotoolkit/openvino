// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"

// clang-format off
#ifndef NOMINMAX
#    define NOMINMAX
#endif

#include <windows.h>
// clang-format on

// Placeholder memory API flags (Windows 10 RS4 / 1803+).
#ifndef MEM_PRESERVE_PLACEHOLDER
#    define MEM_PRESERVE_PLACEHOLDER 0x00000002
#endif
#ifndef MEM_REPLACE_PLACEHOLDER
#    define MEM_REPLACE_PLACEHOLDER 0x00004000
#endif
#ifndef MEM_RESERVE_PLACEHOLDER
#    define MEM_RESERVE_PLACEHOLDER 0x00040000
#endif

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

class MapHolder;

// Function pointers for  Windows 10 1803+ placeholder memory APIs.
// Using void* for MEM_EXTENDED_PARAMETER* since we always pass nullptr/0.
using PFNVirtualAlloc2 = PVOID(WINAPI*)(HANDLE, PVOID, SIZE_T, ULONG, ULONG, void*, ULONG);
using PFNMapViewOfFile3 = PVOID(WINAPI*)(HANDLE, HANDLE, PVOID, ULONG64, SIZE_T, ULONG, ULONG, void*, ULONG);
using PFNUnmapViewOfFile2 = BOOL(WINAPI*)(HANDLE, PVOID, ULONG);

/** @brief Helper class to load placeholder APIs dynamically and check availability at runtime. */
struct PlaceholderApis {
    PFNVirtualAlloc2 m_virtual_alloc2{};
    PFNMapViewOfFile3 m_map_view_of_file3{};
    PFNUnmapViewOfFile2 m_unmap_view_of_file2{};
    bool m_available{};

    static const PlaceholderApis& instance() {
        static const PlaceholderApis s = load();
        return s;
    }

private:
    static PlaceholderApis load() {
        PlaceholderApis a;
        if (const HMODULE h = ::GetModuleHandleW(L"kernelbase.dll")) {
            a.m_virtual_alloc2 = reinterpret_cast<PFNVirtualAlloc2>(::GetProcAddress(h, "VirtualAlloc2"));
            a.m_map_view_of_file3 = reinterpret_cast<PFNMapViewOfFile3>(::GetProcAddress(h, "MapViewOfFile3"));
            a.m_unmap_view_of_file2 = reinterpret_cast<PFNUnmapViewOfFile2>(::GetProcAddress(h, "UnmapViewOfFile2"));
            a.m_available = a.m_virtual_alloc2 && a.m_map_view_of_file3 && a.m_unmap_view_of_file2;
        }
        return a;
    }
};

// Replaces the placeholder at exactly `base` with a file section view starting at `offset`.
// Note: MapViewOfFile3 truncates its ULONG64 Offset to LONG32 internally, so offsets >= 2^31
// will fail with ERROR_ACCESS_DENIED. Files larger than 2 GiB require an NtMapViewOfSectionEx
// call instead; this path is not yet implemented.
static PVOID replace_placeholder(const PlaceholderApis& api,
                                 HANDLE section,
                                 HANDLE proc,
                                 char* base,
                                 ULONG64 offset,
                                 size_t size) {
    return api
        .m_map_view_of_file3(section, proc, base, offset, size, MEM_REPLACE_PLACEHOLDER, PAGE_READONLY, nullptr, 0);
}

// Returns the address just past the end of the described memory region.
static char* region_end(const MEMORY_BASIC_INFORMATION& mbi) {
    return static_cast<char*>(mbi.BaseAddress) + mbi.RegionSize;
}

// Queries the region containing addr. Returns false if VirtualQuery fails.
static bool query_region(void* addr, MEMORY_BASIC_INFORMATION& mbi) {
    return ::VirtualQuery(addr, &mbi, sizeof(mbi)) != 0;
}

/**
 * @brief VEH-based registry of MapHolders for on-demand remapping of evicted slots.
 *
 * When hint_evict() evicts a slot by unmapping it as a PAGE_NOACCESS placeholder, the VEH below will catch
 * the resulting access violation and ask this registry to find the owning MapHolder and re-map the slot on demand.
 */
struct MmapVehRegistry {
    struct Entry {
        MapHolder* m_holder;     ///< Pointer to the MapHolder owning this VA range
        char* m_base;            ///< Base address of the placeholder VA reservation
        size_t m_total_va_size;  ///< Total size of the VA reservation in bytes
    };

    /**
     * @brief Mutex protecting m_ranges and m_veh_handle. Shared lock for find() (concurrent faults),
     * exclusive lock for add()/remove() (registration changes).
     */
    std::shared_mutex m_mtx{};
    std::map<uintptr_t, Entry> m_ranges{};  // Maps VA reservation base address to MapHolder and size
    PVOID m_veh_handle{};

    /**
     * @brief Returns the process-wide singleton registry instance.
     *
     * Singleton is required because:
     * - Vectored Exception Handler (VEH) is process-wide and must be registered once
     * - Provides centralized VA-to-MapHolder routing for access violation faults
     * - Enables thread-safe concurrent fault handling via shared_mutex (multiple threads can simultaneously fault
     *   and remap different regions without blocking each other)
     *
     * @return Reference to the global MmapVehRegistry instance.
     */
    static MmapVehRegistry& instance() {
        static MmapVehRegistry reg;
        return reg;
    }

    /**
     * @brief Registers a MapHolder's VA reservation range in the global registry.
     *
     * Lazily registers the process-wide VEH on first call (when m_veh_handle is nullptr).
     * Subsequent calls only update m_ranges without re-registering the VEH.
     *
     * @param h Pointer to the MapHolder to register (must remain valid until remove() is called)
     * @param base Base address of the MapHolder's VA reservation (used as map key)
     * @param total_va_size Total size of the VA reservation in bytes
     */
    void add(MapHolder* h, char* base, size_t total_va_size) {
        std::unique_lock lock(m_mtx);
        const auto key = reinterpret_cast<uintptr_t>(base);
        m_ranges[key] = Entry{h, base, total_va_size};
        if (!m_veh_handle) {
            // Register as first VEH so it runs before debugger/CRT handlers.
            m_veh_handle = ::AddVectoredExceptionHandler(1, MmapVehRegistry::veh);
        }
    }

    /**
     * @brief Unregisters a MapHolder's VA reservation range from the global registry.
     *
     * If this was the last entry (m_ranges becomes empty), also unregisters the process-wide
     * VEH by calling RemoveVectoredExceptionHandler() and sets m_veh_handle to nullptr.
     *
     * @param base Base address of the VA reservation to remove (must match the key used in add())
     */
    void remove(char* base) {
        std::unique_lock lock(m_mtx);
        m_ranges.erase(reinterpret_cast<uintptr_t>(base));
        if (m_ranges.empty() && m_veh_handle) {
            ::RemoveVectoredExceptionHandler(m_veh_handle);
            m_veh_handle = nullptr;
        }
    }

    /**
     * @brief Finds the registry entry containing the given fault address.
     *
     * Uses upper_bound() to efficiently locate the entry whose VA range contains fault_addr.
     * Algorithm: Find the first entry with key > fault_addr, step back one entry, then verify
     * that fault_addr < (entry.m_base + entry.m_total_va_size).
     *
     * @param fault_addr The faulting virtual address (from EXCEPTION_ACCESS_VIOLATION)
     * @return Pointer to Entry if fault_addr falls within a registered VA range; nullptr otherwise.
     */
    const Entry* find(uintptr_t fault_addr) const {
        if (auto it = m_ranges.upper_bound(fault_addr); it == m_ranges.begin()) {
            return nullptr;
        } else {
            --it;
            const auto& e = it->second;
            const auto end = reinterpret_cast<uintptr_t>(e.m_base) + e.m_total_va_size;
            return (fault_addr < end) ? &e : nullptr;
        }
    }

private:
    /**
     * @brief Vectored Exception Handler callback for EXCEPTION_ACCESS_VIOLATION.
     *
     * Registered as first-chance VEH (priority 1) so it runs before debugger/CRT handlers.
     * On access violation, extracts the faulting address, finds the owning MapHolder via find(), and calls
     * try_remap_slot() to restore the evicted page.
     *
     * @param ep Exception pointers provided by Windows SEH dispatcher
     * @return EXCEPTION_CONTINUE_EXECUTION if handled, EXCEPTION_CONTINUE_SEARCH if not our fault
     */
    static LONG NTAPI veh(PEXCEPTION_POINTERS ep);
};

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

class MapHolder : public ov::MappedMemory {
public:
    MapHolder() = default;
    ~MapHolder() override;

    void set(const std::filesystem::path& path, size_t offset, size_t size);
    void set_from_handle(FileHandle handle, size_t offset, size_t size);
    bool try_remap_slot(uintptr_t fault_addr);

    // ov::MappedMemory interface
    char* data() noexcept override {
        return static_cast<char*>(m_data);
    }
    size_t size() const noexcept override {
        return m_size;
    }

    uint64_t get_id() const noexcept override {
        return m_id;
    }

    void hint_evict(size_t offset, size_t size) noexcept override;

private:
    /**
     * @brief Remaps a placeholder region by replacing it with a file-backed view.
     *
     * @param proc Process handle (typically GetCurrentProcess()).
     * @param base Base address of the placeholder region to remap (must be within m_view_base range).
     * @param size Size of the placeholder region in bytes (must match the placeholder's region size).
     * @return true if succeeded and returned the expected address; false otherwise
     */
    bool remap_placeholder(HANDLE proc, char* base, size_t size);

    void set_id(HANDLE h, size_t offset, size_t size);

    /** @brief Core setup shared by set() and set_from_handle(). */
    void setup(HANDLE file_handle, size_t offset, size_t size);

    /** @brief Try to establish the placeholder mapping.
     *  Returns true on success; caller falls back to legacy path on false.
     */
    bool try_placeholder_setup(size_t aligned_offset, size_t head_pad, size_t total_va_size, size_t file_size);

    /** @brief Legacy single-call MapViewOfFile path (no partial-release support). */
    void legacy_setup(size_t aligned_offset, size_t head_pad, size_t size);

    /**
     * @brief Computes the clamped, gran-aligned VA range to evict.
     * Validates all preconditions (placeholder path, API availability, offset bounds).
     * @return {range_begin, range_end} into the placeholder VA, or {nullptr, nullptr} if nothing to evict.
     */
    std::pair<char*, char*> compute_evict_range(size_t offset, size_t size) const noexcept;

    /**
     * @brief Evicts [evict_begin, evict_end) bytes within one already-queried mapped chunk [chunk_base, chunk_end).
     * Unmaps the whole chunk as a placeholder, splits off the kept before/after pieces, then remaps them.
     */
    void evict_chunk(HANDLE proc, char* chunk_base, char* chunk_end, char* evict_begin, char* evict_end) noexcept;

    /**
     * @brief Maps the file section in ≤1 GiB chunks into [base, base+actual_map_size) of an existing placeholder.
     * On failure, fully cleans up (unmaps mapped chunks and frees all split VA) and returns an empty vector.
     */
    std::vector<char*> map_file_chunks(const PlaceholderApis& api,
                                       HANDLE proc,
                                       char* base,
                                       size_t alloc_size,
                                       size_t actual_map_size,
                                       size_t aligned_offset);

    /**
     * @brief Creates a pagefile-backed anonymous section of tail_size bytes, copies tail_data_size file bytes
     * (from file_tail_offset) into it, and maps it read-only into tail_placeholder.
     * @return HandleHolder for the anonymous section (valid = success).
     *         On failure, tail_placeholder is left as a placeholder for the caller to free.
     */
    HandleHolder fill_anon_tail(const PlaceholderApis& api,
                                HANDLE proc,
                                char* tail_placeholder,
                                size_t tail_size,
                                size_t file_tail_offset,
                                size_t tail_data_size);

    void* m_data{};   //!< pointer exposed to callers
    size_t m_size{};  //!< user-visible byte count
    uint64_t m_id{std::numeric_limits<uint64_t>::max()};

    HandleHolder m_handle{};       //!< section object from CreateFileMappingW
    HandleHolder m_file_handle{};  //!< file HANDLE kept open to block DeleteFile (set-by-path only)
    HandleHolder m_anon_handle{};  //!< pagefile-backed section for the partial last 64KB tail (if needed)
    size_t m_aligned_offset{};     //!< gran-aligned file offset of placeholder base
    char* m_view_base{};  //!< base VA of the placeholder reservation (placeholder path) or MapViewOfFile base (legacy
                          //!< path); nullptr when unmapped
    size_t m_total_va_size{};      //!< total VA reservation size in bytes
    size_t m_file_mapped_size{};   //!< bytes of VA backed by the file section (≤ m_total_va_size; tail is anonymous)

    /**
     * @brief Guards all Unmap/VirtualAlloc2 calls in hint_release and try_remap_slot.
     * Must be recursive: hint_release may unmap a slot, then the VEH fires for an adjacent read on the same thread
     * and calls try_remap_slot, which also needs the same mutex.
     */
    std::recursive_mutex m_slot_mutex;
};

LONG NTAPI MmapVehRegistry::veh(PEXCEPTION_POINTERS ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    // Only handle read faults (ExceptionInformation[0] == 0).
    // Write/execute faults to read-only or no-access pages are not remappable and must propagate
    // to avoid an infinite fault loop (remap as PAGE_READONLY, re-execute write, fault again).
    if (ep->ExceptionRecord->ExceptionInformation[0] != 0) {
        return EXCEPTION_CONTINUE_SEARCH;
    }
    const auto fault_addr = static_cast<uintptr_t>(ep->ExceptionRecord->ExceptionInformation[1]);
    auto& reg = instance();
    std::shared_lock lock(reg.m_mtx);
    const Entry* e = reg.find(fault_addr);
    return (e && e->m_holder->try_remap_slot(fault_addr)) ? EXCEPTION_CONTINUE_EXECUTION : EXCEPTION_CONTINUE_SEARCH;
}

MapHolder::~MapHolder() {
    if (m_view_base && m_total_va_size != 0) {
        // Placeholder path: unregister VEH, unmap all views, free all VA allocations.
        const auto& api = PlaceholderApis::instance();
        const auto proc = GetCurrentProcess();

        // Unregister from VEH registry BEFORE unmapping so the VEH cannot
        // fire for this holder after we start tearing down its state.
        MmapVehRegistry::instance().remove(m_view_base);

        // Single-pass enumeration: unmap mapped regions and collect allocation bases.
        // This reduces VirtualQuery calls from 2N to N.
        std::vector<void*> allocations_to_free;

        if (std::lock_guard lock(m_slot_mutex); api.m_available) {
            char* current = m_view_base;
            const char* end = m_view_base + m_total_va_size;

            while (current < end) {
                MEMORY_BASIC_INFORMATION mbi{};
                if (!query_region(current, mbi))
                    break;
                if (mbi.Type == MEM_MAPPED) {
                    // Pass AllocationBase: UnmapViewOfFile2 requires the view's base address
                    // as originally mapped; BaseAddress may be a sub-region mid-view.
                    api.m_unmap_view_of_file2(proc, mbi.AllocationBase, MEM_PRESERVE_PLACEHOLDER);
                }
                void* alloc_base = mbi.AllocationBase;
                if (allocations_to_free.empty() || allocations_to_free.back() != alloc_base)
                    allocations_to_free.push_back(alloc_base);
                current = region_end(mbi);
            }
        }

        // Now free each unique allocation.
        for (void* alloc_base : allocations_to_free) {
            ::VirtualFree(alloc_base, 0, MEM_RELEASE);
        }
    } else if (m_view_base) {
        // Legacy path (MapViewOfFile without placeholders).
        ::UnmapViewOfFile(m_view_base);
    }
}

void MapHolder::set_id(HANDLE h, size_t offset, size_t size) {
    if (FILE_ID_INFO info; GetFileInformationByHandleEx(h, FileIdInfo, &info, sizeof(info))) {
        static_assert(sizeof(info.FileId) == 16);
        uint64_t fid_l, fid_r;
        std::memcpy(&fid_l, &info.FileId, sizeof(fid_l));
        std::memcpy(&fid_r, reinterpret_cast<const char*>(&info.FileId) + sizeof(fid_l), sizeof(fid_r));
        m_id = util::u64_hash_combine(offset, {size, info.VolumeSerialNumber, fid_l, fid_r});
    } else {
        // fallback to hashing the handle value if FileIdInfo is unavailable
        // e.g network filesystems or older Windows versions
        m_id = util::u64_hash_combine(offset, {size, std::hash<HANDLE>{}(h)});
    }
}

std::vector<char*> MapHolder::map_file_chunks(const PlaceholderApis& api,
                                              HANDLE proc,
                                              char* base,
                                              size_t alloc_size,
                                              size_t actual_map_size,
                                              size_t aligned_offset) {
    // 1 GiB per chunk: independently evictable/remappable via VEH, and stays below
    // MapViewOfFile3's ~2 GiB signed-32-bit limit when NtMapViewOfSectionEx is absent.
    constexpr size_t max_chunk = 1024 * 1024 * 1024;
    std::vector<char*> mapped;

    for (size_t pos = 0; pos < actual_map_size;) {
        const size_t chunk = std::min(actual_map_size - pos, max_chunk);
        // Split [base+pos, base+pos+chunk) from the remaining reservation.
        // The residual [base+pos+chunk, base+alloc_size) is always non-empty because
        // alloc_size > actual_map_size ensures dwSize < total allocation.
        if (!::VirtualFree(base + pos, chunk, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            // Split failed: [base+pos, base+alloc_size) is still one unsplit allocation.
            ::VirtualFree(base + pos, 0, MEM_RELEASE);
            for (char* cb : mapped) {
                api.m_unmap_view_of_file2(proc, cb, MEM_PRESERVE_PLACEHOLDER);
                ::VirtualFree(cb, 0, MEM_RELEASE);
            }
            return {};
        }
        auto v = replace_placeholder(api,
                                     m_handle.get(),
                                     proc,
                                     base + pos,
                                     static_cast<ULONG64>(aligned_offset + pos),
                                     chunk);
        if (v != base + pos) {
            // Map failed: split and residual are now two separate allocations.
            ::VirtualFree(base + pos, 0, MEM_RELEASE);
            ::VirtualFree(base + pos + chunk, 0, MEM_RELEASE);
            for (char* cb : mapped) {
                api.m_unmap_view_of_file2(proc, cb, MEM_PRESERVE_PLACEHOLDER);
                ::VirtualFree(cb, 0, MEM_RELEASE);
            }
            return {};
        }
        mapped.push_back(base + pos);
        pos += chunk;
    }
    return mapped;
}

HandleHolder MapHolder::fill_anon_tail(const PlaceholderApis& api,
                                       HANDLE proc,
                                       char* tail_placeholder,
                                       size_t tail_size,
                                       size_t file_tail_offset,
                                       size_t tail_data_size) {
    const auto hi = static_cast<DWORD>(tail_size >> 32);
    const auto lo = static_cast<DWORD>(tail_size & 0xFFFFFFFF);
    HandleHolder anon{::CreateFileMappingW(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE, hi, lo, nullptr)};
    if (!anon.valid())
        return HandleHolder{};

    if (tail_data_size > 0) {
        const auto off = static_cast<ULONG64>(file_tail_offset);
        const auto src = static_cast<const char*>(::MapViewOfFile(m_handle.get(),
                                                                  FILE_MAP_READ,
                                                                  static_cast<DWORD>(off >> 32),
                                                                  static_cast<DWORD>(off & 0xFFFFFFFF),
                                                                  tail_data_size));
        if (!src)
            return HandleHolder{};
        const auto dst = static_cast<char*>(::MapViewOfFile(anon.get(), FILE_MAP_WRITE, 0, 0, tail_data_size));
        if (!dst) {
            ::UnmapViewOfFile(src);
            return HandleHolder{};
        }
        std::memcpy(dst, src, tail_data_size);
        ::UnmapViewOfFile(dst);
        ::UnmapViewOfFile(src);
    }

    void* tv = api.m_map_view_of_file3(anon.get(),
                                       proc,
                                       tail_placeholder,
                                       0,
                                       tail_size,
                                       MEM_REPLACE_PLACEHOLDER,
                                       PAGE_READONLY,
                                       nullptr,
                                       0);
    return (tv != tail_placeholder) ? HandleHolder{} : std::move(anon);
}

bool MapHolder::try_placeholder_setup(size_t aligned_offset, size_t head_pad, size_t total_va_size, size_t file_size) {
    const auto& api = PlaceholderApis::instance();
    if (!api.m_available) {
        return false;
    }

    const auto gran = util::get_system_alloc_granularity();

    // NtMapViewOfSectionEx rejects a view that extends past raw file_size even by one byte
    // (STATUS_INVALID_VIEW_SIZE, 0xC000001F) — use raw file_size, not the page-rounded section size.
    // When the file is not a multiple of 64 KB, fill the sub-granule tail with an anonymous section.
    const size_t available = (file_size > aligned_offset) ? file_size - aligned_offset : 0;
    const size_t actual_map_size =
        (available >= total_va_size) ? total_va_size : util::align_size_down(available, gran);
    if (actual_map_size == 0) {
        return false;
    }

    const size_t tail_size = total_va_size - actual_map_size;
    const auto proc = GetCurrentProcess();

    // VirtualFree(MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER) requires dwSize < total reservation.
    // Always over-allocate by one granule: when tail_size == 0 it is freed immediately after the
    // first split; when tail_size > 0 it IS the tail placeholder.
    const size_t alloc_size = actual_map_size + (tail_size > 0 ? tail_size : gran);
    auto base = static_cast<char*>(api.m_virtual_alloc2(proc,
                                                        nullptr,
                                                        alloc_size,
                                                        MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                                        PAGE_NOACCESS,
                                                        nullptr,
                                                        0));
    if (!base) {
        return false;
    }

    // Map the file section in ≤1 GiB chunks into [base, base+actual_map_size).
    // Each chunk is independently evictable/remappable via VEH.
    // map_file_chunks fully cleans up (unmaps and frees all VA) on failure.
    auto mapped_chunks = map_file_chunks(api, proc, base, alloc_size, actual_map_size, aligned_offset);
    if (mapped_chunks.empty()) {
        return false;
    }

    if (tail_size > 0) {
        // Fill [base+actual_map_size, base+alloc_size) with a pagefile-backed anonymous section
        // containing the partial last file bytes followed by zero padding.
        const size_t tail_data = (head_pad + m_size > actual_map_size) ? head_pad + m_size - actual_map_size : 0;
        HandleHolder anon =
            fill_anon_tail(api, proc, base + actual_map_size, tail_size, aligned_offset + actual_map_size, tail_data);
        if (!anon.valid()) {
            // Cleanup for tail-setup failures: unmap all file chunks and release the tail/dummy placeholder.
            for (char* cb : mapped_chunks) {
                api.m_unmap_view_of_file2(proc, cb, MEM_PRESERVE_PLACEHOLDER);
                ::VirtualFree(cb, 0, MEM_RELEASE);
            }
            ::VirtualFree(base + actual_map_size, 0, MEM_RELEASE);
            return false;
        }
        m_anon_handle = std::move(anon);
    } else {
        // No tail: the extra dummy granule was only needed to satisfy the split requirement; release it now.
        ::VirtualFree(base + actual_map_size, 0, MEM_RELEASE);
    }

    m_view_base = base;
    m_total_va_size = total_va_size;
    m_file_mapped_size = actual_map_size;
    m_data = base + head_pad;
    MmapVehRegistry::instance().add(this, base, total_va_size);
    return true;
}

void MapHolder::legacy_setup(size_t aligned_offset, size_t head_pad, size_t size) {
    if (auto view = ::MapViewOfFile(m_handle.get(),
                                    FILE_MAP_READ,
                                    static_cast<DWORD>(aligned_offset >> 32),
                                    static_cast<DWORD>(aligned_offset & 0xFFFFFFFF),
                                    head_pad + size)) {
        m_view_base = static_cast<char*>(view);
        m_data = m_view_base + head_pad;
    } else {
        throw std::runtime_error{"MapViewOfFile failed: " + std::to_string(::GetLastError())};
    }
}

void MapHolder::setup(HANDLE file_handle, size_t offset, size_t size) {
    LARGE_INTEGER file_size_li{};
    if (!::GetFileSizeEx(file_handle, &file_size_li)) {
        throw std::runtime_error{"GetFileSizeEx failed: " + std::to_string(::GetLastError())};
    }
    const auto file_size = static_cast<size_t>(file_size_li.QuadPart);

    m_size = (size == auto_size) ? file_size - offset : size;
    if (offset + m_size > file_size || offset + m_size < offset) {
        throw std::runtime_error{"Requested mapping range exceeds file size"};
    }

    const auto gran = util::get_system_alloc_granularity();
    const auto& [r_offset, r_length, head_pad] = util::align_region(static_cast<uintptr_t>(offset), m_size, gran);
    m_aligned_offset = r_offset;
    // Round up to 64KB - required for VirtualFree MEM_PRESERVE_PLACEHOLDER split.
    const size_t total_va_size = util::align_size_up(r_length, gran);

    set_id(file_handle, offset, size);

    if (m_size == 0) {
        return;
    }

    // Create a read-only file-mapping object for the whole file.
    m_handle = HandleHolder{::CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr)};
    if (!m_handle.valid()) {
        throw std::runtime_error{"CreateFileMappingW failed: " + std::to_string(::GetLastError())};
    }

    // Prefer placeholder path for real RSS reduction; fall back to legacy.
    if (!try_placeholder_setup(m_aligned_offset, head_pad, total_va_size, file_size)) {
        legacy_setup(m_aligned_offset, head_pad, m_size);
    }
}

void MapHolder::set(const std::filesystem::path& path, size_t offset, size_t size) {
    auto fh = ::CreateFileW(path.c_str(),
                            GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_DELETE,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                            nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error{"Cannot open file: " + ov::util::path_to_string(path) +
                                 " error: " + std::to_string(::GetLastError())};
    }

    HandleHolder fh_holder{fh};
    setup(fh, offset, size);
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
    setup(owned.get(), offset, size);
    // owned goes out of scope here: file handle closed.
    // m_handle (section object) keeps the file data accessible independently.
}

bool MapHolder::remap_placeholder(HANDLE proc, char* base, size_t size) {
    const auto& api = PlaceholderApis::instance();
    const size_t va_offset = base - m_view_base;
    const ULONG64 file_off = static_cast<ULONG64>(m_aligned_offset + va_offset);

    void* v = replace_placeholder(api, m_handle.get(), proc, base, file_off, size);
    return v == base;
}

bool MapHolder::try_remap_slot(uintptr_t fault_addr) {
    const auto& api = PlaceholderApis::instance();
    if (!api.m_available || !m_view_base || m_total_va_size == 0) {
        return false;
    }

    const auto base = reinterpret_cast<uintptr_t>(m_view_base);

    if (fault_addr < base || fault_addr >= base + m_total_va_size) {
        return false;
    }

    std::lock_guard lock(m_slot_mutex);

    // Query the memory region containing the fault address.
    MEMORY_BASIC_INFORMATION mbi{};
    if (!query_region(reinterpret_cast<void*>(fault_addr), mbi))
        return false;

    // If already mapped (MEM_MAPPED), another thread beat us to it.
    if (mbi.Type == MEM_MAPPED)
        return true;

    // Use AllocationBase as the true placeholder start. VirtualQuery's BaseAddress may report a sub-region within
    // the placeholder if pages have different internal states (e.g. residual PTE differences after unmap),
    // but AllocationBase always reflects the actual placeholder boundary used by VirtualFree(MEM_PRESERVE_PLACEHOLDER).
    auto placeholder_base = static_cast<char*>(mbi.AllocationBase);

    // Determine full placeholder extent by scanning forward from AllocationBase. A single placeholder allocation
    // may be reported as multiple VirtualQuery regions with different page-level attributes,
    // but they share the same AllocationBase.
    size_t placeholder_size = 0;
    char* scan = placeholder_base;
    const char* va_end = m_view_base + m_total_va_size;
    while (scan < va_end) {
        MEMORY_BASIC_INFORMATION scan_mbi{};
        if (!query_region(scan, scan_mbi) || scan_mbi.AllocationBase != placeholder_base ||
            scan_mbi.Type == MEM_MAPPED) {
            break;
        }
        placeholder_size += scan_mbi.RegionSize;
        scan = region_end(scan_mbi);
    }
    if (placeholder_size == 0) {
        return false;
    }

    const auto proc = GetCurrentProcess();
    return remap_placeholder(proc, placeholder_base, placeholder_size);
}

std::pair<char*, char*> MapHolder::compute_evict_range(size_t offset, size_t size) const noexcept {
    if (!m_view_base || m_total_va_size == 0 || !PlaceholderApis::instance().m_available)
        return {};

    // Clamp offset and size to [0, m_size) to prevent size_t overflow in subsequent arithmetic.
    const size_t clamped_offset = std::min(offset, m_size);
    const size_t available = m_size - clamped_offset;
    const auto effective_size = (size == auto_size) ? available : std::min(size, available);
    if (effective_size == 0)
        return {};

    const auto gran = util::get_system_alloc_granularity();

    // Convert user [offset, size) to a VA range relative to m_view_base.
    const size_t head_pad = static_cast<size_t>(static_cast<char*>(m_data) - m_view_base);
    const size_t va_begin_raw = head_pad + clamped_offset;
    if (va_begin_raw >= m_total_va_size)
        return {};

    const size_t va_end = std::min(head_pad + clamped_offset + effective_size, m_total_va_size);

    // Outward gran-rounding: expand the eviction range to cover any partial granule.
    // Safe with VEH: a fault on any byte in the granule triggers a full-granule remap.
    // Cap at m_file_mapped_size so we never evict into the anonymous tail section
    // (which uses m_anon_handle, not m_handle, and would be remapped incorrectly).
    const size_t safe_begin = util::align_size_down(va_begin_raw, gran);
    const size_t safe_end = std::min(util::align_size_up(va_end, gran), m_file_mapped_size);
    if (safe_begin >= safe_end)
        return {};  // Range falls entirely within the anonymous tail — nothing to evict.

    return {m_view_base + safe_begin, m_view_base + safe_end};
}

void MapHolder::evict_chunk(HANDLE proc,
                            char* chunk_base,
                            char* chunk_end,
                            char* evict_begin,
                            char* evict_end) noexcept {
    const auto& api = PlaceholderApis::instance();
    if (!api.m_unmap_view_of_file2(proc, chunk_base, MEM_PRESERVE_PLACEHOLDER))
        return;

    const size_t before_size = static_cast<size_t>(evict_begin - chunk_base);
    const size_t after_size = static_cast<size_t>(chunk_end - evict_end);

    // Split the placeholder to isolate the evicted middle.
    if (before_size > 0 && after_size > 0) {
        if (::VirtualFree(chunk_base, before_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER))
            ::VirtualFree(evict_begin,
                          static_cast<size_t>(evict_end - evict_begin),
                          MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    } else if (before_size > 0) {
        ::VirtualFree(chunk_base, before_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    } else if (after_size > 0) {
        ::VirtualFree(evict_begin,
                      static_cast<size_t>(evict_end - evict_begin),
                      MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    }

    // Remap the before/after pieces so they remain accessible.
    if (before_size > 0) {
        remap_placeholder(proc, chunk_base, before_size);
    }
    if (after_size > 0) {
        remap_placeholder(proc, evict_end, after_size);
    }
}

void MapHolder::hint_evict(size_t offset, size_t size) noexcept {
    const auto [range_begin, range_end] = compute_evict_range(offset, size);
    if (!range_begin)
        return;

    const auto proc = GetCurrentProcess();
    std::lock_guard lock(m_slot_mutex);

    char* scan = range_begin;
    while (scan < range_end) {
        MEMORY_BASIC_INFORMATION mbi{};
        if (!query_region(scan, mbi))
            break;
        if (mbi.Type != MEM_MAPPED) {
            // Already evicted or non-file region (e.g. anonymous tail) — skip.
            scan = region_end(mbi);
            continue;
        }

        auto chunk_base = static_cast<char*>(mbi.AllocationBase);
        auto chunk_end = region_end(mbi);
        const char* max_end = m_view_base + m_total_va_size;

        // Scan forward to find the full extent of this mapped chunk
        // (handles the case where VirtualQuery splits a large view into multiple regions).
        for (auto s = chunk_end; s < max_end;) {
            MEMORY_BASIC_INFORMATION mbi_next{};
            if (!query_region(s, mbi_next) || mbi_next.Type != MEM_MAPPED || mbi_next.AllocationBase != chunk_base)
                break;
            chunk_end = region_end(mbi_next);
            s = chunk_end;
        }

        evict_chunk(proc, chunk_base, chunk_end, scan, std::min(range_end, chunk_end));
        scan = chunk_end;
    }
}

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::filesystem::path& path, size_t offset, size_t size) {
    auto holder = std::make_shared<MapHolder>();
    holder->set(path, offset, size);
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
