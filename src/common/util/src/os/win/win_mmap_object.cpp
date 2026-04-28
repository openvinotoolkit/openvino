// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <vector>

#include "openvino/util/common_util.hpp"
#include "openvino/util/file_util.hpp"
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
        if (auto it = m_ranges.upper_bound(fault_addr); it != m_ranges.begin()) {
            --it;
            const auto& e = it->second;
            const auto end = reinterpret_cast<uintptr_t>(e.m_base) + e.m_total_va_size;
            return (fault_addr < end) ? &e : nullptr;
        } else {
            return nullptr;
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
        return m_handle != INVALID_HANDLE_VALUE;
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
    bool try_placeholder_setup(HANDLE file_handle, size_t aligned_offset, size_t head_pad, size_t total_va_size);

    /** bbrief Legacy single-call MapViewOfFile path (no partial-release support). */
    void legacy_setup(HANDLE file_handle, size_t aligned_offset, size_t head_pad, size_t size);

private:
    void* m_data = nullptr;  //!< pointer exposed to callers
    size_t m_size = 0;       //!< user-visible byte count
    uint64_t m_id = std::numeric_limits<uint64_t>::max();

    HandleHolder m_handle{};    //!< file HANDLE (owned when opened by path)
    HandleHolder m_mapping{};   //!< file-mapping HANDLE
    size_t m_aligned_offset{};  //!< gran-aligned file offset of placeholder base
    void* m_mapped_view{};      //!< non-null only in legacy path
    char* m_view_base{};        //!< base VA of the full placeholder reservation (nullptr for legacy path)
    size_t m_total_va_size{};   //!< total VA reservation size in bytes

    /**
     * @brief Guards all Unmap/VirtualAlloc2 calls in hint_release and try_remap_slot.
     * Must be recursive: hint_release may unmap a slot, then the VEH fires for an adjacent read on the same thread
     * and calls try_remap_slot, which also needs the same mutex.
     */
    std::recursive_mutex m_slot_mutex;

    bool is_mapped(void* va) const {
        MEMORY_BASIC_INFORMATION mbi{};
        const auto status = ::VirtualQuery(va, &mbi, sizeof(mbi));
        return status == 0 ? false : mbi.Type == MEM_MAPPED;
    }
};

LONG NTAPI MmapVehRegistry::veh(PEXCEPTION_POINTERS ep) {
    if (ep->ExceptionRecord->ExceptionCode != EXCEPTION_ACCESS_VIOLATION) {
        return EXCEPTION_CONTINUE_SEARCH;
    } else {
        // ExceptionInformation[0] = 0 (read) or 1 (write); [1] = faulting address.
        const auto fault_addr = static_cast<uintptr_t>(ep->ExceptionRecord->ExceptionInformation[1]);
        auto& reg = instance();
        std::shared_lock lock(reg.m_mtx);
        const Entry* e = reg.find(fault_addr);
        return (e != nullptr && e->m_holder->try_remap_slot(fault_addr)) ? EXCEPTION_CONTINUE_EXECUTION
                                                                         : EXCEPTION_CONTINUE_SEARCH;
    }
}

MapHolder::~MapHolder() {
    if (m_view_base) {
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
                if (::VirtualQuery(current, &mbi, sizeof(mbi)) == 0)
                    break;

                // Unmap if file-backed, converting it back to a placeholder.
                if (mbi.Type == MEM_MAPPED) {
                    api.m_unmap_view_of_file2(proc, mbi.BaseAddress, MEM_PRESERVE_PLACEHOLDER);
                }

                // Collect unique allocation bases to avoid double-free.
                void* alloc_base = mbi.AllocationBase;
                if (allocations_to_free.empty() || allocations_to_free.back() != alloc_base) {
                    allocations_to_free.push_back(alloc_base);
                }

                current = static_cast<char*>(mbi.BaseAddress) + mbi.RegionSize;
            }
        }

        // Now free each unique allocation.
        for (void* alloc_base : allocations_to_free) {
            ::VirtualFree(alloc_base, 0, MEM_RELEASE);
        }

        m_view_base = nullptr;
    } else if (m_mapped_view) {
        ::UnmapViewOfFile(m_mapped_view);
        m_mapped_view = nullptr;
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
        throw std::runtime_error{"Cannot obtain file id info for handle " +
                                 std::to_string(reinterpret_cast<uint64_t>(h))};
    }
}

bool MapHolder::try_placeholder_setup(HANDLE file_handle,
                                      size_t aligned_offset,
                                      size_t head_pad,
                                      size_t total_va_size) {
    const auto& api = PlaceholderApis::instance();
    if (!api.m_available) {
        return false;
    }

    const auto proc = GetCurrentProcess();

    // 1. Reserve the full VA range as a single placeholder.
    auto base = static_cast<char*>(api.m_virtual_alloc2(proc,
                                                        nullptr,
                                                        total_va_size,
                                                        MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                                                        PAGE_NOACCESS,
                                                        nullptr,
                                                        0));
    if (!base)
        return false;

    // 2. Map the entire range as ONE contiguous file view.
    //    No per-slot splitting - allows unmapping arbitrary sub-ranges later.
    auto v = api.m_map_view_of_file3(m_mapping.get(),
                                     proc,
                                     base,
                                     static_cast<ULONG64>(aligned_offset),
                                     total_va_size,
                                     MEM_REPLACE_PLACEHOLDER,
                                     PAGE_READONLY,
                                     nullptr,
                                     0);

    if (v != base) {
        // Cleanup: free the placeholder reservation.
        ::VirtualFree(base, 0, MEM_RELEASE);
        return false;
    }

    m_view_base = base;
    m_total_va_size = total_va_size;
    m_data = base + head_pad;

    // Register with the VEH registry so released ranges can be re-mapped on fault.
    MmapVehRegistry::instance().add(this, base, total_va_size);
    return true;
}

void MapHolder::legacy_setup(HANDLE file_handle, size_t aligned_offset, size_t head_pad, size_t size) {
    if (auto view = ::MapViewOfFile(m_mapping.get(),
                                    FILE_MAP_READ,
                                    static_cast<DWORD>(aligned_offset >> 32),
                                    static_cast<DWORD>(aligned_offset & 0xFFFFFFFF),
                                    head_pad + size)) {
        m_mapped_view = view;
        m_data = static_cast<char*>(view) + head_pad;

    } else {
        throw std::runtime_error{"MapViewOfFile failed: " + std::to_string(::GetLastError())};
    }
}

void MapHolder::setup(HANDLE file_handle, size_t offset, size_t size) {
    m_size = size;
    const auto gran = util::get_system_alloc_granularity();
    m_aligned_offset = (offset / gran) * gran;
    const size_t head_pad = offset - m_aligned_offset;
    const size_t total_va_size = ((head_pad + m_size + gran - 1) / gran) * gran;

    // Create a read-only file-mapping object (no size limit needed for read).
    m_mapping = HandleHolder{::CreateFileMappingW(file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr)};
    if (!m_mapping.valid()) {
        throw std::runtime_error{"CreateFileMappingW failed: " + std::to_string(::GetLastError())};
    }

    set_id(file_handle, offset, size);

    // Prefer placeholder path for real RSS reduction; fall back to legacy.
    if (!try_placeholder_setup(file_handle, m_aligned_offset, head_pad, total_va_size)) {
        legacy_setup(file_handle, m_aligned_offset, head_pad, size);
    }
}

void MapHolder::set(const std::filesystem::path& path, size_t offset, size_t size) {
    auto fh = ::CreateFileW(path.c_str(),
                            GENERIC_READ,
                            FILE_SHARE_READ,
                            nullptr,
                            OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL | FILE_FLAG_RANDOM_ACCESS,
                            nullptr);
    if (fh == INVALID_HANDLE_VALUE) {
        throw std::runtime_error{"Cannot open file: " + path.string() + " error: " + std::to_string(::GetLastError())};
    }

    m_handle = HandleHolder{fh};

    // If size is auto_size (default), map the whole file from offset.
    if (size == auto_size) {
        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(fh, &file_size))
            throw std::runtime_error{"GetFileSizeEx failed: " + std::to_string(::GetLastError())};
        size = static_cast<size_t>(file_size.QuadPart) - offset;
    }

    setup(fh, offset, size);
}

void MapHolder::set_from_handle(FileHandle handle, size_t offset, size_t size) {
    if (handle == INVALID_HANDLE_VALUE || handle == nullptr)
        throw std::runtime_error{"Invalid handle provided to load_mmap_object"};

    // If size is auto_size (default), map the whole file from offset.
    if (size == auto_size) {
        LARGE_INTEGER file_size{};
        if (!::GetFileSizeEx(static_cast<HANDLE>(handle), &file_size))
            throw std::runtime_error{"GetFileSizeEx failed: " + std::to_string(::GetLastError())};
        size = static_cast<size_t>(file_size.QuadPart) - offset;
    }

    // We do NOT take ownership of the caller's handle.
    setup(static_cast<HANDLE>(handle), offset, size);
}

bool MapHolder::remap_placeholder(HANDLE proc, char* base, size_t size) {
    const auto& api = PlaceholderApis::instance();
    const size_t va_offset = base - m_view_base;
    const ULONG64 file_off = static_cast<ULONG64>(m_aligned_offset + va_offset);

    void* v = api.m_map_view_of_file3(m_mapping.get(),
                                      proc,
                                      base,
                                      file_off,
                                      size,
                                      MEM_REPLACE_PLACEHOLDER,
                                      PAGE_READONLY,
                                      nullptr,
                                      0);
    return v == base;
}

bool MapHolder::try_remap_slot(uintptr_t fault_addr) {
    const auto& api = PlaceholderApis::instance();
    if (!api.m_available || !m_view_base) {
        return false;
    }

    const auto base = reinterpret_cast<uintptr_t>(m_view_base);

    if (fault_addr < base || fault_addr >= base + m_total_va_size) {
        return false;
    }

    std::lock_guard lock(m_slot_mutex);

    // Query the memory region containing the fault address.
    MEMORY_BASIC_INFORMATION mbi{};
    if (::VirtualQuery(reinterpret_cast<void*>(fault_addr), &mbi, sizeof(mbi)) == 0) {
        return false;
    }

    // If already mapped (MEM_MAPPED), another thread beat us to it.
    if (mbi.Type == MEM_MAPPED)
        return true;

    // Not mapped - must be a placeholder.
    // Query gives us the exact placeholder region bounds (the region previously unmapped by hint_release).
    const auto placeholder_base = static_cast<char*>(mbi.BaseAddress);
    const size_t placeholder_size = mbi.RegionSize;

    const auto proc = GetCurrentProcess();
    return remap_placeholder(proc, placeholder_base, placeholder_size);
}

void MapHolder::hint_evict(size_t offset, size_t size) noexcept {
    if (!m_view_base) {
        return;
    }

    const auto effective_size = (size == auto_size) ? m_size - std::min(offset, m_size) : size;
    if (effective_size == 0) {
        return;
    }

    const auto& api = PlaceholderApis::instance();
    if (!api.m_available) {
        return;
    }

    const auto gran = util::get_system_alloc_granularity();

    // Convert user [offset, size) to VA range, then inward gran-align.
    const size_t head_pad = static_cast<size_t>(static_cast<char*>(m_data) - m_view_base);
    const size_t va_begin_raw = head_pad + offset;
    const size_t va_end_raw = head_pad + offset + effective_size;

    if (va_begin_raw >= m_total_va_size)
        return;

    const size_t va_end = std::min(va_end_raw, m_total_va_size);
    const size_t safe_begin = (va_begin_raw + gran - 1u) & ~(gran - 1u);
    const size_t safe_end = va_end & ~(gran - 1u);

    if (safe_begin >= safe_end)
        return;

    const auto proc = GetCurrentProcess();
    char* unmap_va = m_view_base + safe_begin;
    size_t unmap_size = safe_end - safe_begin;

    std::lock_guard lock(m_slot_mutex);

    MEMORY_BASIC_INFORMATION mbi{};
    if (::VirtualQuery(unmap_va, &mbi, sizeof(mbi)) == 0 || mbi.Type != MEM_MAPPED) {
        return;
    }

    // Find the contiguous mapped view containing unmap_va.
    auto view_base = static_cast<char*>(mbi.AllocationBase);

    // Scan forward to find the full extent of the view.
    auto view_end = static_cast<char*>(mbi.BaseAddress) + mbi.RegionSize;
    auto max_end = m_view_base + m_total_va_size;

    for (auto scan = view_end; scan < max_end;) {
        MEMORY_BASIC_INFORMATION mbi_next{};
        if (::VirtualQuery(scan, &mbi_next, sizeof(mbi_next)) == 0 || mbi_next.Type != MEM_MAPPED ||
            mbi_next.AllocationBase != view_base)
            break;
        view_end = static_cast<char*>(mbi_next.BaseAddress) + mbi_next.RegionSize;
        scan = view_end;
    }

    // Unmap the entire contiguous view, converting it to one placeholder.
    if (!api.m_unmap_view_of_file2(proc, view_base, MEM_PRESERVE_PLACEHOLDER))
        return;

    // Calculate region boundaries (all gran-aligned).
    auto after_base = unmap_va + unmap_size;
    size_t before_size = unmap_va - view_base;
    size_t after_size = view_end - after_base;

    // Split the placeholder into independent regions.
    if (before_size > 0 && after_size > 0) {
        // 3-way split: before | middle (unmapped) | after
        if (::VirtualFree(view_base, before_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)) {
            ::VirtualFree(unmap_va, unmap_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
        }
    } else if (before_size > 0) {
        // 2-way split: before | middle (unmapped)
        ::VirtualFree(view_base, before_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    } else if (after_size > 0) {
        // 2-way split: middle (unmapped) | after
        ::VirtualFree(unmap_va, unmap_size, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
    }

    // Remap before/after pieces to keep them accessible.
    if (before_size > 0) {
        remap_placeholder(proc, view_base, before_size);
    }

    if (after_size > 0) {
        remap_placeholder(proc, after_base, after_size);
    }
}

std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::filesystem::path& path, size_t offset, size_t size) {
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
