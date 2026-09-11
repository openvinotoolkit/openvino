// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator_mmap.hpp"

#if defined(_WIN32)

#    include <algorithm>
#    include <cstddef>
#    include <cstdint>
#    include <filesystem>
#    include <limits>
#    include <string>
#    include <system_error>

#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>

namespace ov {
namespace {
struct AllocationHeader {
    void* base = nullptr;
    size_t map_size = 0;
    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
};

size_t align_up(size_t value, size_t alignment) {
    OPENVINO_ASSERT(alignment > 0, "Alignment must be positive");
    OPENVINO_ASSERT(value <= (std::numeric_limits<size_t>::max)() - alignment + 1,
                    "Requested mmap allocation is too large");
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t get_page_size() {
    SYSTEM_INFO system_info{};
    ::GetSystemInfo(&system_info);
    return system_info.dwPageSize > 0 ? static_cast<size_t>(system_info.dwPageSize) : static_cast<size_t>(4096);
}

std::filesystem::path get_temporary_directory() {
    std::error_code error;
    auto temp_dir = std::filesystem::temp_directory_path(error);
    OPENVINO_ASSERT(!error, "Cannot get temporary directory for mmap constant storage: ", error.message());
    return temp_dir;
}

void check_available_space(size_t bytes) {
    const auto temp_dir = get_temporary_directory();
    std::error_code error;
    const auto space = std::filesystem::space(temp_dir, error);
    OPENVINO_ASSERT(!error,
                    "Cannot query available space in temporary directory for mmap constant storage: ",
                    error.message());
    OPENVINO_ASSERT(space.available >= bytes,
                    "Not enough available space in temporary directory '",
                    temp_dir.string(),
                    "' for mmap constant storage. Required ",
                    bytes,
                    " bytes, available ",
                    space.available,
                    " bytes.");
}

std::string get_windows_error_message(DWORD error) {
    return std::system_category().message(static_cast<int>(error));
}

HANDLE create_temporary_file(size_t bytes) {
    OPENVINO_ASSERT(bytes <= static_cast<size_t>((std::numeric_limits<LONGLONG>::max)()),
                    "Requested mmap allocation is too large");

    const auto temp_dir = get_temporary_directory();
    const DWORD process_id = ::GetCurrentProcessId();
    const DWORD thread_id = ::GetCurrentThreadId();
    const ULONGLONG tick_count = ::GetTickCount64();

    for (uint32_t attempt = 0; attempt < 1024; ++attempt) {
        const auto path = temp_dir / (L"openvino_mmap_" + std::to_wstring(process_id) + L"_" +
                                      std::to_wstring(thread_id) + L"_" + std::to_wstring(tick_count) + L"_" +
                                      std::to_wstring(attempt) + L".tmp");
        HANDLE file = ::CreateFileW(path.wstring().c_str(),
                                    GENERIC_READ | GENERIC_WRITE,
                                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                                    nullptr,
                                    CREATE_NEW,
                                    FILE_ATTRIBUTE_TEMPORARY | FILE_FLAG_DELETE_ON_CLOSE,
                                    nullptr);
        if (file == INVALID_HANDLE_VALUE) {
            const DWORD error = ::GetLastError();
            if (error == ERROR_FILE_EXISTS || error == ERROR_ALREADY_EXISTS) {
                continue;
            }
            OPENVINO_THROW("Cannot create temporary file for mmap constant storage in '",
                           temp_dir.string(),
                           "': ",
                           get_windows_error_message(error));
        }

        LARGE_INTEGER file_size{};
        file_size.QuadPart = static_cast<LONGLONG>(bytes);
        if (::SetFilePointerEx(file, file_size, nullptr, FILE_BEGIN) == 0 || ::SetEndOfFile(file) == 0) {
            const DWORD error = ::GetLastError();
            (void)::CloseHandle(file);
            OPENVINO_THROW("Cannot resize temporary file for mmap constant storage to ",
                           bytes,
                           " bytes: ",
                           get_windows_error_message(error));
        }

        return file;
    }

    OPENVINO_THROW("Cannot create unique temporary file for mmap constant storage in '", temp_dir.string(), "'");
}
}  // namespace

void* TemporaryFileBackedAllocator::allocate(size_t bytes, size_t alignment) {
    if (bytes == 0) {
        bytes = 1;
    }

    if (alignment == 0) {
        alignment = alignof(std::max_align_t);
    }
    OPENVINO_ASSERT((alignment & (alignment - 1)) == 0, "Alignment is not power of 2: ", alignment);

    const size_t min_alignment = std::max(sizeof(void*), alignment);
    const size_t page_size = get_page_size();
    OPENVINO_ASSERT(bytes <= (std::numeric_limits<size_t>::max)() - sizeof(AllocationHeader) - min_alignment + 1,
                    "Requested mmap allocation is too large");
    const size_t request = align_up(bytes + sizeof(AllocationHeader) + min_alignment - 1, page_size);
    check_available_space(request);

    HANDLE file = INVALID_HANDLE_VALUE;
    HANDLE mapping = nullptr;
    void* base = nullptr;
    try {
        file = create_temporary_file(request);
        const auto mapping_size = static_cast<unsigned long long>(request);
        mapping = ::CreateFileMappingW(file,
                                       nullptr,
                                       PAGE_READWRITE,
                                       static_cast<DWORD>(mapping_size >> 32),
                                       static_cast<DWORD>(mapping_size & 0xFFFFFFFFULL),
                                       nullptr);
        if (!mapping) {
            OPENVINO_THROW("CreateFileMappingW failed for temporary constant storage: ",
                           get_windows_error_message(::GetLastError()));
        }

        base = ::MapViewOfFile(mapping, FILE_MAP_ALL_ACCESS, 0, 0, request);
        if (!base) {
            OPENVINO_THROW("MapViewOfFile failed for temporary constant storage: ",
                           get_windows_error_message(::GetLastError()));
        }

        const auto base_address = reinterpret_cast<std::uintptr_t>(base);
        const auto start_address = base_address + sizeof(AllocationHeader);
        const auto aligned_address = align_up(static_cast<size_t>(start_address), min_alignment);
        auto* header = reinterpret_cast<AllocationHeader*>(aligned_address - sizeof(AllocationHeader));
        header->base = base;
        header->map_size = request;
        header->file = file;
        header->mapping = mapping;
        return reinterpret_cast<void*>(aligned_address);
    } catch (...) {
        if (base) {
            (void)::UnmapViewOfFile(base);
        }
        if (mapping) {
            (void)::CloseHandle(mapping);
        }
        if (file != INVALID_HANDLE_VALUE) {
            (void)::CloseHandle(file);
        }
        throw;
    }
}

void TemporaryFileBackedAllocator::deallocate(void* handle, size_t, size_t) noexcept {
    if (!handle) {
        return;
    }

    const auto* header_ptr = reinterpret_cast<AllocationHeader*>(reinterpret_cast<std::uintptr_t>(handle) -
                                                                 sizeof(AllocationHeader));
    const AllocationHeader header = *header_ptr;
    if (header.base) {
        (void)::UnmapViewOfFile(header.base);
    }
    if (header.mapping) {
        (void)::CloseHandle(header.mapping);
    }
    if (header.file != INVALID_HANDLE_VALUE) {
        (void)::CloseHandle(header.file);
    }
}

}  // namespace ov

#endif  // defined(_WIN32)
