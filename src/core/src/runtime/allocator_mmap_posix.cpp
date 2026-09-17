// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/allocator_mmap.hpp"

#if defined(__unix__) || defined(__APPLE__)

#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>

#    include <algorithm>
#    include <cerrno>
#    include <cstddef>
#    include <cstdint>
#    include <cstring>
#    include <filesystem>
#    include <limits>
#    include <string>
#    include <system_error>

namespace ov {
namespace {
struct AllocationHeader {
    void* base = nullptr;
    size_t map_size = 0;
    int fd = -1;
};

size_t align_up(size_t value, size_t alignment) {
    OPENVINO_ASSERT(alignment > 0, "Alignment must be positive");
    OPENVINO_ASSERT(value <= (std::numeric_limits<size_t>::max)() - alignment + 1,
                    "Requested mmap allocation is too large");
    return ((value + alignment - 1) / alignment) * alignment;
}

size_t get_page_size() {
    static const size_t page_size = [] {
        const long value = ::sysconf(_SC_PAGESIZE);
        return value > 0 ? static_cast<size_t>(value) : static_cast<size_t>(4096);
    }();
    return page_size;
}

std::filesystem::path get_temporary_directory() {
    std::error_code error;
    auto temp_dir = std::filesystem::temp_directory_path(error);
    OPENVINO_ASSERT(!error, "Cannot get temporary directory for mmap constant storage: ", error.message());
    return temp_dir;
}

void check_available_space(const std::filesystem::path& temp_dir, size_t bytes) {
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

int create_temporary_file(const std::filesystem::path& temp_dir, size_t bytes) {
    std::string path_template = (temp_dir / "openvino_mmap_XXXXXX").string();
    int fd = ::mkstemp(path_template.data());
    if (fd < 0) {
        OPENVINO_THROW("Cannot create temporary file for mmap constant storage in '",
                       temp_dir.string(),
                       "': ",
                       std::strerror(errno));
    }
    (void)::unlink(path_template.c_str());

    const int flags = ::fcntl(fd, F_GETFD);
    if (flags >= 0) {
        (void)::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    }

    OPENVINO_ASSERT(bytes <= static_cast<size_t>((std::numeric_limits<off_t>::max)()),
                    "Requested mmap allocation is too large");
#    if defined(__linux__)
    // Blocks must be reserved here: a sparse file would raise SIGBUS on write once the filesystem fills up.
    if (const int error = ::posix_fallocate(fd, 0, static_cast<off_t>(bytes)); error != 0) {
        const auto error_message = std::strerror(error);
        (void)::close(fd);
        OPENVINO_THROW("Cannot reserve ",
                       bytes,
                       " bytes in temporary directory for mmap constant storage: ",
                       error_message);
    }
#    else
    if (::ftruncate(fd, static_cast<off_t>(bytes)) != 0) {
        const auto error_message = std::strerror(errno);
        (void)::close(fd);
        OPENVINO_THROW("Cannot resize temporary file for mmap constant storage to ", bytes, " bytes: ", error_message);
    }
#    endif
    return fd;
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
    const auto temp_dir = get_temporary_directory();
    check_available_space(temp_dir, request);

    int fd = -1;
    void* base = MAP_FAILED;
    try {
        fd = create_temporary_file(temp_dir, request);
        base = ::mmap(nullptr, request, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            OPENVINO_THROW("mmap() failed for temporary constant storage: ",
                           std::strerror(errno),
                           ". Every mmap-backed constant needs its own mapping, so a small "
                           "CONSTANT_OFFLOAD_MIN_SIZE can exceed the vm.max_map_count limit. "
                           "Increase CONSTANT_OFFLOAD_MIN_SIZE or raise vm.max_map_count.");
        }

        const auto base_address = reinterpret_cast<std::uintptr_t>(base);
        const auto start_address = base_address + sizeof(AllocationHeader);
        const auto aligned_address = align_up(static_cast<size_t>(start_address), min_alignment);
        auto* header = reinterpret_cast<AllocationHeader*>(aligned_address - sizeof(AllocationHeader));
        header->base = base;
        header->map_size = request;
        header->fd = fd;
        return reinterpret_cast<void*>(aligned_address);
    } catch (...) {
        if (base != MAP_FAILED) {
            (void)::munmap(base, request);
        }
        if (fd >= 0) {
            (void)::close(fd);
        }
        throw;
    }
}

void TemporaryFileBackedAllocator::deallocate(void* handle, size_t, size_t) noexcept {
    if (!handle) {
        return;
    }

    const auto* header_ptr =
        reinterpret_cast<AllocationHeader*>(reinterpret_cast<std::uintptr_t>(handle) - sizeof(AllocationHeader));
    const AllocationHeader header = *header_ptr;
    if (header.base && header.map_size) {
        (void)::munmap(header.base, header.map_size);
    }
    if (header.fd >= 0) {
        (void)::close(header.fd);
    }
}

}  // namespace ov

#endif  // defined(__unix__) || defined(__APPLE__)
