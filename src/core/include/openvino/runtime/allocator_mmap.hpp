// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/// \file allocator_mmap.hpp
/// \brief Public API for an allocator backed by OS virtual memory mapping.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "openvino/core/except.hpp"

#if defined(__unix__) || defined(__APPLE__)
#    define OV_HAVE_POSIX_MMAP 1
#    include <fcntl.h>
#    include <sys/mman.h>
#    include <unistd.h>
#else
#    define OV_HAVE_POSIX_MMAP 0
#endif

namespace ov {

#if OV_HAVE_POSIX_MMAP

inline size_t ov_get_page_size() {
    const long ps = ::sysconf(_SC_PAGESIZE);
    return ps > 0 ? static_cast<size_t>(ps) : static_cast<size_t>(4096);
}

inline size_t ov_align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

inline int ov_create_tmp_file_fd() {
    // mkstemp requires a writable buffer.
    char path[] = "/tmp/openvino_mmap_XXXXXX";
    int fd = ::mkstemp(path);
    if (fd < 0) {
        OPENVINO_THROW("mkstemp() failed");
    }

    // Remove directory entry right away; file will be removed when last fd is closed.
    (void)::unlink(path);

    // Best-effort CLOEXEC.
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags >= 0) {
        (void)::fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    }

    return fd;
}

struct MmapAnonymousAllocator {
    struct Header {
        void* base;
        size_t map_size;
        int fd;
    };

    void* allocate(size_t bytes, size_t alignment) {
        if (bytes == 0) {
            bytes = 1;
        }

        if (alignment == 0) {
            alignment = alignof(std::max_align_t);
        }

        OPENVINO_ASSERT(alignment && (alignment & (alignment - 1)) == 0, "Alignment is not power of 2: ", alignment);

        const size_t page = ov_get_page_size();
        const size_t min_align = std::max(alignof(void*), alignment);

        OPENVINO_ASSERT(min_align <= (std::numeric_limits<size_t>::max)() - bytes - sizeof(Header),
                        "Requested allocation is too large");

        size_t request = bytes + sizeof(Header) + (min_align - 1);
        request = ov_align_up(request, page);

        int fd = -1;
        void* base = MAP_FAILED;

        try {
            fd = ov_create_tmp_file_fd();

            if (::ftruncate(fd, static_cast<off_t>(request)) != 0) {
                OPENVINO_THROW("ftruncate() failed");
            }

            base = ::mmap(nullptr, request, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            if (base == MAP_FAILED) {
                OPENVINO_THROW("mmap(tmpfile) failed");
            }

            const auto base_u = reinterpret_cast<std::uintptr_t>(base);
            const auto start = base_u + sizeof(Header);
            const auto aligned = ov_align_up(static_cast<size_t>(start), min_align);

            auto* hdr = reinterpret_cast<Header*>(static_cast<std::uintptr_t>(aligned) - sizeof(Header));
            hdr->base = base;
            hdr->map_size = request;
            hdr->fd = fd;

            std::cout << "Allocated " << bytes << " bytes at " << reinterpret_cast<void*>(aligned)
                      << " using mmap allocator." << std::endl;
            return reinterpret_cast<void*>(static_cast<std::uintptr_t>(aligned));
        } catch (...) {
            if (fd >= 0) {
                (void)::close(fd);
            }
            if (base != MAP_FAILED) {
                (void)::munmap(base, request);
            }
            throw;
        }
    }

    void deallocate(void* handle, size_t /*bytes*/, size_t /*alignment*/) noexcept {
        if (!handle) {
            return;
        }
        std::cout << "Deallocating mmap memory: " << handle << std::endl;

        auto* hdr = reinterpret_cast<Header*>(reinterpret_cast<std::uintptr_t>(handle) - sizeof(Header));
        if (hdr->fd >= 0) {
            (void)::close(hdr->fd);
        }
        if (hdr->base && hdr->map_size) {
            (void)::munmap(hdr->base, hdr->map_size);
        }
    }

    bool is_equal(const MmapAnonymousAllocator&) const {
        return true;
    }
};

#else  // OV_HAVE_POSIX_MMAP

using MmapAnonymousAllocator = DefaultAllocator;

#endif  // OV_HAVE_POSIX_MMAP

}  // namespace ov