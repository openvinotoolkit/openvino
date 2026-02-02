// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
/// \file allocator_mmap.hpp
/// \brief Public API for an allocator backed by OS virtual memory mapping.


#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <limits>

#include <sys/mman.h>
#include <unistd.h>

#include "openvino/core/except.hpp"

namespace ov {

inline size_t ov_get_page_size() {
    const long ps = ::sysconf(_SC_PAGESIZE);
    return ps > 0 ? static_cast<size_t>(ps) : static_cast<size_t>(4096);
}

inline size_t ov_align_up(size_t v, size_t a) {
    return (v + (a - 1)) & ~(a - 1);
}

struct MmapAnonymousAllocator {
    struct Header {
        void* base;
        size_t map_size;
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

        void* base = ::mmap(nullptr, request, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) {
            OPENVINO_THROW("mmap(MAP_ANONYMOUS) failed");
        }

        const auto base_u = reinterpret_cast<std::uintptr_t>(base);
        const auto start = base_u + sizeof(Header);
        const auto aligned = ov_align_up(static_cast<size_t>(start), min_align);

        auto* hdr = reinterpret_cast<Header*>(static_cast<std::uintptr_t>(aligned) - sizeof(Header));
        hdr->base = base;
        hdr->map_size = request;

        return reinterpret_cast<void*>(static_cast<std::uintptr_t>(aligned));
    }

    void deallocate(void* handle,
                    size_t bytes,
                    size_t alignment) noexcept {
        if (!handle) {
            return;
        }

        auto* hdr = reinterpret_cast<Header*>(reinterpret_cast<std::uintptr_t>(handle) - sizeof(Header));
        if (hdr->base && hdr->map_size) {
            (void)::munmap(hdr->base, hdr->map_size);
        }
    }

    bool is_equal(const MmapAnonymousAllocator&) const {
        return true;
    }
};

} // namespace ov