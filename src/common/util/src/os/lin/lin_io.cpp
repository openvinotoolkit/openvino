// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>

#include "openvino/util/io.hpp"

#if defined ENABLE_IO_URING && defined MADV_POPULATE_READ
#    include <liburing.h>
#    define USE_IO_URING
#endif

namespace ov::util {

namespace {
void fallback_madvise(void* ptr, size_t size) noexcept {
    madvise(ptr, size, MADV_SEQUENTIAL);
    madvise(ptr, size, MADV_WILLNEED);
}
}  // namespace

bool io_populate_mmap(void* ptr, size_t size, size_t offset, size_t queue_depth) noexcept {
    if (ptr == nullptr || size == 0) {
        return false;
    }

    const auto page_size = static_cast<size_t>(sysconf(_SC_PAGESIZE));
    const auto raw = static_cast<uint8_t*>(ptr) + offset;
    const auto prefix = reinterpret_cast<uintptr_t>(raw) % page_size;
    const auto base = raw - prefix;
    const auto aligned_size = size + prefix;

#if defined USE_IO_URING
    queue_depth = std::max<size_t>(1, queue_depth);
    queue_depth = std::min<size_t>(queue_depth, 4096);
    // Round chunk_size up to a page multiple so every subsequent chunk address stays aligned.
    const size_t raw_chunk = (aligned_size + queue_depth - 1) / queue_depth;
    const size_t chunk_size = ((raw_chunk + page_size - 1) / page_size) * page_size;

    io_uring ring;
    if (io_uring_queue_init(static_cast<unsigned>(queue_depth), &ring, 0) < 0) {
        fallback_madvise(base, aligned_size);
        return true;
    }

    madvise(base, aligned_size, MADV_NORMAL);

    const auto drain_cq = [&ring]() noexcept -> bool {
        io_uring_cqe* cqe;
        unsigned head;
        unsigned seen = 0;
        bool all_ok = true;
        io_uring_for_each_cqe(&ring, head, cqe) {
            if (cqe->res < 0) {
                all_ok = false;
            }
            ++seen;
        }
        io_uring_cq_advance(&ring, seen);
        return all_ok;
    };

    size_t pending = 0;
    bool sqe_error = false;
    for (size_t shift = 0; shift < aligned_size;) {
        io_uring_sqe* const sqe = io_uring_get_sqe(&ring);
        if (sqe == nullptr) {
            sqe_error = true;
            break;
        }
        const auto len = std::min(chunk_size, aligned_size - shift);
        io_uring_prep_madvise(sqe, base + shift, static_cast<off_t>(len), MADV_POPULATE_READ);
        sqe->flags = 0;
        shift += len;
        ++pending;
    }

    bool uring_ok = false;
    if (!sqe_error && pending > 0) {
        const auto ret = io_uring_submit_and_wait(&ring, static_cast<unsigned>(pending));
        if (ret >= 0) {
            uring_ok = drain_cq();
        }
    }

    io_uring_queue_exit(&ring);

    if (!uring_ok) {
        fallback_madvise(base, aligned_size);
    }

    return true;
#else
    fallback_madvise(base, aligned_size);
    return true;
#endif
}

std::error_code io_read_into(FileHandle /*handle*/,
                             void* /*dst*/,
                             size_t /*file_offset*/,
                             size_t /*size*/,
                             size_t /*queue_depth*/) noexcept {
    // CVS-186707
    return std::make_error_code(std::errc::function_not_supported);
}
}  // namespace ov::util
