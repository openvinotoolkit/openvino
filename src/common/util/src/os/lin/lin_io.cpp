// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <liburing.h>
#include <sys/mman.h>

#include <algorithm>
#include <cstdint>

#include "openvino/util/io.hpp"

namespace ov::util {

namespace {
constexpr bool is_io_uring_supported() {
#if defined ENABLE_IO_URING && defined MADV_POPULATE_READ
    return true;
#    else
    return false;
#    endif
}

void fallback_madvise(void* ptr, size_t size) noexcept {
    madvise(ptr, size, MADV_SEQUENTIAL);
    madvise(ptr, size, MADV_WILLNEED);
}
}  // namespace

bool io_populate_mmap(void* ptr, size_t size, size_t offset, size_t queue_depth) noexcept {
    if (ptr == nullptr || size == 0 || queue_depth == 0) {
        return false;
    }

    auto* const base = static_cast<uint8_t*>(ptr) + offset;

    if constexpr (is_io_uring_supported()) {
        const auto depth = static_cast<unsigned>(std::min<size_t>(queue_depth, 4096));
        const size_t chunk_size = (size + depth - 1) / depth;

        io_uring ring;
        if (io_uring_queue_init(depth, &ring, 0) < 0) {
            fallback_madvise(base, size);
            return true;
        }

        madvise(base, size, MADV_NORMAL);

        auto drain_cq = [&ring]() noexcept {
            io_uring_cqe* cqe;
            unsigned head;
            unsigned seen = 0;
            io_uring_for_each_cqe(&ring, head, cqe) {
                ++seen;
            }
            io_uring_cq_advance(&ring, seen);
        };

        size_t pending = 0;
        for (size_t shift = 0; shift < size;) {
            io_uring_sqe* const sqe = io_uring_get_sqe(&ring);
            const size_t len = std::min(chunk_size, size - shift);
            io_uring_prep_madvise(sqe, base + shift, static_cast<off_t>(len), MADV_POPULATE_READ);
            sqe->flags = 0;
            shift += len;
            ++pending;
        }
        if (pending > 0) {
            io_uring_submit_and_wait(&ring, static_cast<unsigned>(pending));
            drain_cq();
        }

        io_uring_queue_exit(&ring);
    } else {
        fallback_madvise(base, size);
    }
    return true;
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
