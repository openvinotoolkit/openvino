// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prefetch_streambuf.hpp"

#include <algorithm>
#include <cstring>

#include "openvino/runtime/make_tensor.hpp"

namespace ov::intel_gpu {

PrefetchStreamBuf::PrefetchStreamBuf(const ov::Tensor& tensor, bool prefetch) {
    const auto begin = static_cast<char*>(const_cast<void*>(tensor.data()));
    setg(begin, begin, begin + tensor.get_byte_size());
    if (prefetch && ov::get_memory_hints(tensor)) {
        m_scheduler = std::make_unique<ov::util::SequentialPrefetchScheduler>(tensor);
    }
}

std::streamsize PrefetchStreamBuf::xsgetn(char_type* dst, std::streamsize count) {
    const auto n = std::min<std::streamsize>(count, egptr() - gptr());
    if (n <= 0) {
        return 0;
    }
    const auto pos = static_cast<size_t>(gptr() - eback());
    if (m_scheduler) {
        m_scheduler->advance(pos, pos + static_cast<size_t>(n));
    }
    std::memcpy(dst, gptr(), static_cast<size_t>(n));
    // gbump() takes int, reads may exceed 2 GB.
    setg(eback(), gptr() + n, egptr());
    return n;
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekoff(off_type off,
                                                       std::ios_base::seekdir dir,
                                                       std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    const off_type base = dir == std::ios_base::beg   ? 0
                          : dir == std::ios_base::cur ? gptr() - eback()
                                                      : egptr() - eback();
    const off_type pos = base + off;
    if (pos < 0 || pos > egptr() - eback()) {
        return pos_type(off_type(-1));
    }
    setg(eback(), eback() + pos, egptr());
    return pos_type(pos);
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(off_type(pos), std::ios_base::beg, which);
}

}  // namespace ov::intel_gpu
