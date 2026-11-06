// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "prefetch_streambuf.hpp"

#include <algorithm>
#include <cstring>

#include "openvino/runtime/make_tensor.hpp"
#include "parallel_memcpy.hpp"

namespace ov::intel_gpu {

PrefetchStreamBuf::PrefetchStreamBuf(const ov::Tensor& tensor, bool prefetch)
    : m_begin(static_cast<const char*>(tensor.data())),
      m_current(m_begin),
      m_end(m_begin + tensor.get_byte_size()) {
    if (prefetch && ov::get_memory_hints(tensor)) {
        m_scheduler = std::make_unique<ov::util::SequentialPrefetchScheduler>(tensor);
    }
}

PrefetchStreamBuf::int_type PrefetchStreamBuf::underflow() {
    return m_current < m_end ? traits_type::to_int_type(*m_current) : traits_type::eof();
}

PrefetchStreamBuf::int_type PrefetchStreamBuf::uflow() {
    return m_current < m_end ? traits_type::to_int_type(*m_current++) : traits_type::eof();
}

std::streamsize PrefetchStreamBuf::showmanyc() {
    return m_current < m_end ? m_end - m_current : -1;
}

std::streamsize PrefetchStreamBuf::xsgetn(char_type* dst, std::streamsize count) {
    const auto n = std::min<std::streamsize>(count, m_end - m_current);
    if (n <= 0) {
        return 0;
    }
    const auto pos = static_cast<size_t>(m_current - m_begin);
    if (m_scheduler) {
        m_scheduler->advance(pos, pos + static_cast<size_t>(n));
    }
    // A single thread cannot saturate memory bandwidth when the pages are already cached.
    parallel_memcpy(dst, m_current, static_cast<size_t>(n));
    m_current += n;
    return n;
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode which) {
    if (!(which & std::ios_base::in)) {
        return pos_type(off_type(-1));
    }
    const off_type base = dir == std::ios_base::beg ? 0 : dir == std::ios_base::cur ? m_current - m_begin : m_end - m_begin;
    const off_type pos = base + off;
    if (pos < 0 || pos > m_end - m_begin) {
        return pos_type(off_type(-1));
    }
    m_current = m_begin + pos;
    return pos_type(pos);
}

PrefetchStreamBuf::pos_type PrefetchStreamBuf::seekpos(pos_type pos, std::ios_base::openmode which) {
    return seekoff(off_type(pos), std::ios_base::beg, which);
}

}  // namespace ov::intel_gpu
