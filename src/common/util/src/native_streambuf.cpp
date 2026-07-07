// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/native_streambuf.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <utility>

#include "openvino/util/memory.hpp"
#include "openvino/util/parallel_io.hpp"

namespace ov::util {

namespace {
// Window size must be a multiple of the system page size.
constexpr size_t window_alignment = min_page_alignment;
// Any caller-supplied window larger than max_window is silently clamped (e.g. a 16 GB value becomes 32 MiB).
constexpr size_t max_window = 32UL * 1024 * 1024;
}  // namespace

NativeStreamBuf::NativeStreamBuf(NativeStreamBuf&& other) noexcept
    : std::streambuf(other),  // copies get/put area pointers — they point into other.m_window, which we steal below
      m_handle(std::exchange(other.m_handle, ov::invalid_handle)),
      m_cursor(other.m_cursor),
      m_start(other.m_start),
      m_end(other.m_end),
      m_window_capacity(std::exchange(other.m_window_capacity, 0)),
      m_bypass_size(other.m_bypass_size),
      m_window(std::exchange(other.m_window, nullptr)) {
    other.setg(nullptr, nullptr, nullptr);  // get-area still points into the stolen window — reset the source
}

NativeStreamBuf& NativeStreamBuf::operator=(NativeStreamBuf&& other) noexcept {
    if (this != &other) {
        swap(other);  // old *this content migrates to other; freed by other's destructor
    }
    return *this;
}

void NativeStreamBuf::swap(NativeStreamBuf& other) noexcept {
    std::streambuf::swap(other);  // swaps get/put area pointers
    std::swap(m_handle, other.m_handle);
    std::swap(m_cursor, other.m_cursor);
    std::swap(m_start, other.m_start);
    std::swap(m_end, other.m_end);
    std::swap(m_window_capacity, other.m_window_capacity);
    std::swap(m_bypass_size, other.m_bypass_size);
    std::swap(m_window, other.m_window);
}

NativeStreamBuf::NativeStreamBuf() noexcept : NativeStreamBuf(ov::invalid_handle, 0, 0) {}

NativeStreamBuf::NativeStreamBuf(FileHandle handle,
                                 std::streamoff offset,
                                 std::streamoff size,
                                 size_t window,
                                 size_t threshold)
    : m_handle(handle),
      m_cursor(offset),
      m_start(offset),
      m_end(offset + size),
      m_window_capacity(align_size_up(std::clamp(window, window_alignment, max_window), window_alignment)),
      m_bypass_size(threshold) {
    assert(offset >= 0);
    assert(size >= 0);
}

NativeStreamBuf::~NativeStreamBuf() {
    aligned_free(m_window);
}

bool NativeStreamBuf::allocate_window() {
    if (m_window == nullptr) {
        m_window = static_cast<char*>(aligned_alloc(m_window_capacity, window_alignment));
    }
    return m_window != nullptr;
}

bool NativeStreamBuf::read_into(char* /* dst */, size_t /* size */, std::streamoff /* abs */) {
    // TODO(CVS-186707): io_read_into backend not yet implemented
    return false;
}

// Refill the get-area (window) with up to m_window_capacity bytes starting at m_cursor.
bool NativeStreamBuf::fill_window() {
    // 	CVS-189123
    if (m_cursor >= m_end || !allocate_window()) {
        return false;
    }
    const size_t to_read =
        static_cast<size_t>(std::min(static_cast<std::streamoff>(m_window_capacity), m_end - m_cursor));
    if (!read_into(m_window, to_read, m_cursor)) {
        return false;
    }
    setg(m_window, m_window, m_window + to_read);
    m_cursor += static_cast<std::streamoff>(to_read);
    return true;
}

// xsgetn: main bulk-read path. Large requests bypass the window; small requests are amortized through it.
std::streamsize NativeStreamBuf::xsgetn(char_type* dst, std::streamsize n) {
    // 	CVS-189123
    if (n <= 0) {
        return 0;
    }

    std::streamsize total = 0;
    while (n > 0) {
        // Drain whatever is already buffered in the get-area.
        if (gptr() != nullptr && gptr() < egptr()) {
            const std::streamsize avail = static_cast<std::streamsize>(egptr() - gptr());
            const std::streamsize from_buf = std::min(n, avail);
            std::memcpy(dst, gptr(), static_cast<size_t>(from_buf));
            gbump(static_cast<int>(from_buf));  // from_buf <= m_window_capacity <= INT_MAX
            dst += from_buf;
            n -= from_buf;
            total += from_buf;
            continue;
        }

        if (m_cursor >= m_end) {
            break;
        }

        const std::streamoff remaining = m_end - m_cursor;
        const std::streamsize want = static_cast<std::streamsize>(std::min(static_cast<std::streamoff>(n), remaining));

        if (static_cast<size_t>(want) >= m_bypass_size) {
            // Bypass the window: read straight into the caller's buffer.
            if (!read_into(dst, static_cast<size_t>(want), m_cursor)) {
                break;
            }
            m_cursor += want;
            dst += want;
            n -= want;
            total += want;
        } else if (!fill_window()) {
            // Refill the window and loop back to drain it; stop on read failure.
            break;
        }
    }

    return total;
}

// underflow: char-by-char / peek path (operator>>, std::getline).
NativeStreamBuf::int_type NativeStreamBuf::underflow() {
    // 	CVS-189123
    if (gptr() != nullptr && gptr() < egptr()) {
        return traits_type::to_int_type(*gptr());
    }
    if (!fill_window()) {
        return traits_type::eof();
    }
    return traits_type::to_int_type(*gptr());
}

NativeStreamBuf::pos_type NativeStreamBuf::seekoff(off_type off,
                                                   std::ios_base::seekdir way,
                                                   std::ios_base::openmode /* which */) {
    // 	CVS-189123
    // Internal offsets are absolute; public stream positions are logical (0 == m_start).
    std::streamoff new_pos = 0;
    if (way == std::ios_base::beg) {
        new_pos = m_start + off;
    } else if (way == std::ios_base::cur) {
        // Account for bytes still buffered in the get-area ahead of the logical cursor.
        const std::streamoff ahead = (gptr() != nullptr) ? static_cast<std::streamoff>(egptr() - gptr()) : 0;
        new_pos = m_cursor - ahead + off;
        if (off == 0) {
            // Pure tell: report position without disturbing the get-area.
            if (new_pos < m_start || new_pos > m_end) {
                return pos_type(off_type(-1));
            }
            return pos_type(new_pos - m_start);
        }
    } else {
        new_pos = m_end + off;
    }

    if (new_pos < m_start || new_pos > m_end) {
        return pos_type(off_type(-1));
    }

    setg(nullptr, nullptr, nullptr);  // drop the get-area; the next read refills from new_pos
    m_cursor = new_pos;
    return pos_type(new_pos - m_start);
}

NativeStreamBuf::pos_type NativeStreamBuf::seekpos(pos_type pos, std::ios_base::openmode /* which */) {
    // 	CVS-189123
    return seekoff(off_type(pos), std::ios_base::beg, std::ios_base::in);
}

std::streamsize NativeStreamBuf::showmanyc() {
    // 	CVS-189123
    std::streamsize buffered = 0;
    if (gptr() != nullptr && egptr() > gptr()) {
        buffered = static_cast<std::streamsize>(egptr() - gptr());
    }
    std::streamoff remaining_off = m_end - m_cursor;
    if (remaining_off < 0) {
        remaining_off = 0;
    }
    const std::streamsize total = buffered + static_cast<std::streamsize>(remaining_off);
    // Per [streambuf.virt.get]: -1 signals that underflow() would return eof().
    return total > 0 ? total : static_cast<std::streamsize>(-1);
}

}  // namespace ov::util
