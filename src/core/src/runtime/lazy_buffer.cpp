// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <csignal>
#include <istream>
#include <mutex>
#include <utility>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/parallel_read_streambuf.hpp"

namespace ov {
namespace {
std::once_flag g_sigsegv_handler_flag;
struct sigaction g_old_sigsegv_action;

std::vector<std::tuple<LazyBuffer*, std::uintptr_t, std::size_t>> g_reserved_regions;

void sigsegv_handler(int signal, siginfo_t* info, void* context) {
    for (const auto& [buf, reserved_ptr, reserved_size] : g_reserved_regions) {
        const auto fault_addr_int = reinterpret_cast<std::uintptr_t>(info->si_addr);

        if (fault_addr_int >= reserved_ptr && fault_addr_int < reserved_ptr + reserved_size) {
            buf->hint_prefetch();
            return;
        }
    }

    // Chain to the previously registered handler.
    if (g_old_sigsegv_action.sa_flags & SA_SIGINFO) {
        g_old_sigsegv_action.sa_sigaction(signal, info, context);
    } else if (g_old_sigsegv_action.sa_handler == SIG_DFL) {
        // Restore the default disposition (terminate + core dump) and re-raise so
        // the OS records the fault address and produces a core file normally.
        std::signal(SIGSEGV, SIG_DFL);
        std::raise(SIGSEGV);
    } else if (g_old_sigsegv_action.sa_handler == SIG_IGN) {
        // SIGSEGV cannot meaningfully be ignored: returning from the handler would
        // resume the faulting instruction unchanged, triggering the same fault
        // again and looping forever.  Force the default crash behaviour instead.
        std::signal(SIGSEGV, SIG_DFL);
        std::raise(SIGSEGV);
    } else {
        g_old_sigsegv_action.sa_handler(signal);
    }
}
}  // namespace

LazyBuffer::LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_offset{offset},
      m_loaded{false} {
    m_byte_size = byte_size;

    const auto file_size = util::file_size(m_file_path);
    OPENVINO_ASSERT(file_size >= 0, "Failed to get file size for ", m_file_path);

    const bool offset_fits = m_offset <= static_cast<size_t>(file_size);
    const bool size_fits = m_byte_size <= static_cast<size_t>(file_size) - m_offset;
    OPENVINO_ASSERT(offset_fits && size_fits,
                    "Requested region ",
                    m_offset,
                    "+",
                    m_byte_size,
                    " exceeds file size ",
                    file_size,
                    " for file: ",
                    m_file_path);

    std::error_code ec;
    m_aligned_buffer = static_cast<char*>(util::vm_reserve(m_byte_size, ec));
    OPENVINO_ASSERT(m_aligned_buffer != nullptr, "Failed to reserve memory for LazyBuffer. Error: ", ec.message());

    g_reserved_regions.emplace_back(this, reinterpret_cast<std::uintptr_t>(m_aligned_buffer), m_byte_size);

    std::call_once(g_sigsegv_handler_flag, []() {
        struct sigaction sa;
        std::memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = sigsegv_handler;
        sa.sa_flags = SA_SIGINFO;
        sigaction(SIGSEGV, &sa, &g_old_sigsegv_action);
    });
}

LazyBuffer::~LazyBuffer() {
    g_reserved_regions.erase(std::remove_if(g_reserved_regions.begin(),
                                            g_reserved_regions.end(),
                                            [this](const auto& entry) {
                                                return std::get<0>(entry) == this;
                                            }),
                             g_reserved_regions.end());
    if (m_aligned_buffer) {
        util::vm_release(m_aligned_buffer, m_byte_size);
        m_aligned_buffer = nullptr;
    }
    m_byte_size = 0;
}

LazyBuffer::LazyBuffer(LazyBuffer&& other) noexcept
    : AlignedBuffer(std::move(other)),
      m_file_path{std::move(other.m_file_path)},
      m_offset{std::exchange(other.m_offset, 0)},
      m_loaded{other.m_loaded.exchange(false, std::memory_order_relaxed)} {
    for (auto& [buf, ptr, size] : g_reserved_regions) {
        if (buf == &other) {
            buf = this;
            break;
        }
    }
}

LazyBuffer& LazyBuffer::operator=(LazyBuffer&& other) noexcept {
    if (this != &other) {
        g_reserved_regions.erase(std::remove_if(g_reserved_regions.begin(),
                                                g_reserved_regions.end(),
                                                [this](const auto& entry) {
                                                    return std::get<0>(entry) == this;
                                                }),
                                 g_reserved_regions.end());
        if (m_aligned_buffer) {
            util::vm_release(m_aligned_buffer, m_byte_size);
            m_aligned_buffer = nullptr;
        }
        AlignedBuffer::operator=(std::move(other));
        m_file_path = std::move(other.m_file_path);
        m_offset = std::exchange(other.m_offset, 0);
        m_loaded = other.m_loaded.exchange(false, std::memory_order_relaxed);
        for (auto& [buf, ptr, size] : g_reserved_regions) {
            if (buf == &other) {
                buf = this;
                break;
            }
        }
    }
    return *this;
}

void LazyBuffer::hint_prefetch() const {
    if (!m_loaded.load(std::memory_order_acquire)) {
        std::lock_guard lock{m_loading};
        if (m_loaded.load(std::memory_order_relaxed)) {
            return;
        }

        std::error_code ec;
        util::vm_commit(m_aligned_buffer, m_byte_size, ec);
        OPENVINO_ASSERT(!ec, "Failed to commit memory for LazyBuffer. Error: ", ec.message());

        try {
            util::ParallelReadStreamBuf par_buf(m_file_path, static_cast<std::streamoff>(m_offset));
            std::istream file(&par_buf);
            OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
            file.read(m_aligned_buffer, m_byte_size);
            OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
            m_loaded.store(true, std::memory_order_release);
        } catch (...) {
            util::vm_decommit(m_aligned_buffer, m_byte_size);
            throw;
        }
    }
}

void LazyBuffer::hint_evict() noexcept {
    hint_evict(0, m_byte_size);
}

void LazyBuffer::hint_evict(size_t offset, size_t size) noexcept {
    if (m_loaded.load(std::memory_order_acquire)) {
        try {
            std::lock_guard lock{m_loading};
            if (m_loaded.load(std::memory_order_relaxed)) {
                util::vm_decommit(m_aligned_buffer, m_byte_size);
                m_loaded.store(false, std::memory_order_release);
            }
        } catch (...) {
        }
    }
}
}  // namespace ov
