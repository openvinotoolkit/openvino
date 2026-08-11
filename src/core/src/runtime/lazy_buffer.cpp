// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#include <algorithm>
#include <istream>
#include <mutex>
#include <utility>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/file_util.hpp"
#include "openvino/util/memory.hpp"
#include "openvino/util/mmap_object.hpp"
#include "openvino/util/parallel_read_streambuf.hpp"

#if defined(__linux__)
#    include <fcntl.h>
#    include <linux/userfaultfd.h>
#    include <poll.h>
#    include <sys/ioctl.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <unistd.h>

#    include <cerrno>
#    include <thread>
#endif

namespace ov {
namespace {

#if defined(__linux__)

/// Delegates page faults on registered LazyBuffer regions to a single background thread via userfaultfd.
class UffdManager {
public:
    static UffdManager& get() {
        // Intentionally leaked: LazyBuffer instances may still be destroyed during static destruction.
        static auto* const instance = new UffdManager();
        return *instance;
    }

    bool is_available() const noexcept {
        return m_uffd != -1;
    }

    bool register_region(LazyBuffer* buffer, void* addr, size_t size) {
        if (!is_available()) {
            return false;
        }

        uffdio_register reg{};
        reg.range.start = reinterpret_cast<__u64>(addr);
        reg.range.len = size;
        reg.mode = UFFDIO_REGISTER_MODE_MISSING;
        if (ioctl(m_uffd, UFFDIO_REGISTER, &reg) == -1) {
            return false;
        }

        std::lock_guard lock{m_mutex};
        m_regions.push_back({buffer, reinterpret_cast<std::uintptr_t>(addr), size});
        return true;
    }

    void unregister_region(LazyBuffer* buffer, void* addr, size_t size) noexcept {
        if (!is_available()) {
            return;
        }

        {
            std::lock_guard lock{m_mutex};
            m_regions.erase(std::remove_if(m_regions.begin(),
                                           m_regions.end(),
                                           [buffer](const Region& region) {
                                               return region.buffer == buffer;
                                           }),
                            m_regions.end());
        }

        uffdio_range range{};
        range.start = reinterpret_cast<__u64>(addr);
        range.len = size;
        std::ignore = ioctl(m_uffd, UFFDIO_UNREGISTER, &range);
    }

    void relocate(LazyBuffer* from, LazyBuffer* to) noexcept {
        if (!is_available()) {
            return;
        }

        std::lock_guard lock{m_mutex};
        for (auto& region : m_regions) {
            if (region.buffer == from) {
                region.buffer = to;
                break;
            }
        }
    }

    /// Populates the whole region at once, waking every thread blocked on any of its pages.
    bool resolve_fault(void* addr, size_t size, const void* src) const noexcept {
        uffdio_copy copy{};
        copy.dst = reinterpret_cast<__u64>(addr);
        copy.src = reinterpret_cast<__u64>(src);
        copy.len = size;
        if (ioctl(m_uffd, UFFDIO_COPY, &copy) != -1) {
            return true;
        }
        // EEXIST means the pages were already installed by a concurrent resolution, but only if
        // the whole range was actually copied; otherwise the copy aborted partway through.
        return errno == EEXIST && copy.copy == static_cast<__s64>(size);
    }

private:
    struct Region {
        LazyBuffer* buffer;
        std::uintptr_t start;
        size_t size;
    };

    UffdManager() {
        m_uffd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK | UFFD_USER_MODE_ONLY));
        if (m_uffd == -1) {
            // UFFD_USER_MODE_ONLY requires Linux 5.11.
            m_uffd = static_cast<int>(syscall(SYS_userfaultfd, O_CLOEXEC | O_NONBLOCK));
        }
        if (m_uffd == -1) {
            return;
        }

        uffdio_api api{};
        api.api = UFFD_API;
        if (ioctl(m_uffd, UFFDIO_API, &api) == -1) {
            close(m_uffd);
            m_uffd = -1;
            return;
        }

        std::thread{[this] {
            handle_faults();
        }}.detach();
    }

    LazyBuffer* find_owner(std::uintptr_t fault_addr) {
        std::lock_guard lock{m_mutex};
        for (const auto& region : m_regions) {
            if (fault_addr >= region.start && fault_addr < region.start + region.size) {
                return region.buffer;
            }
        }
        return nullptr;
    }

    void handle_faults() {
        pollfd poll_fd{m_uffd, POLLIN, 0};
        for (;;) {
            if (poll(&poll_fd, 1, -1) == -1) {
                if (errno == EINTR) {
                    continue;
                }
                return;
            }

            uffd_msg msg{};
            const auto bytes_read = read(m_uffd, &msg, sizeof(msg));
            if (bytes_read == -1) {
                if (errno == EAGAIN || errno == EINTR) {
                    continue;
                }
                return;
            }
            if (bytes_read != sizeof(msg) || msg.event != UFFD_EVENT_PAGEFAULT) {
                continue;
            }

            const auto fault_addr = static_cast<std::uintptr_t>(msg.arg.pagefault.address);
            if (auto* const buffer = find_owner(fault_addr)) {
                try {
                    buffer->hint_prefetch();
                } catch (...) {
                    // An unresolved fault would block the faulting thread forever, so drop the registration
                    // and let it fail with SIGBUS instead.
                    unregister_region(buffer, reinterpret_cast<void*>(fault_addr), 1);
                }
            }
        }
    }

    int m_uffd{-1};
    std::mutex m_mutex;
    std::vector<Region> m_regions;
};

void* reserve_faulting(size_t size) noexcept {
    auto* const addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return addr == MAP_FAILED ? nullptr : addr;
}

void release_faulting(void* addr, size_t size) noexcept {
    std::ignore = munmap(addr, size);
}

bool page_faulting_available() noexcept {
    return UffdManager::get().is_available();
}

#else

void* reserve_faulting(size_t size) noexcept {
    std::error_code ec;
    auto* addr = util::vm_reserve(size, ec);
    if (addr != nullptr) {
        util::vm_commit(addr, size, ec);
        if (ec) {
            util::vm_release(addr, size);
            addr = nullptr;
        }
    }
    return addr;
}

void release_faulting(void* addr, size_t size) noexcept {
    util::vm_release(addr, size);
}

bool page_faulting_available() noexcept {
    return false;
}

#endif
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

    // userfaultfd operates on whole pages, so the reservation is rounded up while size() stays as requested.
    const auto page_size = static_cast<size_t>(util::get_system_page_size());
    m_mapped_size = util::align_size_up(std::max<size_t>(1, m_byte_size), page_size);

    m_aligned_buffer = static_cast<char*>(reserve_faulting(m_mapped_size));
    OPENVINO_ASSERT(m_aligned_buffer != nullptr, "Failed to reserve memory for LazyBuffer: ", m_file_path);

    const bool delegated =
#if defined(__linux__)
        UffdManager::get().register_region(this, m_aligned_buffer, m_mapped_size);
#else
        false;
#endif
    if (!delegated) {
        // Without fault delegation the mapping reads as zeros, so the data has to be loaded up front.
        try {
            hint_prefetch();
        } catch (...) {
            release_faulting(m_aligned_buffer, m_mapped_size);
            // The base destructor must not free memory which it did not allocate.
            m_aligned_buffer = nullptr;
            throw;
        }
    }
}

LazyBuffer::~LazyBuffer() {
    if (m_aligned_buffer) {
#if defined(__linux__)
        UffdManager::get().unregister_region(this, m_aligned_buffer, m_mapped_size);
#endif
        release_faulting(m_aligned_buffer, m_mapped_size);
        m_aligned_buffer = nullptr;
    }
    m_byte_size = 0;
    m_mapped_size = 0;
}

LazyBuffer::LazyBuffer(LazyBuffer&& other) noexcept
    : AlignedBuffer(std::move(other)),
      m_file_path{std::move(other.m_file_path)},
      m_offset{std::exchange(other.m_offset, 0)},
      m_mapped_size{std::exchange(other.m_mapped_size, 0)},
      m_loaded{other.m_loaded.exchange(false, std::memory_order_relaxed)} {
#if defined(__linux__)
    UffdManager::get().relocate(&other, this);
#endif
}

LazyBuffer& LazyBuffer::operator=(LazyBuffer&& other) noexcept {
    if (this != &other) {
        if (m_aligned_buffer) {
#if defined(__linux__)
            UffdManager::get().unregister_region(this, m_aligned_buffer, m_mapped_size);
#endif
            release_faulting(m_aligned_buffer, m_mapped_size);
            m_aligned_buffer = nullptr;
        }
        AlignedBuffer::operator=(std::move(other));
        m_file_path = std::move(other.m_file_path);
        m_offset = std::exchange(other.m_offset, 0);
        m_mapped_size = std::exchange(other.m_mapped_size, 0);
        m_loaded = other.m_loaded.exchange(false, std::memory_order_relaxed);
#if defined(__linux__)
        UffdManager::get().relocate(&other, this);
#endif
    }
    return *this;
}

void LazyBuffer::hint_prefetch() const {
    if (m_loaded.load(std::memory_order_acquire)) {
        return;
    }

    std::lock_guard lock{m_loading};
    if (m_loaded.load(std::memory_order_relaxed)) {
        return;
    }

#if defined(__linux__)
    if (page_faulting_available()) {
        // The page-rounded tail has to be zeroed because UFFDIO_COPY installs the whole reservation at once.
        std::vector<char> staging(m_mapped_size, 0);
        read_file_data(staging.data());

        OPENVINO_ASSERT(UffdManager::get().resolve_fault(m_aligned_buffer, m_mapped_size, staging.data()),
                        "Failed to resolve page fault for LazyBuffer: ",
                        m_file_path);
        m_loaded.store(true, std::memory_order_release);
        return;
    }
#endif

    read_file_data(m_aligned_buffer);
    m_loaded.store(true, std::memory_order_release);
}

void LazyBuffer::read_file_data(char* destination) const {
    util::ParallelReadStreamBuf par_buf(m_file_path, static_cast<std::streamoff>(m_offset));
    std::istream file(&par_buf);
    OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
    file.read(destination, m_byte_size);
    OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
}

void LazyBuffer::hint_evict() noexcept {
    hint_evict(0, m_byte_size);
}

void LazyBuffer::hint_evict(size_t offset, size_t size) noexcept {
    if (!page_faulting_available()) {
        // Without fault delegation the pages cannot be repopulated on access, so dropping them would lose data.
        return;
    }

    if (m_loaded.load(std::memory_order_acquire)) {
        try {
            std::lock_guard lock{m_loading};
            if (m_loaded.load(std::memory_order_relaxed)) {
#if defined(__linux__)
                // Zapping the pages makes the region missing again, so the next access re-faults and reloads.
                std::ignore = madvise(m_aligned_buffer, m_mapped_size, MADV_DONTNEED);
#endif
                m_loaded.store(false, std::memory_order_release);
            }
        } catch (...) {
        }
    }
}
}  // namespace ov
