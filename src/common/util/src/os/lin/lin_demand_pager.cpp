// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#if defined(__linux__)
#    include <fcntl.h>
#    include <linux/userfaultfd.h>
#    include <poll.h>
#    include <sys/eventfd.h>
#    include <sys/ioctl.h>
#    include <sys/mman.h>
#    include <sys/syscall.h>
#    include <unistd.h>

#    include <algorithm>
#    include <cerrno>
#    include <cstdint>
#    include <mutex>
#    include <thread>
#    include <vector>
#endif

#include <system_error>
#include <tuple>

#include "openvino/util/demand_pager.hpp"
#include "openvino/util/memory.hpp"

namespace ov::util {

#if defined(__linux__)

struct DemandPager::Impl {
    struct Region {
        callback_type callback{nullptr};
        void* user_data{nullptr};
        std::uintptr_t start{0};
        size_type size{0};
    };

    Impl() {
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
        m_stop_fd = eventfd(0, EFD_CLOEXEC);
        if (ioctl(m_uffd, UFFDIO_API, &api) == -1 || m_stop_fd == -1) {
            close_fds();
            return;
        }

        try {
            m_thread = std::thread{[this] {
                handle_faults();
            }};
        } catch (...) {
            close_fds();
        }
    }

    ~Impl() {
        if (m_thread.joinable()) {
            constexpr std::uint64_t stop_token = 1;
            std::ignore = write(m_stop_fd, &stop_token, sizeof(stop_token));
            m_thread.join();
        }
        close_fds();
    }

    Impl(const Impl&) = delete;
    Impl& operator=(const Impl&) = delete;

    bool is_available() const noexcept {
        return m_uffd != -1;
    }

    /// Looks up the region containing addr. The caller must hold m_mutex.
    std::vector<Region>::iterator find_it(std::uintptr_t addr) {
        return std::find_if(m_regions.begin(), m_regions.end(), [addr](const Region& region) {
            return addr >= region.start && addr < region.start + region.size;
        });
    }

    void remove(std::uintptr_t addr) noexcept {
        uffdio_range range{};
        {
            std::lock_guard lock{m_mutex};
            const auto it = find_it(addr);
            if (it == m_regions.end()) {
                return;
            }
            range.start = static_cast<__u64>(it->start);
            range.len = it->size;
            m_regions.erase(it);
        }
        std::ignore = ioctl(m_uffd, UFFDIO_UNREGISTER, &range);
    }

    void handle_faults() {
        pollfd poll_fds[2]{{m_uffd, POLLIN, 0}, {m_stop_fd, POLLIN, 0}};
        for (;;) {
            if (poll(poll_fds, 2, -1) == -1) {
                if (errno == EINTR) {
                    continue;
                }
                return;
            }
            if ((poll_fds[1].revents & POLLIN) != 0) {
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

            Region region{};
            {
                std::lock_guard lock{m_mutex};
                const auto it = find_it(static_cast<std::uintptr_t>(msg.arg.pagefault.address));
                if (it == m_regions.end()) {
                    continue;
                }
                region = *it;
            }

            region.callback(region.user_data, reinterpret_cast<pointer_type>(region.start), region.size);
        }
    }

    void close_fds() noexcept {
        if (m_stop_fd != -1) {
            close(m_stop_fd);
            m_stop_fd = -1;
        }
        if (m_uffd != -1) {
            close(m_uffd);
            m_uffd = -1;
        }
    }

    int m_uffd{-1};
    int m_stop_fd{-1};
    std::thread m_thread;
    std::mutex m_mutex;
    std::vector<Region> m_regions;
};

DemandPager::DemandPager() : m_impl{std::make_unique<Impl>()} {}

DemandPager::~DemandPager() = default;

bool DemandPager::is_available() const noexcept {
    return m_impl->is_available();
}

DemandPager::pointer_type DemandPager::reserve(size_type size) noexcept {
    auto* const addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    return addr == MAP_FAILED ? nullptr : addr;
}

void DemandPager::release(pointer_type addr, size_type size) noexcept {
    std::ignore = munmap(addr, size);
}

bool DemandPager::register_region(callback_type user_callback,
                                  void* user_data,
                                  pointer_type addr,
                                  size_type size) noexcept {
    if (!m_impl->is_available() || user_callback == nullptr) {
        return false;
    }

    uffdio_register reg{};
    reg.range.start = reinterpret_cast<__u64>(addr);
    reg.range.len = size;
    reg.mode = UFFDIO_REGISTER_MODE_MISSING;
    if (ioctl(m_impl->m_uffd, UFFDIO_REGISTER, &reg) == -1) {
        return false;
    }

    std::lock_guard lock{m_impl->m_mutex};
    m_impl->m_regions.push_back(Impl::Region{user_callback, user_data, reinterpret_cast<std::uintptr_t>(addr), size});
    return true;
}

void DemandPager::unregister_region(pointer_type addr) noexcept {
    if (m_impl->is_available()) {
        m_impl->remove(reinterpret_cast<std::uintptr_t>(addr));
    }
}

void DemandPager::update_user_data(pointer_type addr, void* user_data) noexcept {
    if (!m_impl->is_available()) {
        return;
    }

    std::lock_guard lock{m_impl->m_mutex};
    const auto it = m_impl->find_it(reinterpret_cast<std::uintptr_t>(addr));
    if (it != m_impl->m_regions.end()) {
        it->user_data = user_data;
    }
}

bool DemandPager::populate(pointer_type addr, size_type size, const void* src) noexcept {
    if (!m_impl->is_available()) {
        return false;
    }

    uffdio_copy copy{};
    copy.dst = reinterpret_cast<__u64>(addr);
    copy.src = reinterpret_cast<__u64>(src);
    copy.len = size;
    if (ioctl(m_impl->m_uffd, UFFDIO_COPY, &copy) != -1) {
        return true;
    }
    // EEXIST means the pages were already installed by a concurrent call, but only if
    // the whole range was actually copied; otherwise the copy aborted partway through.
    return errno == EEXIST && copy.copy == static_cast<__s64>(size);
}

void DemandPager::evict(pointer_type addr, size_type size) noexcept {
    if (m_impl->is_available()) {
        // Zapping the pages makes the region missing again, so the next access re-faults.
        std::ignore = madvise(addr, size, MADV_DONTNEED);
    }
}

#else

// userfaultfd is Linux only, so on the remaining Unix platforms every region has to be populated up front.
struct DemandPager::Impl {};

DemandPager::DemandPager() = default;

DemandPager::~DemandPager() = default;

bool DemandPager::is_available() const noexcept {
    return false;
}

DemandPager::pointer_type DemandPager::reserve(size_type size) noexcept {
    std::error_code ec;
    auto* addr = vm_reserve(size, ec);
    if (addr != nullptr) {
        vm_commit(addr, size, ec);
        if (ec) {
            vm_release(addr, size);
            addr = nullptr;
        }
    }
    return addr;
}

void DemandPager::release(pointer_type addr, size_type size) noexcept {
    vm_release(addr, size);
}

bool DemandPager::register_region(callback_type, void*, pointer_type, size_type) noexcept {
    return false;
}

void DemandPager::unregister_region(pointer_type) noexcept {}

void DemandPager::update_user_data(pointer_type, void*) noexcept {}

bool DemandPager::populate(pointer_type, size_type, const void*) noexcept {
    return false;
}

void DemandPager::evict(pointer_type, size_type) noexcept {}

#endif
}  // namespace ov::util
