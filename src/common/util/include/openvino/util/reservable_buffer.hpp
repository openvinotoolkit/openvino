// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace ov::util {
class ReservableBuffer {
public:
    ReservableBuffer(size_t byte_size);
    ~ReservableBuffer();

    ReservableBuffer(const ReservableBuffer&) = delete;
    ReservableBuffer& operator=(const ReservableBuffer&) = delete;

    void* reserve();
    bool acquire();
    void evict() noexcept;
    void evict(size_t offset, size_t size) noexcept;

    std::string_view get_last_error() const noexcept;

private:
    const size_t m_reserved_size{0};
    void* m_reserved_buffer{nullptr};

    std::string m_last_error;
};
}  // namespace ov::util
