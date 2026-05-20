// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/reservable_buffer.hpp"

#include <sys/mman.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ov::util {
ReservableBuffer::ReservableBuffer(size_t byte_size) : m_reserved_size{byte_size}, m_reserved_buffer{MAP_FAILED} {
    if (byte_size == 0) {
        std::runtime_error("Cannot reserve zero bytes.");
    }
}

void* ReservableBuffer::reserve() {
    if (m_reserved_buffer != MAP_FAILED) {
        m_last_error = "Region is already reserved.";
        return nullptr;
    }
    m_last_error.clear();
    m_reserved_buffer = mmap(nullptr, m_reserved_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m_reserved_buffer == MAP_FAILED) {
        m_last_error = std::string{"mmap failed, err: "} + std::strerror(errno);
        return nullptr;
    }
    return m_reserved_buffer;
}

ReservableBuffer::~ReservableBuffer() {
    if (m_reserved_buffer != MAP_FAILED) {
        std::ignore = munmap(m_reserved_buffer, m_reserved_size);
    }
}

bool ReservableBuffer::acquire() {
    m_last_error.clear();
    if (m_reserved_buffer != MAP_FAILED && m_reserved_size > 0) {
        if (mprotect(m_reserved_buffer, m_reserved_size, PROT_READ | PROT_WRITE) == -1) {
            m_last_error = std::string{"mprotect failed, err: "} + std::strerror(errno);
            return false;
        }
        return true;
    }
    m_last_error = "Region is not reserved.";
    return false;
}

void ReservableBuffer::evict() noexcept {
    m_last_error.clear();
    std::ignore = mprotect(m_reserved_buffer, m_reserved_size, PROT_NONE);
}

void ReservableBuffer::evict(size_t offset, size_t size) noexcept {
    m_last_error.clear();
    if (offset == 0 && size >= m_reserved_size) {
        evict();
    }
}

std::string_view ReservableBuffer::get_last_error() const noexcept {
    return m_last_error;
}
}  // namespace ov::util
