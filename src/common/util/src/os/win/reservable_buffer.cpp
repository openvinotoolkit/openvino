// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/util/reservable_buffer.hpp"

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#include <windows.h>

#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ov::util {
ReservableBuffer::ReservableBuffer(size_t byte_size)
    : m_reserved_size{byte_size},
      m_reserved_buffer{nullptr},
      m_last_error{} {
    if (byte_size == 0) {
        throw std::runtime_error("Zero size buffer makes no sense");
    }
    m_reserved_buffer = VirtualAlloc(nullptr, m_reserved_size, MEM_RESERVE, PAGE_NOACCESS);
    if (m_reserved_buffer == nullptr) {
        throw std::runtime_error(std::string{"VirtualAlloc failed, err: "} + std::to_string(GetLastError()));
    }
}

ReservableBuffer::~ReservableBuffer() {
    std::ignore = VirtualFree(m_reserved_buffer, 0, MEM_RELEASE);
}

bool ReservableBuffer::acquire() {
    m_last_error.clear();
    if (VirtualAlloc(m_reserved_buffer, m_reserved_size, MEM_COMMIT, PAGE_READWRITE) == nullptr) {
        m_last_error = std::string{"VirtualAlloc commit failed, err: "} + std::to_string(GetLastError());
        return false;
    }
    return true;
}

void ReservableBuffer::evict() noexcept {
    m_last_error.clear();
    std::ignore = VirtualFree(m_reserved_buffer, m_reserved_size, MEM_DECOMMIT);
}

void ReservableBuffer::evict(size_t offset, size_t size) noexcept {
    m_last_error.clear();
    if (offset == 0 && size >= m_reserved_size) {
        evict();
    }
}

std::string_view ReservableBuffer::last_error() const noexcept {
    return m_last_error;
}
}  // namespace ov::util
