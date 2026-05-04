// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/lazy_buffer.hpp"

#ifdef _WIN32
#    include <windows.h>
#else
#    include <sys/mman.h>
#    include <unistd.h>
#endif

#include <fstream>

#include "openvino/core/except.hpp"
#include "openvino/core/memory_util.hpp"
#include "openvino/util/file_util.hpp"

namespace ov {
LazyBuffer::LazyBuffer(std::filesystem::path file_path, size_t offset, size_t byte_size, size_t alignment)
    : AlignedBuffer(),
      m_file_path{std::move(file_path)},
      m_offset{offset},
      m_alignment{alignment} {
    m_byte_size = byte_size;
    const auto file_size = util::file_size(m_file_path);
    OPENVINO_ASSERT(file_size >= 0 && static_cast<size_t>(file_size) >= m_offset + m_byte_size,
                    "If file exists it's size is smaller than requested range (file size: ",
                    file_size,
                    ", requested offset: ",
                    m_offset,
                    ", requested byte size: ",
                    m_byte_size,
                    ").");

    m_reserved_size = m_byte_size + m_alignment - 1;
    OPENVINO_ASSERT(m_reserved_size >= m_byte_size,
                    "Integer overflow occurred while calculating reserved size for LazyBuffer (requested byte size: ",
                    m_byte_size,
                    ", alignment: ",
                    m_alignment,
                    ").");
    m_reserved_buffer = mmap(nullptr, m_reserved_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    OPENVINO_ASSERT(m_reserved_buffer != MAP_FAILED, "mmap failed, err: ", std::strerror(errno));

    m_aligned_buffer = static_cast<char*>(m_reserved_buffer) +
                       util::align_padding_size(m_alignment, reinterpret_cast<size_t>(m_reserved_buffer));
}

LazyBuffer::~LazyBuffer() {
    if (m_reserved_buffer) {
        munmap(m_reserved_buffer, m_reserved_size);
    }
}

void LazyBuffer::load() const {
    if (!m_loaded && m_byte_size > 0) {
#ifdef _WIN32
        void* result = VirtualAlloc(static_cast<char*>(base_addr), to_commit, MEM_COMMIT, PAGE_READWRITE);
        if (!result)
            throw std::runtime_error("VirtualAlloc commit failed");
#else
        if (mprotect(m_reserved_buffer, m_reserved_size, PROT_READ | PROT_WRITE) == -1) {
            OPENVINO_THROW("mprotect failed, err: ", std::strerror(errno));
        }
#endif

        try {
            std::ifstream file(m_file_path, std::ios::binary);
            OPENVINO_ASSERT(file, "Failed to open file: ", m_file_path);
            file.seekg(m_offset).read(m_aligned_buffer, m_byte_size);
            OPENVINO_ASSERT(file, "Failed to read data from file: ", m_file_path);
        } catch (...) {
            std::ignore = mprotect(m_reserved_buffer, m_reserved_size, PROT_NONE);
            throw;
        }
        m_loaded = true;
    }
}

void LazyBuffer::evict() {
    if (m_loaded) {
        m_loaded = false;
        mprotect(m_reserved_buffer, m_reserved_size, PROT_NONE);
    }
}
}  // namespace ov
