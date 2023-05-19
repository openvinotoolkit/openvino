// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared memory map objects
 * @file mmap_object.hpp
 */

#pragma once

#include <memory>

namespace ov {
namespace util {

class MmapBuffer {
public:
    MmapBuffer() : m_buffer(nullptr), m_size(0), m_offset(0) {}

    char* get_ptr() const {
        return m_buffer;
    }

    size_t get_size() const {
        return m_size;
    }

    size_t get_offset() const {
        return m_offset;
    }
    void set_offset(size_t offset) {
        m_offset = offset;
    }

    virtual ~MmapBuffer() {}

protected:
    char* m_buffer;
    size_t m_size;
    size_t m_offset;
};

template <typename T>
class SharedMmapBuffer : public MmapBuffer {
public:
    SharedMmapBuffer(char* data, const size_t size, const T& shared_object) : _shared_object(shared_object) {
        m_buffer = data;
        m_size = size;
        m_offset = 0;
    }

    virtual ~SharedMmapBuffer() {
        m_buffer = nullptr;
        m_size = 0;
        m_offset = 0;
    }

private:
    T _shared_object;
};

std::shared_ptr<MmapBuffer> load_mmap_object(const std::string& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

std::shared_ptr<MmapBuffer> load_mmap_object(const std::wstring& path);

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
}  // namespace util
}  // namespace ov
