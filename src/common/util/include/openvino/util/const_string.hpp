// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <stdexcept>

namespace ov {
namespace util {
class ConstString {
public:
    template <size_t SIZE>
    constexpr ConstString(const char (&p)[SIZE]) : m_string(p),
                                                   m_size(SIZE) {}

    constexpr char operator[](size_t i) const {
        return i < m_size ? m_string[i] : throw std::out_of_range("");
    }
    constexpr const char* get_ptr(size_t offset) const {
        return offset < m_size ? &m_string[offset] : m_string;
    }
    constexpr size_t size() const {
        return m_size;
    }

private:
    const char* m_string;
    size_t m_size;
};

constexpr const char* find_last(ConstString s, size_t offset, char ch) {
    return offset == 0 ? s.get_ptr(0) : (s[offset] == ch ? s.get_ptr(offset + 1) : find_last(s, offset - 1, ch));
}

constexpr const char* find_last(ConstString s, char ch) {
    return find_last(s, s.size() - 1, ch);
}

constexpr const char* get_file_name(ConstString s) {
    return find_last(s, '/');
}

}  // namespace util
}  // namespace ov
