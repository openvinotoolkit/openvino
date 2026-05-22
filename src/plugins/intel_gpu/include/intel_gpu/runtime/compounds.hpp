// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

#include <vector>
#include <cassert>
#include <iterator>
#include <cstring>
#include <string>
#include <stdexcept>


namespace cldnn {

/// @addtogroup cpp_api C++ API
/// @{

/// @cond CPP_HELPERS

/// @defgroup cpp_helpers Helpers
/// @{

template <typename T>
class mutable_array_ref {
public:
    typedef size_t size_type;

    mutable_array_ref() : _data(nullptr), _size(0) {}
    explicit mutable_array_ref(T& val) : _data(&val), _size(1) {}
    mutable_array_ref(T* data, size_t size) : _data(data), _size(size) {}

    template <size_t N>
    explicit mutable_array_ref(T (&arr)[N]) : _data(arr), _size(N) {}

    mutable_array_ref(const mutable_array_ref& other) : _data(other._data), _size(other._size) {}

    mutable_array_ref& operator=(const mutable_array_ref& other) {
        if (this == &other)
            return *this;
        _data = other._data;
        _size = other._size;
        return *this;
    }

    T* data() const { return _data; }
    size_t size() const { return _size; }
    bool empty() const { return _size == 0; }

    typedef T* iterator;
    typedef T* const_iterator;
    iterator begin() const { return _data; }
    iterator end() const { return _data + _size; }
    const_iterator cbegin() const { return _data; }
    const_iterator cend() const { return _data + _size; }

    T& operator[](size_t idx) const {
        assert(idx < _size);
        return _data[idx];
    }

    T& at(size_t idx) const {
        if (idx >= _size) throw std::out_of_range("idx");
        return _data[idx];
    }

    std::vector<T> vector() const { return std::vector<T>(_data, _data + _size); }

private:
    T* _data;
    size_t _size;
};

/// @}

/// @endcond

/// @}
}  // namespace cldnn
