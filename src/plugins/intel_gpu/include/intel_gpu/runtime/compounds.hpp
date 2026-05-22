// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

#include <vector>
#include <cassert>
#include <cstddef>
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

#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL != 0
template <typename T>
class checked_array_iterator {
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = std::remove_cv_t<T>;
    using difference_type   = std::ptrdiff_t;
    using pointer           = T*;
    using reference         = T&;

    checked_array_iterator() = default;
    checked_array_iterator(T* data, std::size_t size, std::size_t index = 0) noexcept
        : _Myptr(data), _Mysize(size), _Myoff(index) {
        _STL_VERIFY(index <= size, "cldnn::checked_array_iterator out of range");
    }

    // Implicit conversion iterator -> const_iterator, matching MSVC vector.
    template <typename U, typename = std::enable_if_t<std::is_convertible<U*, T*>::value>>
    checked_array_iterator(const checked_array_iterator<U>& other) noexcept
        : _Myptr(other._Unwrapped_base()), _Mysize(other._Unwrapped_size()), _Myoff(other._Unwrapped_off()) {}

    reference operator*() const noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot dereference value-initialized iterator");
        _STL_VERIFY(_Myoff < _Mysize, "cannot dereference end array iterator");
        return _Myptr[_Myoff];
    }
    pointer operator->() const noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot dereference value-initialized iterator");
        _STL_VERIFY(_Myoff < _Mysize, "cannot dereference end array iterator");
        return _Myptr + _Myoff;
    }
    reference operator[](difference_type n) const noexcept {
        return *(*this + n);
    }

    checked_array_iterator& operator++() noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot increment value-initialized iterator");
        _STL_VERIFY(_Myoff < _Mysize, "cannot increment iterator past end");
        ++_Myoff;
        return *this;
    }
    checked_array_iterator operator++(int) noexcept { auto tmp = *this; ++(*this); return tmp; }

    checked_array_iterator& operator--() noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot decrement value-initialized iterator");
        _STL_VERIFY(_Myoff != 0, "cannot decrement iterator before begin");
        --_Myoff;
        return *this;
    }
    checked_array_iterator operator--(int) noexcept { auto tmp = *this; --(*this); return tmp; }

    checked_array_iterator& operator+=(difference_type n) noexcept {
        _Verify_offset(n);
        _Myoff += static_cast<std::size_t>(n);
        return *this;
    }
    checked_array_iterator& operator-=(difference_type n) noexcept { return *this += -n; }

    friend checked_array_iterator operator+(checked_array_iterator it, difference_type n) noexcept { return it += n; }
    friend checked_array_iterator operator+(difference_type n, checked_array_iterator it) noexcept { return it += n; }
    friend checked_array_iterator operator-(checked_array_iterator it, difference_type n) noexcept { return it -= n; }
    friend difference_type operator-(const checked_array_iterator& a, const checked_array_iterator& b) noexcept {
        a._Compat(b);
        return static_cast<difference_type>(a._Myoff) - static_cast<difference_type>(b._Myoff);
    }

    friend bool operator==(const checked_array_iterator& a, const checked_array_iterator& b) noexcept {
        a._Compat(b);
        return a._Myoff == b._Myoff;
    }
    friend bool operator!=(const checked_array_iterator& a, const checked_array_iterator& b) noexcept {
        a._Compat(b);
        return a._Myoff != b._Myoff;
    }
    friend bool operator<(const checked_array_iterator& a, const checked_array_iterator& b) noexcept {
        a._Compat(b);
        return a._Myoff < b._Myoff;
    }
    friend bool operator>(const checked_array_iterator& a, const checked_array_iterator& b) noexcept {
        a._Compat(b);
        return a._Myoff > b._Myoff;
    }
    friend bool operator<=(const checked_array_iterator& a, const checked_array_iterator& b) noexcept {
        a._Compat(b);
        return a._Myoff <= b._Myoff;
    }
    friend bool operator>=(const checked_array_iterator& a, const checked_array_iterator& b) noexcept {
        a._Compat(b);
        return a._Myoff >= b._Myoff;
    }

    // MSVC-style unwrapping helpers used by the iterator->const_iterator ctor above.
    T*          _Unwrapped_base() const noexcept { return _Myptr; }
    std::size_t _Unwrapped_size() const noexcept { return _Mysize; }
    std::size_t _Unwrapped_off()  const noexcept { return _Myoff; }

private:
    void _Verify_offset(difference_type n) const noexcept {
        if (n < 0) {
            _STL_VERIFY(static_cast<std::size_t>(-n) <= _Myoff, "cannot seek array iterator before begin");
        } else if (n > 0) {
            _STL_VERIFY(static_cast<std::size_t>(n) <= _Mysize - _Myoff, "cannot seek array iterator after end");
        }
    }
    void _Compat(const checked_array_iterator& other) const noexcept {
        _STL_VERIFY(_Myptr == other._Myptr && _Mysize == other._Mysize,
                    "array iterators from different ranges are incompatible");
    }

    T*          _Myptr  = nullptr;
    std::size_t _Mysize = 0;
    std::size_t _Myoff  = 0;
};
#endif  // _ITERATOR_DEBUG_LEVEL

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

#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL != 0
    typedef checked_array_iterator<T>       iterator;
    typedef checked_array_iterator<const T> const_iterator;
    iterator       begin()  const { return iterator(_data, _size, 0); }
    iterator       end()    const { return iterator(_data, _size, _size); }
    const_iterator cbegin() const { return const_iterator(_data, _size, 0); }
    const_iterator cend()   const { return const_iterator(_data, _size, _size); }
#else
    typedef T* iterator;
    typedef T* const_iterator;
    iterator begin() const { return _data; }
    iterator end() const { return _data + _size; }
    const_iterator cbegin() const { return _data; }
    const_iterator cend() const { return _data + _size; }
#endif

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
