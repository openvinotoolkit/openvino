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
//TODO - delete this class when c++20 in project and use std::span instead
#if defined(_ITERATOR_DEBUG_LEVEL) && _ITERATOR_DEBUG_LEVEL != 0
template <class _Ptr>
class checked_array_iterator {
private:
    using _Pointee_type     = std::remove_pointer_t<_Ptr>;
    static_assert(std::is_pointer_v<_Ptr> && std::is_object_v<_Pointee_type>,
        "checked_array_iterator requires pointers to objects");
public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type        = std::remove_cv_t<std::remove_pointer_t<_Ptr>>;
    using difference_type   = std::ptrdiff_t;
    using pointer           = _Ptr;
    using reference         = std::remove_pointer_t<_Ptr>&;

    checked_array_iterator() = default;
    
    checked_array_iterator(const pointer data, const std::size_t size, const std::size_t index = 0) noexcept
        : _Myptr(data), _Mysize(size), _Myoff(index) {
        _STL_VERIFY(index <= size, "cldnn::checked_array_iterator out of range");
    }

    template <class _Ty = _Pointee_type, std::enable_if_t<!std::is_const_v<_Ty>, int> = 0>
    constexpr operator checked_array_iterator<const _Ty*>() const noexcept {
        return checked_array_iterator<const _Ty*>{_Myptr, _Mysize, _Myoff};
    }

    _NODISCARD constexpr _Ptr base() const noexcept {
        return _Myptr + _Myoff;
    }

    _NODISCARD constexpr reference operator*() const noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot dereference value-initialized iterator");
        _STL_VERIFY(_Myoff < _Mysize, "cannot dereference end array iterator");
        return _Myptr[_Myoff];
    }

    _NODISCARD constexpr pointer operator->() const noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot dereference value-initialized iterator");
        _STL_VERIFY(_Myoff < _Mysize, "cannot dereference end array iterator");
        return _Myptr + _Myoff;
    }

    constexpr checked_array_iterator& operator++() noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot increment value-initialized iterator");
        _STL_VERIFY(_Myoff < _Mysize, "cannot increment iterator past end");
        ++_Myoff;
        return *this;
    }

    constexpr checked_array_iterator operator++(int) noexcept { 
        auto tmp = *this; 
        ++(*this); 
        return tmp; 
    }

    constexpr checked_array_iterator& operator--() noexcept {
        _STL_VERIFY(_Myptr != nullptr, "cannot decrement value-initialized iterator");
        _STL_VERIFY(_Myoff != 0, "cannot decrement iterator before begin");
        --_Myoff;
        return *this;
    }

    constexpr checked_array_iterator operator--(int) noexcept { 
        auto tmp = *this; 
        --(*this); 
        return tmp; 
    }

    constexpr checked_array_iterator& operator+=(difference_type n) noexcept {
        _Verify_offset(n);
        _Myoff += static_cast<std::size_t>(n);
        return *this;
    }

    _NODISCARD constexpr checked_array_iterator operator+(const difference_type n) const noexcept { 
        auto _Tmp = *this;
        _Tmp += n;
        return _Tmp;
    }

    _NODISCARD friend constexpr checked_array_iterator operator+(const difference_type n, const checked_array_iterator<_Ptr>& it) noexcept { 
        return it + n; 
    }

    constexpr checked_array_iterator& operator-=(const difference_type n) noexcept { 
        return *this += -n; 
    }

    _NODISCARD constexpr checked_array_iterator operator-(const difference_type n) const noexcept { 
        auto _Tmp = *this;
        _Tmp -= n;
        return _Tmp;
    }

    _NODISCARD constexpr difference_type operator-(const checked_array_iterator& a) const noexcept {
        _Compat(a);
        return static_cast<difference_type>(_Myoff) - static_cast<difference_type>(a._Myoff);
    }

    _NODISCARD constexpr reference operator[](const difference_type n) const noexcept {
        return *(*this + n);
    }

    _NODISCARD constexpr bool operator==(const checked_array_iterator& a) const noexcept {
        _Compat(a);
        return _Myoff == a._Myoff;
    }

    _NODISCARD constexpr bool operator!=(const checked_array_iterator& a) const noexcept {
        _Compat(a);
        return _Myoff != a._Myoff;
    }
    _NODISCARD constexpr bool operator<(const checked_array_iterator& a) const noexcept {
        _Compat(a);
        return Myoff < a._Myoff;
    }
    _NODISCARD constexpr bool operator>(const checked_array_iterator& a) const noexcept {
        _Compat(a);
        return _Myoff > a._Myoff;
    }
    _NODISCARD constexpr bool operator<=(const checked_array_iterator& a) const noexcept {
        _Compat(a);
        return _Myoff <= a._Myoff;
    }
    _NODISCARD constexpr bool operator>=(const checked_array_iterator& a) const noexcept {
        _Compat(a);
        return _Myoff >= a._Myoff;
    }

    // MSVC-style unwrapping helpers used by the iterator->const_iterator ctor above.
    pointer      _Unwrapped_base() const noexcept { return _Myptr; }
    std::size_t _Unwrapped_size() const noexcept { return _Mysize; }
    std::size_t _Unwrapped_off()  const noexcept { return _Myoff; }

    friend constexpr void _Verify_range(
        const checked_array_iterator& _First, const checked_array_iterator& _Last) noexcept {
        _STL_VERIFY(_First._Myptr == _Last._Myptr && _First._Mysize == _Last._Mysize,
            "mismatching checked_array_iterators");
        _STL_VERIFY(_First._Myoff <= _Last._Myoff, "transposed checked_array_iterator range");
    }

    constexpr void _Verify_offset(const difference_type n) const noexcept {
        if (n < 0) {
            _STL_VERIFY(static_cast<std::size_t>(-n) <= _Myoff, "cannot seek array iterator before begin");
        } else if (n > 0) {
            _STL_VERIFY(static_cast<std::size_t>(n) <= _Mysize - _Myoff, "cannot seek array iterator after end");
        }
    }

    using _Prevent_inheriting_unwrap = checked_array_iterator;

    _NODISCARD constexpr _Ptr _Unwrapped() const noexcept {
        return _Myptr + _Myoff;
    }

    constexpr void _Seek_to(_Ptr _It) noexcept {
        _Myoff = static_cast<size_t>(_It - _Myptr);
    }

private:
    void _Compat(const checked_array_iterator& other) const noexcept {
        _STL_VERIFY(_Myptr == other._Myptr && _Mysize == other._Mysize,
                    "array iterators from different ranges are incompatible");
    }

    pointer     _Myptr  = nullptr;
    std::size_t _Mysize = 0;
    std::size_t _Myoff  = 0;
};

template <class _Ptr>
_NODISCARD constexpr checked_array_iterator<_Ptr> make_checked_array_iterator(
    const _Ptr _Myptr, const size_t _Size, const size_t _Myoff = 0) noexcept {
    return checked_array_iterator<_Ptr>(_Myptr, _Size, _Myoff);
}

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
    typedef checked_array_iterator<T*> iterator;
    typedef checked_array_iterator<const T*> const_iterator;
    iterator begin() const { return make_checked_array_iterator(_data, _size); }
    iterator end() const { return make_checked_array_iterator(_data, _size, _size); }
    const_iterator cbegin() const { return make_checked_array_iterator(_data, _size); }
    const_iterator cend() const { return make_checked_array_iterator(_data, _size, _size); }
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
