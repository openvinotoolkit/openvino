// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cassert>
#include <cstdint>

#include <limits>
#include <utility>
#include <type_traits>
#include <iterator>
#include <vector>
#include <memory>
#include <array>

namespace vpu {

namespace details {

template <typename T>
struct SmallBufElemMemory {
    static constexpr const size_t ElemSize = sizeof(T);

#ifdef _WIN32
    static constexpr const size_t ExtraSize = 16;
#else
    static constexpr const size_t ExtraSize = 0;
#endif

    static constexpr const size_t Align = alignof(T);

    typename std::aligned_storage<ElemSize + ExtraSize, Align>::type mem;
};

template <typename T, int _Capacity>
struct SmallBufHolder {
    using ElemMemory = SmallBufElemMemory<T>;

    static constexpr const size_t ElemSize = sizeof(ElemMemory);
    static constexpr const int Capacity = _Capacity;

    std::array<ElemMemory, Capacity> buf = {};
    bool bufLocked = false;
};

template <typename T, class BufHolder, class BaseAllocator = std::allocator<T>>
class SmallBufAllocator {
    using ElemMemory = typename BufHolder::ElemMemory;

    static constexpr const size_t ElemSize = BufHolder::ElemSize;
    static constexpr const int Capacity = BufHolder::Capacity;

    static_assert(sizeof(T) <= ElemSize, "sizeof(T) <= ElemSize");

public:
    using value_type = typename std::allocator_traits<BaseAllocator>::value_type;

    using pointer = typename std::allocator_traits<BaseAllocator>::pointer;
    using const_pointer = typename std::allocator_traits<BaseAllocator>::const_pointer;
    using void_pointer = typename std::allocator_traits<BaseAllocator>::void_pointer;
    using const_void_pointer = typename std::allocator_traits<BaseAllocator>::const_void_pointer;

    using size_type = typename std::allocator_traits<BaseAllocator>::size_type;
    using difference_type = typename std::allocator_traits<BaseAllocator>::difference_type;

    using propagate_on_container_copy_assignment = std::false_type;
    using propagate_on_container_move_assignment = std::false_type;
    using propagate_on_container_swap = std::false_type;

    template <typename T2> struct rebind {
        typedef SmallBufAllocator<
            T2, BufHolder,
            typename std::allocator_traits<BaseAllocator>::template rebind_alloc<T2>
        > other;
    };

    SmallBufAllocator() = default;
    explicit SmallBufAllocator(const BaseAllocator& baseAllocator) :
            _baseAllocator(baseAllocator) {
    }

    explicit SmallBufAllocator(BufHolder& h) :
            _buf(h.buf.data()), _bufLocked(&h.bufLocked) {
        *_bufLocked = false;
    }
    SmallBufAllocator(BufHolder& h, const BaseAllocator& baseAllocator) :
            _baseAllocator(baseAllocator),
            _buf(h.buf.data()), _bufLocked(&h.bufLocked) {
        *_bufLocked = false;
    }

    SmallBufAllocator(const SmallBufAllocator& other) :
            _baseAllocator(other._baseAllocator),
            _buf(other._buf), _bufLocked(other._bufLocked) {
    }
    SmallBufAllocator& operator=(const SmallBufAllocator& other) {
        if (&other != this) {
#ifndef NDEBUG
            if (_buf != nullptr && _bufLocked != nullptr) {
                assert(!*_bufLocked);
            }
#endif

            _baseAllocator = other._baseAllocator;
            _buf = other._buf;
            _bufLocked = other._bufLocked;
        }

        return *this;
    }

    template <typename T2, typename BufHolder2, class BaseAllocator2>
    SmallBufAllocator(const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& other) :
            _baseAllocator(other._baseAllocator),
            _buf(other._buf), _bufLocked(other._bufLocked) {
        static_assert(
            sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize,
            "sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize");
    }
    template <typename T2, typename BufHolder2, class BaseAllocator2>
    SmallBufAllocator& operator=(const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& other) {
        static_assert(
            sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize,
            "sizeof(T) <= SmallBufAllocator<T2, BufHolder2, BaseAllocator2>::ElemSize");

        if (&other != this) {
#ifndef NDEBUG
            if (_buf != nullptr && _bufLocked != nullptr) {
                assert(!*_bufLocked);
            }
#endif

            _baseAllocator = other._baseAllocator;
            _buf = other._buf;
            _bufLocked = other._bufLocked;
        }

        return *this;
    }

    pointer allocate(size_type n, const_void_pointer hint = const_void_pointer()) {
        if (n <= Capacity && _buf != nullptr && _bufLocked != nullptr) {
            if (!*_bufLocked) {
                *_bufLocked = true;
                return static_cast<pointer>(_buf);
            }
        }

        return std::allocator_traits<BaseAllocator>::allocate(_baseAllocator, n, hint);
    }

    void deallocate(pointer ptr, size_type n) {
        if (_buf != nullptr && _bufLocked != nullptr) {
            if (ptr == static_cast<pointer>(_buf)) {
                assert(*_bufLocked);
                *_bufLocked = false;
                return;
            }
        }

        _baseAllocator.deallocate(ptr, n);
    }

    template <class U, class ...Args>
    void construct(U* ptr, Args&& ...args) {
        _baseAllocator.construct(ptr, std::forward<Args>(args)...);
    }

    template <class U>
    void destroy(U* ptr) {
        _baseAllocator.destroy(ptr);
    }

    void* getBuf() const { return _buf; }
    const BaseAllocator& getBaseAllocator() const { return _baseAllocator; }

private:
    BaseAllocator _baseAllocator;

    void* _buf = nullptr;
    bool* _bufLocked = nullptr;

    template <typename T2, typename BufHolder2, class BaseAllocator2>
    friend class SmallBufAllocator;
};

template <
    typename T1, typename BufHolder1, class BaseAllocator1,
    typename T2, typename BufHolder2, class BaseAllocator2
>
bool operator==(
        const SmallBufAllocator<T1, BufHolder1, BaseAllocator1>& a1,
        const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& a2) {
    return a1.getBuf() == a2.getBuf() && a1.getBaseAllocator() == a2.getBaseAllocator();
}
template <
    typename T1, typename BufHolder1, class BaseAllocator1,
    typename T2, typename BufHolder2, class BaseAllocator2
>
bool operator!=(
        const SmallBufAllocator<T1, BufHolder1, BaseAllocator1>& a1,
        const SmallBufAllocator<T2, BufHolder2, BaseAllocator2>& a2) {
    return a1.getBuf() != a2.getBuf() || a1.getBaseAllocator() != a2.getBaseAllocator();
}

}  // namespace details

template <typename T, int Capacity = 8, class BaseAllocator = std::allocator<T>>
class SmallVector {
    using BufHolder = details::SmallBufHolder<T, Capacity>;
    using Alloc = details::SmallBufAllocator<T, BufHolder, BaseAllocator>;
    using BaseCont = std::vector<T, Alloc>;

public:
    using value_type = typename BaseCont::value_type;

    using size_type = typename BaseCont::size_type;

    using iterator = typename BaseCont::iterator;
    using const_iterator = typename BaseCont::const_iterator;
    using reverse_iterator = typename BaseCont::reverse_iterator;
    using const_reverse_iterator = typename BaseCont::const_reverse_iterator;

    SmallVector() : _allocator(_bufs), _base(_allocator) {
        _base.reserve(Capacity);
    }

    ~SmallVector() = default;

    explicit SmallVector(size_type count) : _allocator(_bufs), _base(count, T(), _allocator) {}
    SmallVector(size_type count, const T& value) : _allocator(_bufs), _base(count, value, _allocator) {}
    SmallVector(std::initializer_list<T> init) : _allocator(_bufs), _base(init, _allocator) {}

    template <class InputIt>
    SmallVector(InputIt first, InputIt last) : _allocator(_bufs), _base(first, last, _allocator) {}

    SmallVector(const SmallVector& other) : _allocator(_bufs), _base(other._base, _allocator) {}
    SmallVector& operator=(const SmallVector& other) {
        if (&other != this) {
            _base = other._base;
        }
        return *this;
    }

    template <typename T2, int Capacity2, class BaseAllocator2>
    SmallVector(const SmallVector<T2, Capacity2, BaseAllocator2>& other) :  // NOLINT
            _allocator(_bufs), _base(other._base.begin(), other._base.end(), _allocator) {
    }
    template <typename T2, int Capacity2, class BaseAllocator2>
    SmallVector& operator=(const SmallVector<T2, Capacity2, BaseAllocator2>& other) {
        if (&other != this) {
            _base.assign(other._base.begin(), other._base.end());
        }
        return *this;
    }

    template <class Alloc2>
    SmallVector(const std::vector<T, Alloc2>& other) :  // NOLINT
            _allocator(_bufs), _base(other.begin(), other.end(), _allocator) {
    }
    template <class Alloc2>
    SmallVector& operator=(const std::vector<T, Alloc2>& other) {
        if (&other != this) {
            _base.assign(other.begin(), other.end());
        }
        return *this;
    }

    operator const BaseCont&() {
        return _base;
    }
    template <class Alloc2>
    operator std::vector<T, Alloc2>() {
        return std::vector<T, Alloc2>(_base.begin(), _base.end());
    }

    T& operator[](size_type pos) { return _base[pos]; }
    const T& operator[](size_type pos) const { return _base[pos]; }

    T& at(size_type pos) { return _base.at(pos); }
    const T& at(size_type pos) const { return _base.at(pos); }

    T& front() { return _base.front(); }
    const T& front() const { return _base.front(); }
    T& back() { return _base.back(); }
    const T& back() const { return _base.back(); }

    T* data() noexcept { return _base.data(); }
    const T* data() const noexcept { return _base.data(); }

    iterator begin() noexcept { return _base.begin(); }
    iterator end() noexcept { return _base.end(); }
    const_iterator begin() const noexcept { return _base.begin(); }
    const_iterator end() const noexcept { return _base.end(); }
    const_iterator cbegin() const noexcept { return _base.cbegin(); }
    const_iterator cend() const noexcept { return _base.cend(); }

    reverse_iterator rbegin() noexcept { return _base.rbegin(); }
    reverse_iterator rend() noexcept { return _base.rend(); }
    const_reverse_iterator rbegin() const noexcept { return _base.rbegin(); }
    const_reverse_iterator rend() const noexcept { return _base.rend(); }
    const_reverse_iterator crbegin() const noexcept { return _base.crbegin(); }
    const_reverse_iterator crend() const noexcept { return _base.crend(); }

    bool empty() const noexcept { return _base.empty(); }
#if ENABLE_MYRIAD
    int size() const noexcept { return static_cast<int>(_base.size()); }
#else
    size_t size() const noexcept { return _base.size(); }
#endif

    void reserve(size_type cap) { _base.reserve(cap); }

    void clear() noexcept { _base.clear(); }

    void resize(size_type count) { _base.resize(count); }
    void resize(size_type count, const T& value) { _base.resize(count, value); }

    void push_back(const T& value) { _base.push_back(value); }
    void push_back(T&& value) { _base.push_back(value); }

    template <class... Args>
    void emplace_back(Args&&... args) { _base.emplace_back(std::forward<Args>(args)...); }

    void insert(iterator pos, const T& value) { _base.insert(pos, value); }
    void insert(iterator pos, T&& value) { _base.insert(pos, value); }
    void insert(iterator pos, size_type count, const T& value) { _base.insert(pos, count, value); }
    template <class InputIt>
    void insert(iterator pos, InputIt first, InputIt last) { _base.insert(pos, first, last); }
    void insert(iterator pos, std::initializer_list<T> ilist) { _base.insert(pos, ilist); }

    template <class... Args>
    iterator emplace(iterator pos, Args&&... args) { return _base.emplace(pos, std::forward<Args>(args)...); }

    void assign(size_type count, const T& value) { _base.assign(count, value); }
    template <class InputIt>
    void assign(InputIt first, InputIt last) { _base.assign(first, last); }
    void assign(std::initializer_list<T> ilist) { _base.assign(ilist); }

    void pop_back() { _base.pop_back(); }

    iterator erase(iterator pos) { return _base.erase(pos); }
    iterator erase(iterator first, iterator last) { return _base.erase(first, last); }

    void swap(SmallVector& other) { std::swap(*this, other); }

    bool operator==(const SmallVector& other) const { return _base == other._base; }
    bool operator!=(const SmallVector& other) const { return _base != other._base; }
    bool operator<(const SmallVector& other) const { return _base < other._base; }
    bool operator<=(const SmallVector& other) const { return _base <= other._base; }
    bool operator>(const SmallVector& other) const { return _base > other._base; }
    bool operator>=(const SmallVector& other) const { return _base >= other._base; }

private:
    template <typename T2, int Capacity2, class BaseAllocator2>
    friend class SmallVector;

    BufHolder _bufs;
    Alloc _allocator;
    BaseCont _base;
};

}  // namespace vpu
