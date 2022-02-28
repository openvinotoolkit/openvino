// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include <ostream>
#include <utility>

namespace vpu {

//
// Max-heap. Collects all elements until capacity is reached; then collects only elements lesser than the max one
// Maintains its size not more than specified in constructor.
//

template <typename T>
class FixedMaxHeap {
    using BaseContainer = std::vector<T>;

public:
    explicit FixedMaxHeap(size_t capacity): _capacity(capacity) {
        _vec.reserve(_capacity);
    }

    FixedMaxHeap(size_t capacity, std::initializer_list<T> list) : FixedMaxHeap(capacity) {
        for (auto&& val : list) {
            push(val);
        }
    }

    FixedMaxHeap(const FixedMaxHeap&) = default;
    FixedMaxHeap& operator=(const FixedMaxHeap&) = default;

    FixedMaxHeap(FixedMaxHeap&&) = default;
    FixedMaxHeap& operator=(FixedMaxHeap&&) = default;

    auto begin() -> decltype(std::declval<BaseContainer&>().begin()) {
        return _vec.begin();
    }
    auto end() -> decltype(std::declval<BaseContainer&>().end()) {
        return _vec.end();
    }

    auto begin() const -> decltype(std::declval<const BaseContainer&>().begin()) {
        return _vec.begin();
    }
    auto end() const -> decltype(std::declval<const BaseContainer&>().end()) {
        return _vec.end();
    }

    auto cbegin() const -> decltype(std::declval<const BaseContainer&>().cbegin()) {
        return _vec.begin();
    }
    auto cend() const -> decltype(std::declval<const BaseContainer&>().cend()) {
        return _vec.end();
    }

    auto front() -> decltype(std::declval<BaseContainer&>().front()) {
        return _vec.front();
    }
    auto front() const -> decltype(std::declval<const BaseContainer&>().front()) {
        return _vec.front();
    }

    auto size() const -> decltype(std::declval<const BaseContainer&>().size()) {
        return _vec.size();
    }
    bool empty() const {
        return _vec.empty();
    }

    // Keep max-heap of constant size: insert only values smaller than max element, discard others.
    bool push(const T& val) {
        return pushImpl(val);
    }
    bool push(T&& val) {
        return pushImpl(val);
    }

    // Return sorted Range of Heap elements.
    std::vector<T> sorted() const {
        std::vector<T> s = _vec;
        std::sort_heap(s.begin(), s.end());
        return s;
    }

    friend std::ostream& operator<<(std::ostream& stream, const FixedMaxHeap& heap) {
        stream << "Heap [" << heap._vec.size() << " / " << heap._capacity << "]: ";
        for (const auto& val : heap._vec) {
            stream << val << " ";
        }
        return stream;
    }

private:
    template <typename U>
    bool pushImpl(U&& val) {
        if (_capacity == 0) {
            return false;
        }

        if (_vec.size() < _capacity) {
            _vec.push_back(std::forward<U>(val));
        } else {
            if (!(val < _vec.front())) {
                return false;
            }
            std::pop_heap(_vec.begin(), _vec.end());
            _vec[_capacity - 1] = std::forward<U>(val);
        }

        std::push_heap(_vec.begin(), _vec.end());

        return true;
    }

private:
    size_t _capacity;
    BaseContainer _vec;
};

}  // namespace vpu
