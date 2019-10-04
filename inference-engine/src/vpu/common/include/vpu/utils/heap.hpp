// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <vector>
#include <ostream>
#include <utility>

namespace vpu {

// Max-heap. Collects all elements until capacity is reached; then collects only elements lesser than the max one
// Maintains its size not more than specified in constructor.
template <typename T>
class FixedMaxHeap {
private:
    size_t _capacity;
    std::vector<T> v;

public:
    explicit FixedMaxHeap(size_t capacity): _capacity(capacity) {
        v.reserve(_capacity);
    }

    FixedMaxHeap(const FixedMaxHeap&) = delete;

    FixedMaxHeap(FixedMaxHeap &&other): _capacity(other._capacity), v(std::move(other.v)) {
    }

    FixedMaxHeap& operator=(FixedMaxHeap &&other) {
        _capacity = other._capacity;
        v = std::move(other.v);
        return *this;
    }

    auto begin() -> decltype(v.begin()) {
        return v.begin();
    }

    auto end() -> decltype(v.begin()) {
        return v.end();
    }

    auto begin() const -> decltype(v.begin()) const {
        return v.begin();
    }

    auto end() const -> decltype(v.begin()) const {
        return v.end();
    }

    bool empty() const {
        return v.empty();
    }

    size_t size() const {
        return v.size();
    }

    // keep max-heap of constant size: insert only values smaller than max element, discard others
    bool push(const T& val) {
        if (_capacity == 0) {
            return false;
        }

        if (v.size() < _capacity) {
            v.push_back(val);
        } else {
            if (!(val < v.front())) {
                return false;
            }
            std::pop_heap(v.begin(), v.end());
            v[_capacity - 1] = val;
        }

        std::push_heap(v.begin(), v.end());

        return true;
    }

    std::vector<T> sorted() const {
        std::vector<T> s = v;
        std::sort_heap(s.begin(), s.end());
        return s;
    }

    void print(std::ostream &o) const {
        o << "Heap [" << v.size() << "]: ";
        for (int i : v) {
            o << i << " ";
        }
        o << " is_heap: " << std::is_heap(v.begin(), v.end()) << " ";
        o << std::endl;
    }

    friend std::ostream& operator<<(std::ostream& o, const FixedMaxHeap &h) {
        h.print(o);
        return o;
    }
};

}  // namespace vpu
