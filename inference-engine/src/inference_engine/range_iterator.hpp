// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>

namespace InferenceEngine {

/**
 * @Brief iterator for accesing standard c-style null terminated strings withing c++ algorithms
 * @tparam Char
 */
template<typename Char>
struct null_terminated_range_iterator : public std::iterator<std::forward_iterator_tag, Char> {
 public:
    null_terminated_range_iterator() = delete;

    // make a non-end iterator (well, unless you pass nullptr ;)
    explicit null_terminated_range_iterator(Char *ptr) : ptr(ptr) {}

    bool operator != (null_terminated_range_iterator const &that) const {
        // iterators are equal if they point to the same location
        return !(operator==(that));
    }

    bool operator == (null_terminated_range_iterator const &that) const {
        // iterators are equal if they point to the same location
        return ptr == that.ptr
            // or if they are both end iterators
            || (is_end() && that.is_end());
    }

    null_terminated_range_iterator<Char> &operator++() {
        get_accessor()++;
        return *this;
    }

    null_terminated_range_iterator<Char> &operator++(int) {
        return this->operator++();
    }

    Char &operator*() {
        return *get_accessor();
    }

 protected:
    Char *& get_accessor()  {
        if (ptr == nullptr) {
            throw std::logic_error("null_terminated_range_iterator dereference: pointer is zero");
        }
        return ptr;
    }
    bool is_end() const {
        // end iterators can be created by the default ctor
        return !ptr
            // or by advancing until a null character
            || !*ptr;
    }

    Char *ptr;
};

template<typename Char>
struct null_terminated_range_iterator_end : public null_terminated_range_iterator<Char> {
 public:
    // make an end iterator
    null_terminated_range_iterator_end() :  null_terminated_range_iterator<Char>(nullptr) {
        null_terminated_range_iterator<Char>::ptr = nullptr;
    }
};


inline null_terminated_range_iterator<const char> null_terminated_string(const char *a) {
    return null_terminated_range_iterator<const char>(a);
}

inline null_terminated_range_iterator<const char> null_terminated_string_end() {
    return null_terminated_range_iterator_end<const char>();
}

}  // namespace InferenceEngine
