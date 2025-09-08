// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Based on the Matrix class in
// The C++ Programming Language
// Fourth edition
// Bjarne Stroustrup
// Addison-Wesley, Boston, 2013.

#pragma once

#include "openvino/core/shape.hpp"

namespace ov {
namespace test {
namespace init {
// Recursively define types for N-deep initializer lists
template <typename T, size_t N>
struct NestedInitializerListWrapper {
    using type = std::initializer_list<typename NestedInitializerListWrapper<T, N - 1>::type>;
};

// 1-deep is a plain initializer_list
template <typename T>
struct NestedInitializerListWrapper<T, 1> {
    using type = std::initializer_list<T>;
};

// Scalar case is just the element type
template <typename T>
struct NestedInitializerListWrapper<T, 0> {
    using type = T;
};

// Convenience type name for N-deep initializer lists of Ts
template <typename T, size_t N>
using NestedInitializerList = typename NestedInitializerListWrapper<T, N>::type;

// Fill in a shape from a nested initializer list
// For a scalar, nothing to do.
template <typename T, size_t N>
typename std::enable_if<(N == 0), void>::type fill_shape(ov::Shape& /* shape */,
                                                         const NestedInitializerList<T, N>& /* inits */) {}

// Check that the inits match the shape
template <typename T, size_t N>
typename std::enable_if<(N == 0), void>::type check_shape(const ov::Shape& shape,
                                                          const NestedInitializerList<T, N>& /* inits */) {
    if (shape.size() != 0) {
        throw std::invalid_argument("Initializers do not match shape");
    }
}

// For a plain initializer list, the shape is the length of the list.
template <typename T, size_t N>
typename std::enable_if<(N == 1)>::type fill_shape(ov::Shape& shape, const NestedInitializerList<T, N>& inits) {
    shape.push_back(inits.size());
}

template <typename T, size_t N>
typename std::enable_if<(N == 1)>::type check_shape(const ov::Shape& shape, const NestedInitializerList<T, N>& inits) {
    if (shape.at(shape.size() - N) != inits.size()) {
        throw std::invalid_argument("Initializers do not match shape");
    }
}

// In the general case, we append our level's length and recurse.
template <typename T, size_t N>
typename std::enable_if<(N > 1), void>::type fill_shape(ov::Shape& shape, const NestedInitializerList<T, N>& inits) {
    shape.push_back(inits.size());
    fill_shape<T, N - 1>(shape, *inits.begin());
}

template <typename T, size_t N>
typename std::enable_if<(N > 1), void>::type check_shape(const ov::Shape& shape,
                                                         const NestedInitializerList<T, N>& inits) {
    if (shape.at(shape.size() - N) != inits.size()) {
        throw std::invalid_argument("Initializers do not match shape");
    }
    for (auto it : inits) {
        check_shape<T, N - 1>(shape, it);
    }
}

// Get the shape of inits.
template <typename T, size_t N>
ov::Shape get_shape(const NestedInitializerList<T, N>& inits) {
    ov::Shape shape;
    fill_shape<T, N>(shape, inits);
    check_shape<T, N>(shape, inits);
    return shape;
}

template <typename IT, typename T, size_t N>
typename std::enable_if<(N == 1), IT>::type flatten(IT it,
                                                    const ov::Shape& shape,
                                                    const NestedInitializerList<T, N>& inits) {
    if (inits.size() != shape.at(shape.size() - N)) {
        throw std::invalid_argument("Initializers do not match shape");
    }
    for (auto it1 : inits) {
        *(it++) = it1;
    }
    return it;
}

template <typename IT, typename T, size_t N>
typename std::enable_if<(N > 1), IT>::type flatten(IT it,
                                                   const ov::Shape& shape,
                                                   const NestedInitializerList<T, N>& inits) {
    if (inits.size() != shape.at(shape.size() - N)) {
        throw std::invalid_argument("Initializers do not match shape");
    }
    for (auto it1 : inits) {
        it = flatten<IT, T, N - 1>(it, shape, it1);
    }
    return it;
}

template <typename IT, typename T, size_t N>
typename std::enable_if<(N == 0), IT>::type flatten(IT it,
                                                    const ov::Shape& shape,
                                                    const NestedInitializerList<T, 0>& init) {
    if (shape.size() != 0) {
        throw std::invalid_argument("Initializers do not match shape");
    }
    *(it++) = init;
    return it;
}
}  // namespace init

template <typename T>
class NDArrayBase {
    using vtype = std::vector<T>;

public:
    using type = T;
    using iterator = typename vtype::iterator;
    using const_iterator = typename vtype::const_iterator;

    NDArrayBase(const ov::Shape& shape) : m_shape(shape), m_elements(shape_size(m_shape)) {}

    const ov::Shape& get_shape() const {
        return m_shape;
    }
    const_iterator begin() const {
        return m_elements.begin();
    }
    const_iterator end() const {
        return m_elements.end();
    }
    vtype get_vector() {
        return m_elements;
    }
    const vtype get_vector() const {
        return m_elements;
    }
    operator const vtype() const {
        return m_elements;
    }
    operator vtype() {
        return m_elements;
    }
    void* data() {
        return m_elements.data();
    }
    const void* data() const {
        return m_elements.data();
    }
    bool operator==(const NDArrayBase<T>& other) const {
        return m_shape == other.m_shape && m_elements == other.m_elements;
    }

protected:
    ov::Shape m_shape;
    vtype m_elements;
};

/// An N dimensional array of elements of type T
template <typename T, size_t N>
class NDArray : public NDArrayBase<T> {
public:
    NDArray(const init::NestedInitializerList<T, N>& initial_value)
        : NDArrayBase<T>(init::get_shape<T, N>(initial_value)) {
        init::flatten<typename std::vector<T>::iterator, T, N>(NDArrayBase<T>::m_elements.begin(),
                                                               NDArrayBase<T>::m_shape,
                                                               initial_value);
    }
};
}  // namespace test
}  // namespace ov
