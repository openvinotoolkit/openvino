// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Core-local partial shape layer. Minimal replacement for
// openvino/core/partial_shape.hpp (plus ov::shape_size from shape_util.hpp).

#pragma once

#include <algorithm>
#include <numeric>
#include <vector>

#include "dimension.hpp"
#include "shape.hpp"

namespace ov {

class Rank {
public:
    Rank() : m_rank(-1) {}
    explicit Rank(int64_t value) : m_rank(value) {}

    int64_t get_length() const { return m_rank; }
    bool is_dynamic() const { return m_rank < 0; }
    bool is_static() const { return m_rank >= 0; }

private:
    int64_t m_rank;
};

class PartialShape : public std::vector<Dimension> {
public:
    PartialShape() = default;
    PartialShape(const std::vector<Dimension>& dims) : std::vector<Dimension>(dims) {}
    PartialShape(std::initializer_list<Dimension> dims) : std::vector<Dimension>(dims) {}
    explicit PartialShape(const Shape& shape) : std::vector<Dimension>(shape.begin(), shape.end()) {}
    template <typename InputIt>
    PartialShape(InputIt first, InputIt last) : std::vector<Dimension>(first, last) {}

    Rank rank() const { return Rank(static_cast<int64_t>(size())); }

    bool is_static() const {
        return std::all_of(begin(), end(), [](const Dimension& d) { return d.is_static(); });
    }
    bool is_dynamic() const {
        return std::any_of(begin(), end(), [](const Dimension& d) { return d.is_dynamic(); });
    }
};

inline size_t shape_size(const Shape& shape) {
    return std::accumulate(shape.begin(), shape.end(), size_t(1), [](size_t a, int64_t b) {
        return a * static_cast<size_t>(b < 0 ? 1 : b);
    });
}

}  // namespace ov
