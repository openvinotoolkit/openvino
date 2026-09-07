// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Named tensor handle carrying shape and type for graph construction.
// Default construction yields an empty handle, also used for absent optional weights.
class GGUF_FRONTEND_API GgufValue {
public:
    GgufValue() = default;

    GgufValue(std::string name, ov::PartialShape shape, ov::element::Type type)
        : m_name(std::move(name)),
          m_shape(std::move(shape)),
          m_type(type),
          m_empty(false) {}

    const std::string& name() const {
        return m_name;
    }

    // Logical shape in OpenVINO order: [ne3, ne2, ne1, ne0] for a rank-4 value.
    const ov::PartialShape& shape() const {
        return m_shape;
    }

    ov::element::Type type() const {
        return m_type;
    }

    // GGML dimension i, counted from the last shape axis. Returns -1 for an empty value
    // or dynamic dimension, and 1 beyond the rank.
    int64_t ne(size_t i) const;

    // False for an empty handle.
    explicit operator bool() const {
        return !m_empty;
    }

    bool empty() const {
        return m_empty;
    }

private:
    std::string m_name;
    ov::PartialShape m_shape;
    ov::element::Type m_type = ov::element::dynamic;
    bool m_empty = true;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
