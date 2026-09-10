// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <utility>

#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Named OpenVINO output; shape and type come from OpenVINO inference.
// Default construction yields an empty handle, also used for absent optional weights.
class GGUF_FRONTEND_API GgufValue {
public:
    GgufValue() = default;

    GgufValue(std::string name, ov::Output<ov::Node> value) : m_name(std::move(name)), m_value(std::move(value)) {}

    const std::string& name() const {
        return m_name;
    }

    // Logical shape in OpenVINO order: [ne3, ne2, ne1, ne0] for a rank-4 value.
    const ov::PartialShape& shape() const {
        static const ov::PartialShape unknown = ov::PartialShape::dynamic();
        return empty() ? unknown : m_value.get_partial_shape();
    }

    ov::element::Type type() const {
        return empty() ? ov::element::dynamic : m_value.get_element_type();
    }

    // GGML dimension i, counted from the last shape axis. Returns -1 for an empty value
    // or dynamic dimension, and 1 beyond the rank.
    int64_t ne(size_t i) const;

    // False for an empty handle.
    explicit operator bool() const {
        return m_value.get_node() != nullptr;
    }

    bool empty() const {
        return m_value.get_node() == nullptr;
    }

private:
    friend class GgufGraphContext;
    std::string m_name;
    ov::Output<ov::Node> m_value;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
