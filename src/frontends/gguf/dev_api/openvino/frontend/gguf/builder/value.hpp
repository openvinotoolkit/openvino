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

// A handle to one tensor in the graph under construction: the counterpart of `ggml_tensor *` in a
// llama.cpp model file.
//
// The builder identifies tensors by NAME, and every emitted op must declare its output shape and
// type. Carrying all three together is what lets a ported llama.cpp graph be written as
// `cur = ctx.add(cur, inpSA)` instead of threading names and shapes by hand: GgufGraphContext
// infers each op's output shape from the shapes its inputs carry.
//
// An EMPTY value is the port of llama.cpp's null `ggml_tensor *`. Model files lean on that idiom
// constantly -- `build_norm(cur, model.layers[il].attn_norm, NULL, ...)`, or a weight that only
// some checkpoints of an architecture carry -- so a GgufTensors lookup for a tensor the file does
// not have returns an empty value rather than throwing, and `if (w)` ports unchanged.
class GGUF_FRONTEND_API GgufValue {
public:
    // An empty value: the port of a null ggml_tensor*.
    GgufValue() = default;

    GgufValue(std::string name, ov::PartialShape shape, ov::element::Type type)
        : m_name(std::move(name)),
          m_shape(std::move(shape)),
          m_type(type),
          m_empty(false) {}

    // Tensor name in the graph; this is what op inputs reference.
    const std::string& name() const {
        return m_name;
    }

    // Shape in the OpenVINO/GGML logical order [ne3, ne2, ne1, ne0] -- the REVERSE of ggml's
    // ne[] indexing. See ne() for the ggml-order accessor a port should use.
    const ov::PartialShape& shape() const {
        return m_shape;
    }

    ov::element::Type type() const {
        return m_type;
    }

    // Extent of ggml dimension `i`, i.e. ggml's `t->ne[i]`, so a ported expression like
    // `cur->ne[0]` reads the same here. Dimension 0 is the fastest-varying one, which is the LAST
    // entry of shape(). Returns -1 for a dynamic dimension, and 1 for an axis beyond the rank
    // (matching ggml, where every tensor is nominally 4D with trailing 1s).
    int64_t ne(size_t i) const;

    // False for an empty value, so `if (w) { ... }` and `w ? ... : ...` port directly from a
    // llama.cpp null-tensor check. explicit, to keep it out of arithmetic contexts.
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
