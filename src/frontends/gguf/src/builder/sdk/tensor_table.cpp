// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Weight lookup for the builder SDK, and the small out-of-line pieces of the value handle and the
// model-builder base.

#include "openvino/frontend/gguf/builder/tensor_table.hpp"

#include "builder/sdk/graph_context_impl.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"
#include "openvino/frontend/gguf/builder/model_builder.hpp"
#include "openvino/frontend/gguf/builder/value.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace frontend {
namespace gguf {

int64_t GgufValue::ne(size_t i) const {
    if (m_empty || m_shape.rank().is_dynamic()) {
        return -1;
    }
    const size_t rank = m_shape.size();
    // ggml's ne[0] is the fastest-varying dimension, i.e. the LAST entry of the stored shape.
    // Beyond the tensor's rank ggml reports 1, every tensor being nominally 4D with trailing 1s.
    if (i >= rank) {
        return 1;
    }
    const auto& d = m_shape[rank - 1 - i];
    return d.is_static() ? d.get_length() : -1;
}

ModelBuilder::~ModelBuilder() = default;

bool GgufTensors::has(const std::string& gguf_name) const {
    return (*m_ctx->m_impl).emitter.has_weight(gguf_name);
}

GgufValue GgufTensors::operator()(const std::string& gguf_name) const {
    auto& impl = (*m_ctx->m_impl);
    auto& e = impl.emitter;
    if (!e.has_weight(gguf_name)) {
        // Absent is a normal, meaningful state: it is how GGUF encodes structure. Report it as an
        // empty value so a ported `if (layer.attn_q_norm)` works.
        return GgufValue();
    }
    impl.check_open();
    // Emission is idempotent -- a weight read repeatedly by a layer loop, or shared between ops,
    // becomes exactly one leaf.
    if (!e.weight_emitted(gguf_name)) {
        if (ov::util::ends_with(gguf_name, ".weight")) {
            e.add_weight(gguf_name);
        } else {
            e.add_named_weight(gguf_name);
        }
    }
    // The emitter's historical metadata is specialized for decoder matmuls. SDK values carry
    // the full logical shape instead, including vectors and expert/kernel dimensions.
    auto shape = e.weight_tensor(gguf_name).get_shape();
    OPENVINO_ASSERT(shape.size() <= 4, "[GGUF] weight rank exceeds GGML's four dimensions");
    shape.insert(shape.begin(), 4 - shape.size(), 1);
    e.set_tensor_meta(gguf_name, shape, e.type_of_tensor(gguf_name));
    return GgufValue(gguf_name, shape, e.type_of_tensor(gguf_name));
}

GgufValue GgufTensors::require(const std::string& gguf_name) const {
    auto v = (*this)(gguf_name);
    OPENVINO_ASSERT(v,
                    "[GGUF] model is missing expected weight tensor '",
                    gguf_name,
                    "' for architecture '",
                    m_ctx->arch(),
                    "'");
    return v;
}

GgufValue GgufTensors::layer(int il, const std::string& suffix) const {
    return (*this)("blk." + std::to_string(il) + "." + suffix);
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
