// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
    if (empty() || shape().rank().is_dynamic()) {
        return -1;
    }
    const size_t rank = shape().size();
    if (i >= rank) {
        return 1;
    }
    const auto& d = shape()[rank - 1 - i];
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
        return GgufValue();
    }
    impl.check_open();
    if (!e.weight_emitted(gguf_name)) {
        if (ov::util::ends_with(gguf_name, ".weight")) {
            e.add_weight(gguf_name);
        } else {
            e.add_named_weight(gguf_name);
        }
    }
    return GgufValue(gguf_name, e.value(gguf_name));
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
