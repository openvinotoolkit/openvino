// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>

#include "openvino/frontend/gguf/builder/value.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov {
namespace frontend {
namespace gguf {

class GgufGraphContext;

// Lookup of a model's weights by GGUF tensor name, as a llama.cpp model file addresses them.
//
// A lookup EMITS the weight into the graph (as the GGML_OP_NONE leaf the translators expect) and
// returns a handle to it. Emission is idempotent: a weight referenced by several ops, or read
// repeatedly by a ported layer loop, is emitted exactly once.
//
// A tensor the file does not contain yields an EMPTY GgufValue rather than an error, because
// "absent" is how GGUF encodes structure -- no `blk.0.attn_q_norm.weight` means the architecture
// has no QK-norm. A builder that requires a tensor says so itself, via require().
class GGUF_FRONTEND_API GgufTensors {
public:
    explicit GgufTensors(GgufGraphContext& ctx) : m_ctx(&ctx) {}

    // Weight by full GGUF name (e.g. "token_embd.weight", "blk.3.attn_q.bias"). Empty if absent.
    GgufValue operator()(const std::string& gguf_name) const;

    // As operator(), but fails with a diagnostic naming the tensor and the architecture when the
    // file does not have it. Use for a tensor whose absence means the file does not match the
    // architecture being built, so the error names the real problem instead of surfacing later as
    // a shape mismatch.
    GgufValue require(const std::string& gguf_name) const;

    // True when the file carries this tensor, WITHOUT emitting it. For a structure probe
    // ("does layer 0 have a QK-norm?") that must not add a leaf to the graph.
    bool has(const std::string& gguf_name) const;

    // Weight of layer `il` by suffix: layer(3, "attn_norm.weight") -> "blk.3.attn_norm.weight".
    GgufValue layer(int il, const std::string& suffix) const;

private:
    GgufGraphContext* m_ctx;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
