// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <string>

#include "openvino/frontend/gguf/builder/value.hpp"
#include "openvino/frontend/gguf/visibility.hpp"

namespace ov::frontend::gguf {

class GgufGraphContext;

// Weight lookup by GGUF name. Each present weight is emitted once as a GGML_OP_NONE leaf.
class GGUF_FRONTEND_API GgufTensors {
public:
    explicit GgufTensors(GgufGraphContext& ctx) : m_ctx(&ctx) {}

    // Weight by full GGUF name (e.g. "token_embd.weight", "blk.3.attn_q.bias"). Empty if absent.
    GgufValue operator()(const std::string& gguf_name) const;

    // Required weight; throws with the tensor and architecture names if absent.
    GgufValue require(const std::string& gguf_name) const;

    // Check presence without emitting a graph node.
    bool has(const std::string& gguf_name) const;

    // Original on-disk GGUF type IDs, before conversion or weight repacking.
    std::optional<uint32_t> source_type(const std::string& gguf_name) const;
    std::set<uint32_t> source_types() const;

    // Weight of layer `il` by suffix: layer(3, "attn_norm.weight") -> "blk.3.attn_norm.weight".
    GgufValue layer(int il, const std::string& suffix) const;

private:
    GgufGraphContext* m_ctx;
};

}  // namespace ov::frontend::gguf
