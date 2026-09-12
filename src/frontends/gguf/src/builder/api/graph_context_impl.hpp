// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Shared implementation for graph operations and emitting weight lookups.

#pragma once

#include <string>

#include "builder/api/metadata_store.hpp"
#include "builder/blocks/attention.hpp"
#include "builder/graph_emitter.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov {
namespace frontend {
namespace gguf {

struct GgufGraphContext::Impl {
    explicit Impl(const BuildContext& ctx)
        : build_ctx(ctx),
          emitter(ctx.weights->weights, ctx.weights->qtypes, ctx.arch, ctx.weights->translators) {}

    BuildContext build_ctx;
    std::unique_ptr<DecoderConfig> decoder;
    blocks::KvCachePlan kv;
    bool finished = false;
    void check_open() const {
        OPENVINO_ASSERT(!finished, "[GGUF] graph is already finished");
    }
    void check_layer(int layer) const {
        check_open();
        OPENVINO_ASSERT(decoder, "[GGUF] call configure_decoder before using decoder blocks");
        OPENVINO_ASSERT(layer >= 0 && layer < decoder->n_layer, "[GGUF] decoder layer is out of range");
    }
    const std::string& value_name(const GgufValue& value) const {
        OPENVINO_ASSERT(value, "[GGUF] builder received an empty value");
        OPENVINO_ASSERT(emitter.value(value.name()) == value.m_value,
                        "[GGUF] value '",
                        value.name(),
                        "' does not belong to this graph");
        return value.name();
    }
    GraphEmitter emitter;

    // Generate unique names for builder operations. Shared decoder blocks use layer prefixes.
    int seq = 0;
    std::string fresh(const std::string& op) {
        std::string name;
        do {
            name = op + "_" + std::to_string(seq++);
        } while (emitter.graph()->values->count(name) || emitter.has_weight(name));
        return name;
    }
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
