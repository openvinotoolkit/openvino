// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Shared implementation for graph operations and emitting weight lookups.

#pragma once

#include <map>
#include <string>
#include <vector>

#include "builder/blocks/attention.hpp"
#include "builder/graph_emitter.hpp"
#include "builder/sdk/metadata_store.hpp"
#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov {
namespace frontend {
namespace gguf {

struct GgufGraphContext::Impl {
    explicit Impl(const BuildContext& ctx)
        : build_ctx(ctx),
          emitter(ctx.weights->weights, ctx.weights->qtypes, ctx.arch) {}

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
    GraphEmitter emitter;

    // Representative token length for static translator metadata.
    static constexpr int64_t T = 1;

    // Generate unique names for SDK operations. Shared decoder blocks use layer prefixes.
    int seq = 0;
    std::string fresh(const std::string& op) {
        return op + "_" + std::to_string(seq++);
    }

    GgufValue emit(const std::string& op_type,
                   const std::vector<GgufValue>& inputs,
                   const ov::PartialShape& out_shape,
                   ov::element::Type out_type,
                   int op_case = 0,
                   std::map<std::string, ov::Any> attrs = {}) {
        check_open();
        std::vector<std::string> in_names;
        in_names.reserve(inputs.size());
        for (const auto& v : inputs) {
            OPENVINO_ASSERT(v, "[GGUF] builder SDK: op '", op_type, "' was given an empty input value");
            in_names.push_back(v.name());
        }
        const auto name = fresh(op_type);
        auto metadata_shape = out_shape;
        OPENVINO_ASSERT(metadata_shape.rank().is_static(), "[GGUF] SDK values require a known rank");
        for (auto& dim : metadata_shape)
            if (dim.is_dynamic())
                dim = T;
        emitter.add_op(op_type, name, in_names, metadata_shape, out_type, op_case, std::move(attrs));
        return GgufValue(name, out_shape, out_type);
    }
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
