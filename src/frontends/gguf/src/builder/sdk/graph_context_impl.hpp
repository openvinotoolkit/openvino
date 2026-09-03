// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Definition of GgufGraphContext::Impl.
//
// Split out of graph_context.cpp because GgufTensors also needs it: a weight lookup emits into the
// same GraphEmitter the context wraps, and the two live in different translation units.

#pragma once

#include <map>
#include <string>
#include <vector>

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
          hparams(ctx.metadata, ctx.arch),
          emitter(ctx.weights->weights, ctx.weights->qtypes, ctx.arch) {}

    BuildContext build_ctx;
    GgufHparams hparams;
    GraphEmitter emitter;

    // Per-node output shapes are static, at a representative token length; see
    // GgufGraphContext::n_tokens().
    static constexpr int64_t T = 1;

    // Op names must be unique. A ported model file names values through cb() at best, and often
    // not at all, so names are generated here and cb() only decorates them.
    int seq = 0;
    std::string fresh(const std::string& op) {
        return op + "_" + std::to_string(seq++);
    }

    // Emit `op_type` and wrap the result. Every op wrapper funnels through this.
    GgufValue emit(const std::string& op_type,
                   const std::vector<GgufValue>& inputs,
                   const ov::PartialShape& out_shape,
                   ov::element::Type out_type,
                   int op_case = 0,
                   std::map<std::string, ov::Any> attrs = {}) {
        std::vector<std::string> in_names;
        in_names.reserve(inputs.size());
        for (const auto& v : inputs) {
            OPENVINO_ASSERT(v, "[GGUF] builder SDK: op '", op_type, "' was given an empty input value");
            in_names.push_back(v.name());
        }
        const auto name = fresh(op_type);
        emitter.add_op(op_type, name, in_names, out_shape, out_type, op_case, std::move(attrs));
        return GgufValue(name, out_shape, out_type);
    }
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
