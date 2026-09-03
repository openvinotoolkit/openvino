// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "openvino/frontend/gguf/builder/metadata.hpp"
#include "openvino/frontend/gguf/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// The built graph. Opaque to an extension: a builder produces one through GgufGraphContext and
// returns it, and never needs to inspect it. Defined in the frontend (builder/gguf_graph.hpp).
struct GgufGraph;

namespace detail {
struct WeightStore;
}

// Everything a model builder is handed about the file it is building from.
//
// It is a VIEW: the metadata and weight tables belong to the parser and stay alive for the whole
// build. A builder holds this, not copies of it -- the weight table maps every tensor in the file,
// which for a real checkpoint is the whole model.
struct GGUF_FRONTEND_API BuildContext {
    // The file's KV metadata.
    GgufMetadata metadata;

    // `general.architecture` as the file spells it.
    std::string arch;

    // The parser's tensor tables (weights and their quantization types), opaque here: they are
    // reached through GgufTensors, which knows how to emit a weight as the graph leaf the
    // translators expect.
    detail::WeightStore* weights = nullptr;
};

// Base class for a whole-model graph builder: one subclass per MODEL FAMILY.
//
// A family is a distinct graph SHAPE with its own inputs, its own block vocabulary and its own
// notion of a "layer": the causal decoder stack is one; a vision/mmproj encoder or an audio
// encoder is another. Within a family, individual architectures are data -- detected from the GGUF
// tensor table and metadata -- rather than code.
//
// This mirrors llama.cpp's split between llm_graph_context (LLMs) and clip_graph (mmproj), where
// each family has its own base and its own build_norm/build_ffn/build_attn vocabulary, rather than
// one builder growing flags for structurally unrelated models.
//
// An EXTENSION supplies a family by subclassing this and registering a factory through
// ArchitectureExtension: that is how a non-decoder architecture is added without touching the
// frontend. See docs/porting_a_llama_cpp_model.md.
class GGUF_FRONTEND_API ModelBuilder {
public:
    virtual ~ModelBuilder();

    // Emit the whole model and return the finished graph.
    virtual std::shared_ptr<GgufGraph> build() = 0;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
