// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>

#include "builder/gguf_graph.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// Base class for a whole-model graph builder: one subclass per MODEL FAMILY, not per architecture.
//
// A family is a distinct graph SHAPE with its own inputs, its own block vocabulary and its own
// notion of a "layer": the causal decoder stack (DecoderBuilder, arch/decoder_builder.hpp) is one;
// a vision/mmproj encoder or an audio encoder would each be another. Within a family, individual
// architectures are data, not code -- they are detected from the GGUF tensor table and metadata
// (see DecoderConfig) and enabled by name in arch_registry.cpp.
//
// This mirrors llama.cpp's split between llm_graph_context (LLMs) and clip_graph (mmproj), where
// each family has its own base and its own build_norm/build_ffn/build_attn vocabulary, rather than
// one builder growing flags for structurally unrelated models.
//
// To add a family: subclass this, implement build(), and dispatch to it from
// build_ggml_graph_from_gguf() on the detected ModelKind. Nothing in the existing families needs
// to change, and the arch-agnostic pieces (GraphEmitter, blocks/common) are reused as-is.
class ModelBuilder {
public:
    virtual ~ModelBuilder() = default;

    // Emit the whole model and return the finished graph.
    virtual std::shared_ptr<GgufGraph> build() = 0;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
