// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Definitions of the opaque holders the extension-facing dev_api types point at.
//
// The dev_api headers deliberately do not name the parser's containers: GGUFMetaData is a
// std::variant whose alternatives are an implementation detail, and the weight table is keyed and
// typed the way the quant reader happens to produce it. Both would otherwise become part of the
// contract an out-of-tree extension compiles against. These structs give the headers something
// opaque to hold instead.

#pragma once

#include <map>
#include <string>
#include <unordered_map>

#include "openvino/runtime/tensor.hpp"
#include "quant/gguf.hpp"

namespace ov {
namespace frontend {
namespace gguf {

class GraphEmitter;

namespace detail {

// The parsed KV metadata, behind GgufMetadata.
struct MetadataStore {
    const std::unordered_map<std::string, GGUFMetaData>& map;
};

// The normalized decoder metadata DecoderConfig is built from, behind detail::DecoderMeta.
struct DecoderMeta {
    const std::map<std::string, GGUFMetaData>& config;
};

// The parser's tensor tables plus the emitter that turns a weight into a graph leaf, behind
// GgufTensors. The emitter is what makes a weight lookup able to emit; the tables are what make it
// able to answer "does this file have that tensor" without emitting.
struct WeightStore {
    std::unordered_map<std::string, ov::Tensor>& weights;
    std::unordered_map<std::string, GgufTensorType>& qtypes;
};

}  // namespace detail
}  // namespace gguf
}  // namespace frontend
}  // namespace ov
