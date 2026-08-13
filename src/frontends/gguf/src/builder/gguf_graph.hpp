// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/frontend/gguf/decoder.hpp"

namespace ov {
namespace frontend {
namespace gguf {

// One operation node in the GGUF-built graph, expressed in the GGML op vocabulary
// ("GGML_OP_MUL_MAT", "GGML_OP_ROPE", ...). It mirrors exactly what the GgufDecoder
// interface exposes per node, so GgufBuilderDecoder is a thin accessor over it. The arch
// builder fills these by construction, so all per-op parameters are typed attributes (no
// raw gguf op_params layout).
struct GgufOp {
    std::string op_type;                   // e.g. "GGML_OP_MUL_MAT"
    std::string name;                      // unique node/op name
    std::vector<std::string> input_names;  // producer tensor names (weights / inputs / other nodes)
    std::string output_name;               // this node's output tensor name
    ov::PartialShape output_shape;
    ov::element::Type output_type = ov::element::dynamic;
    int op_case = 0;

    // Per-input shape/stride/type and view-offset, keyed by input name. Populated for the
    // inputs that translators query (shapes for MUL_MAT/RESHAPE, view offsets for VIEW).
    std::map<std::string, ov::PartialShape> input_shapes;
    std::map<std::string, std::vector<size_t>> input_strides;
    std::map<std::string, ov::element::Type> input_types;
    std::map<std::string, int64_t> input_view_offsets;

    // Typed scalar/struct op attributes consumed by translators via get_attribute<T>
    // (e.g. "eps", "scale", "bias", "max_bias", "swapped", "rope_config").
    std::map<std::string, ov::Any> attributes;
};

// The whole model as a flat, topologically-ordered list of GgufOp nodes plus the
// model-level I/O the decoder reports. Built by an architecture builder (e.g. qwen3) from
// a parsed GGUF file; consumed by GgufBuilderDecoder.
struct GgufGraph {
    std::vector<GgufOp> nodes;

    // Model inputs (Parameters) and extra inputs (e.g. attention_size; Parameter or Constant).
    // Same semantics as the corresponding GgufDecoder getters. Weights are not here: they are
    // emitted into `nodes` as GGML_OP_NONE leaves (see emit_weight_op).
    std::map<std::string, std::shared_ptr<ov::Node>> model_inputs;
    std::map<std::string, std::shared_ptr<ov::Node>> model_extra_inputs;
    std::vector<std::string> model_output_names;

    // Recurrent (overwritten, non-appending) states as {input name, output name} pairs; see
    // GgufDecoder::get_recurrent_states. Empty for every architecture without linear attention.
    std::vector<std::pair<std::string, std::string>> recurrent_states;

    bool has_rope = false;
    RopeConfig rope_config;

    // When true, each ROPE op builds its own sin/cos table from its per-op rope_config
    // (useful when different layers need different n_dims, e.g. gemma4 SWA vs global).
    // TranslateSession::add_rope_sin_cos skips the shared table when this is set.
    bool use_per_op_rope = false;

    // GGUF tokenizer metadata (the `tokenizer.*` keys), keyed by the sub-key after the last
    // dot (e.g. "model", "tokens", "merges", "scores", "token_type", "pre", "bos_token_id",
    // "chat_template"). Values are std::string / std::vector<std::string> / ov::Tensor,
    // mirroring the GGUF metadata variant. Attached to the model's (non-serializable) rt_info
    // by TranslateSession so a downstream consumer can build the tokenizer without re-reading
    // the .gguf. Empty if the file carries no tokenizer metadata.
    ov::AnyMap tokenizer_config;
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
