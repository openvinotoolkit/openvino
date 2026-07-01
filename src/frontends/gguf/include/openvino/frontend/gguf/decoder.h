// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <map>
#include <openvino/core/node.hpp>
#include <openvino/frontend/decoder.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace gguf {

// Typed RoPE configuration, replacing the raw gguf `op_params`/`rope_params` int32 array
// the translators used to dereference by byte offset. A decoder is responsible for
// producing this (the llama.cpp cgraph decoder by parsing ggml's layout, a future
// gguf-file decoder by construction), so the op translators never need to know ggml's
// memory layout.
struct RopeConfig {
    int n_dims = 0;  // 0 means the model uses no RoPE (replaces a separate has_rope() query)
    int n_ctx_orig = 0;
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 0.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
    // When true, each ROPE op builds its own sin/cos from its per-op config (e.g. gemma4, where
    // SWA and global layers use different n_dims), so the shared rope_cos/rope_sin table that
    // TranslateSession::preprocess would otherwise pre-build is skipped.
    bool per_op = false;
};

// Decoder interface consumed by the gguf frontend translators.
//
// Following the established OpenVINO frontend pattern (cf. the PyTorch frontend), a GgufDecoder
// is node-scoped: visit_subgraph hands the visitor a fresh decoder bound to a single node, and
// every per-node accessor (get_attribute, get_input_*, get_output_*, get_op_*) refers to that
// node -- no node index is threaded through. The same object type, queried at model scope (the
// instance returned by InputModel::get_decoder and iterated by visit_subgraph), answers the
// model-level questions (get_model_inputs, get_model_output_names, get_rope_config, ...).
//
// This is a typed, ggml-free interface: operation parameters are exposed through
// get_attribute(name) / get_input_view_offset / RopeConfig rather than as raw ggml `op_params`
// int32 arrays. A concrete decoder (e.g. the llama.cpp cgraph decoder) only has to translate
// ggml's layout into these typed accessors -- the op translators here never touch ggml memory.
class GgufDecoder : public DecoderBase {
public:
    // Per-node typed attribute access (the bound node). The op translators use this to read
    // scalar operation parameters (e.g. "eps", "scale", "bias", "swapped", "rope_config")
    // without dereferencing ggml's raw op_params layout.
    ov::Any get_attribute(const std::string& name) const override = 0;

    virtual PartialShape get_input_shape(const std::string& name) const = 0;

    virtual std::vector<size_t> get_input_stride(const std::string& name) const = 0;

    // Byte offset of an input that is a gguf VIEW into a larger tensor (0 when the input
    // is not a view). Replaces the raw `get_input_op_params(...)[0]` read: a view's start
    // offset is a single semantic scalar the decoder knows (by parsing ggml, or by
    // construction), so translators never touch ggml's op_params layout. Translators
    // convert bytes -> elements using get_input_stride as needed.
    virtual int64_t get_input_view_offset(const std::string& name) const = 0;

    virtual element::Type get_input_type(const std::string& name) const = 0;

    size_t get_input_size() const override = 0;

    void get_input_node(size_t input_port_idx,
                        std::string& producer_name,
                        std::string& producer_output_port_name,
                        size_t& producer_output_port_index) const override = 0;

    virtual std::vector<std::string> get_input_names() const = 0;

    virtual PartialShape get_output_shape() const = 0;

    virtual element::Type get_output_type() const = 0;

    virtual std::vector<std::string> get_output_names() const = 0;

    const std::string& get_op_type() const override = 0;

    const std::string& get_op_name() const override = 0;

    virtual void visit_subgraph(std::function<void(std::shared_ptr<GgufDecoder>)> node_visitor) const = 0;

    virtual int get_op_case() const = 0;

    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_inputs() const = 0;
    virtual const std::map<std::string, std::shared_ptr<ov::Node>>& get_model_extra_inputs() const = 0;
    virtual std::vector<std::string> get_model_output_names() const = 0;

    // NOTE: there is no get_model_weights(). A GGUF weight is surfaced as a regular node in
    // visit_subgraph with the genuine ggml leaf op type "GGML_OP_NONE": the decoder marks it as a
    // weight by exposing the raw weight bytes via get_attribute<ov::Tensor>("data"), the ggml
    // quant type name via get_attribute<std::string>("quant_type") (e.g. "Q4_K", "F16") and the
    // logical [rows, cols] shape via get_output_shape(). The frontend's translate_weight does the
    // dequant / repacking / requantization, so the decoder never builds OV nodes itself. (Model
    // inputs are also GGML_OP_NONE leaves, but they are returned via get_model_inputs() and
    // resolved to Parameters before the walk, so they carry no "data".)

    // RoPE configuration, exposed through get_attribute<RopeConfig>("rope_config"):
    //   - at model scope, used by TranslateSession::preprocess to pre-build the shared rope
    //     sin/cos table (skipped when RopeConfig::n_dims == 0, i.e. no RoPE, or per_op == true);
    //   - at node scope, the ROPE translator reads the same key for the op's own config.
};

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
