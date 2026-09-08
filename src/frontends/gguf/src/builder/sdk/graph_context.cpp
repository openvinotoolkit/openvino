// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Generic graph operations and adapters to the shared native decoder blocks.

#include "openvino/frontend/gguf/builder/graph_context.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

#include "builder/blocks/common.hpp"
#include "builder/blocks/ffn.hpp"
#include "builder/blocks/gated_delta_net.hpp"
#include "builder/sdk/graph_context_impl.hpp"

namespace ov {
namespace frontend {
namespace gguf {

using ov::element::f16;
using ov::element::f32;
using ov::element::i32;
using ov::element::i64;

namespace {

// Dynamic extent, for model-input Parameters.
constexpr int64_t D = -1;

}  // namespace

GgufGraphContext::GgufGraphContext(const BuildContext& ctx) {
    OPENVINO_ASSERT(ctx.weights, "[GGUF] builder SDK: BuildContext has no weight table");
    m_impl = std::make_unique<Impl>(ctx);
}

GgufGraphContext::~GgufGraphContext() = default;

const GgufMetadata& GgufGraphContext::metadata() const {
    return m_impl->build_ctx.metadata;
}

DecoderDimensions GgufGraphContext::configure_decoder(RopeMode rope, const DecoderOptions& options) {
    m_impl->check_open();
    OPENVINO_ASSERT(!m_impl->decoder, "[GGUF] decoder is already configured");
    auto normalized = decoder_config_from_meta(detail::MetadataAccess::get(metadata()).map);
    auto config = std::make_unique<DecoderConfig>(normalized, m_impl->build_ctx.weights->weights, rope, options);
    auto& graph = *m_impl->emitter.graph();
    graph.has_rope = true;
    graph.rope_config = config->rope_config;
    graph.use_per_op_rope = config->use_per_op_rope;
    graph.swa_window_size = config->swa_window_size;
    const DecoderDimensions dimensions{config->n_layer, config->n_embd, config->rms_eps};
    m_impl->kv = blocks::KvCachePlan::build(*config);
    m_impl->decoder = std::move(config);
    return dimensions;
}

DecoderLayerParameters GgufGraphContext::decoder_layer_parameters(int layer) const {
    m_impl->check_layer(layer);
    const auto& cfg = *m_impl->decoder;
    return {cfg.n_head,
            cfg.layer_n_head_kv(layer),
            cfg.layer_head_size(layer),
            cfg.layer_kq_scale(layer),
            cfg.layer_rope_config(layer),
            cfg.layer_is_swa(layer),
            cfg.is_recurrent_layer(layer)};
}

GgufValue GgufGraphContext::decoder_attention(int layer, const GgufValue& input) {
    m_impl->check_layer(layer);
    OPENVINO_ASSERT(input, "[GGUF] decoder attention requires a normalized input");
    const auto& cfg = *m_impl->decoder;
    auto& emitter = m_impl->emitter;
    const auto output = cfg.is_recurrent_layer(layer)
                            ? blocks::gated_delta_net(emitter, cfg, layer, input.name())
                            : blocks::attention(emitter, cfg, m_impl->kv, layer, input.name());
    return GgufValue(output, emitter.value(output));
}

GgufValue GgufGraphContext::decoder_ffn(int layer, const GgufValue& input) {
    m_impl->check_layer(layer);
    OPENVINO_ASSERT(input, "[GGUF] decoder FFN requires a normalized input");
    const auto& cfg = *m_impl->decoder;
    auto& emitter = m_impl->emitter;
    const auto prefix = "blk." + std::to_string(layer) + ".";
    const auto output = cfg.layer_is_moe(layer) ? blocks::moe_ffn(emitter, cfg, prefix, input.name())
                        : cfg.is_geglu          ? blocks::geglu_ffn(emitter, cfg, prefix, input.name())
                                                : blocks::dense_ffn(emitter, cfg, prefix, input.name());
    return GgufValue(output, emitter.value(output));
}

const std::string& GgufGraphContext::arch() const {
    return m_impl->build_ctx.arch;
}

GgufTensors GgufGraphContext::tensors() {
    return GgufTensors(*this);
}

// ---- model inputs ----

GgufValue GgufGraphContext::add_input(const std::string& name, ov::element::Type type, const ov::PartialShape& shape) {
    m_impl->check_open();
    OPENVINO_ASSERT(shape.rank().is_static() && shape.size() <= 4, "[GGUF] input rank must be known and at most four");
    auto& e = m_impl->emitter;
    if (e.has_model_input(name)) {
        const auto& previous = e.graph()->model_inputs.at(name);
        OPENVINO_ASSERT(previous->get_output_partial_shape(0) == shape && previous->get_output_element_type(0) == type,
                        "[GGUF] conflicting declarations for input ",
                        name);
    }
    if (!e.has_model_input(name)) {
        e.add_input(name, type, shape);
    }
    return GgufValue(name, e.value(name));
}

GgufValue GgufGraphContext::build_inp_embd(const GgufValue& tok_embd) {
    OPENVINO_ASSERT(tok_embd, "[GGUF] build_inp_embd: the token embedding weight is missing");
    auto tokens = add_input("inp_tokens", i32, ov::PartialShape({1, 1, 1, D}));
    return get_rows(tok_embd, tokens);
}

GgufValue GgufGraphContext::build_inp_pos() {
    return add_input("inp_pos", i32, ov::PartialShape({1, 1, 1, D}));
}

GgufValue GgufGraphContext::build_inp_out_ids() {
    return add_input("inp_out_ids", i32, ov::PartialShape({1, 1, 1, D}));
}

void GgufGraphContext::build_attn_inp_kv(bool swa) {
    add_input("self_kq_mask", f32, ov::PartialShape({1, 1, D, D}));
    if (swa || (m_impl->decoder && m_impl->decoder->has_swa)) {
        add_input("self_kq_mask_swa", f32, ov::PartialShape({1, 1, D, D}));
    }
    add_input("inp_kv_idx", i32, ov::PartialShape({1, 1, 1, D}));
    add_input("token_len_per_seq", i64, ov::PartialShape({1}));
}

// ---- ggml op vocabulary ----

GgufValue GgufGraphContext::add(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_ADD", {a, b}, a.type());
}

GgufValue GgufGraphContext::sub(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_SUB", {a, b}, a.type());
}

GgufValue GgufGraphContext::mul(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_MUL", {a, b}, a.type());
}

GgufValue GgufGraphContext::div(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_DIV", {a, b}, a.type());
}

GgufValue GgufGraphContext::scale(const GgufValue& x, float factor) {
    return m_impl->emit("GGML_OP_SCALE", {x}, x.type(), 0, {{"scale", factor}, {"bias", 0.0f}});
}

GgufValue GgufGraphContext::mul_mat(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_MUL_MAT", {a, b}, f32);
}

GgufValue GgufGraphContext::get_rows(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_GET_ROWS", {a, b}, f32);
}

GgufValue GgufGraphContext::rms_norm(const GgufValue& x, float eps) {
    return m_impl->emit("GGML_OP_RMS_NORM", {x}, x.type(), 0, {{"eps", eps}});
}

GgufValue GgufGraphContext::norm(const GgufValue& x, float eps) {
    return m_impl->emit("GGML_OP_NORM", {x}, x.type(), 0, {{"eps", eps}});
}

GgufValue GgufGraphContext::soft_max(const GgufValue& x) {
    return m_impl->emit("GGML_OP_SOFT_MAX", {x}, x.type());
}

GgufValue GgufGraphContext::silu(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_SILU", {x}, x.type());
}

GgufValue GgufGraphContext::gelu(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_GELU", {x}, x.type());
}

GgufValue GgufGraphContext::gelu_quick(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_GELU_QUICK", {x}, x.type());
}

GgufValue GgufGraphContext::sigmoid(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_SIGMOID", {x}, x.type());
}

GgufValue GgufGraphContext::tanh(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_TANH", {x}, x.type());
}

GgufValue GgufGraphContext::relu(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_RELU", {x}, x.type());
}

GgufValue GgufGraphContext::sqr(const GgufValue& x) {
    return m_impl->emit("GGML_OP_SQR", {x}, x.type());
}

GgufValue GgufGraphContext::sqrt(const GgufValue& x) {
    return m_impl->emit("GGML_OP_SQRT", {x}, x.type());
}

GgufValue GgufGraphContext::reshape(const GgufValue& x, const std::vector<int64_t>& ne) {
    OPENVINO_ASSERT(!ne.empty() && ne.size() <= 4, "[GGUF] reshape: expected 1..4 ggml dimensions");
    // `ne` arrives in ggml order (fastest-varying first); shapes are stored reversed.
    std::vector<int64_t> dims(ne.rbegin(), ne.rend());

    OPENVINO_ASSERT(std::count(ne.begin(), ne.end(), -1) <= 1 && std::all_of(ne.begin(),
                                                                             ne.end(),
                                                                             [](int64_t d) {
                                                                                 return d > 0 || d == -1;
                                                                             }),
                    "[GGUF] reshape dimensions must be positive, with at most one inferred dimension");
    dims.insert(dims.begin(), 4 - dims.size(), 1);
    return m_impl->emit("GGML_OP_RESHAPE", {x}, x.type(), 6, {{"reshape_target", dims}});
}

GgufValue GgufGraphContext::split_heads(const GgufValue& x, int64_t heads, int64_t head_size) {
    OPENVINO_ASSERT(heads > 0 && head_size > 0, "[GGUF] head dimensions must be positive");
    return m_impl->emit("GGML_OP_RESHAPE",
                        {x},
                        x.type(),
                        6,
                        {{"reshape_target", std::vector<int64_t>{0, -1, heads, head_size}}, {"special_zero", true}});
}

GgufValue GgufGraphContext::merge_heads(const GgufValue& x) {
    return m_impl->emit("GGML_OP_RESHAPE", {x}, x.type(), 2, {{"merge_heads", true}});
}

GgufValue GgufGraphContext::cont(const GgufValue& x) {
    // OpenVINO handles tensor layout; GGML CONT case 1 is a pass-through.
    return m_impl->emit("GGML_OP_CONT", {x}, x.type(), 1);
}

GgufValue GgufGraphContext::permute(const GgufValue& x, const std::vector<int64_t>& perm) {
    OPENVINO_ASSERT(perm.size() == 4, "[GGUF] permute: expected 4 axes");
    OPENVINO_ASSERT(std::set<int64_t>(perm.begin(), perm.end()).size() == 4, "[GGUF] permute axes must be unique");
    for (auto axis : perm) {
        OPENVINO_ASSERT(axis >= 0 && axis < 4, "[GGUF] permute: axis out of range");
    }
    return m_impl->emit("GGML_OP_PERMUTE", {x}, x.type(), 1, {{"perm", perm}});
}

GgufValue GgufGraphContext::transpose(const GgufValue& x) {
    return m_impl->emit("GGML_OP_TRANSPOSE", {x}, x.type());
}

GgufValue GgufGraphContext::concat(const GgufValue& a, const GgufValue& b, int ggml_dim) {
    OPENVINO_ASSERT(ggml_dim >= 0 && ggml_dim < 4, "[GGUF] concat: ggml dimension must be 0..3");
    return m_impl->emit("GGML_OP_CONCAT", {a, b}, a.type(), ggml_dim, {{"concat_axis", ggml_dim}});
}

GgufValue GgufGraphContext::rope_ext(const GgufValue& x,
                                     const GgufValue& positions,
                                     const GgufValue& freq_factors,
                                     const RopeConfig& cfg,
                                     int rope_op_case) {
    m_impl->check_open();
    auto& graph = *m_impl->emitter.graph();
    graph.has_rope = true;
    graph.use_per_op_rope = true;
    graph.rope_config.is_imrope |= ((rope_op_case >> 16) == 2);
    std::vector<GgufValue> inputs{x, positions};
    if (freq_factors) {
        inputs.push_back(freq_factors);
    }
    return m_impl->emit("GGML_OP_ROPE", inputs, x.type(), rope_op_case, {{"rope_config", cfg}});
}

GgufValue GgufGraphContext::raw_op(const std::string& op_type,
                                   const std::vector<GgufValue>& inputs,
                                   ov::element::Type out_type,
                                   int op_case,
                                   const std::map<std::string, ov::Any>& attrs) {
    return m_impl->emit(op_type, inputs, out_type, op_case, attrs);
}

// Normalization blocks

GgufValue GgufGraphContext::build_norm(const GgufValue& cur, const GgufValue& w, float eps) {
    m_impl->check_open();
    OPENVINO_ASSERT(cur, "[GGUF] normalization requires an input");
    if (!w)
        return rms_norm(cur, eps);
    const auto out = blocks::rms_norm(m_impl->emitter, cur.name(), w.name(), m_impl->fresh("norm"), eps);
    return GgufValue(out, m_impl->emitter.value(out));
}

GgufValue GgufGraphContext::build_norm_ln(const GgufValue& cur, const GgufValue& w, const GgufValue& b, float eps) {
    auto out = norm(cur, eps);
    if (w) {
        out = mul(out, w);
    }
    if (b) {
        out = add(out, b);
    }
    return out;
}

void GgufGraphContext::set_output(const GgufValue& logits) {
    m_impl->check_open();
    OPENVINO_ASSERT(logits, "[GGUF] set_output: the output value is empty");
    m_impl->emitter.graph()->model_output_names.push_back(logits.name());
}

void GgufGraphContext::set_sliding_window(int64_t tokens) {
    m_impl->check_open();
    OPENVINO_ASSERT(tokens > 0 && tokens <= std::numeric_limits<int>::max(),
                    "[GGUF] sliding window must be a positive int");
    m_impl->emitter.graph()->swa_window_size = static_cast<int>(tokens);
}

void GgufGraphContext::add_recurrent_state(const GgufValue& input, const GgufValue& update) {
    m_impl->check_open();
    OPENVINO_ASSERT(input && update && m_impl->emitter.has_model_input(input.name()),
                    "[GGUF] recurrent state must refer to a model input and an update");
    OPENVINO_ASSERT(input.type() == update.type() && input.shape().compatible(update.shape()),
                    "[GGUF] recurrent state input and update must have compatible shapes and equal types");
    auto& graph = *m_impl->emitter.graph();
    for (const auto& pair : graph.recurrent_states) {
        OPENVINO_ASSERT(pair.first != input.name() && pair.second != update.name(),
                        "[GGUF] recurrent state is already registered");
    }
    graph.recurrent_states.emplace_back(input.name(), update.name());
    if (std::find(graph.model_output_names.begin(), graph.model_output_names.end(), update.name()) ==
        graph.model_output_names.end())
        set_output(update);
}

std::shared_ptr<GgufGraph> GgufGraphContext::finish() {
    m_impl->check_open();
    const auto graph = m_impl->emitter.graph();
    OPENVINO_ASSERT(!graph->model_output_names.empty(), "[GGUF] graph has no outputs");
    std::set<std::string> available;
    for (const auto& entry : graph->model_inputs)
        available.insert(entry.first);
    for (const auto& entry : graph->model_extra_inputs)
        available.insert(entry.first);
    for (const auto& node : graph->nodes) {
        for (const auto& input : node.input_names) {
            OPENVINO_ASSERT(available.count(input), "[GGUF] node '", node.name, "' uses unknown value '", input, "'");
        }
        available.insert(node.output_name);
    }
    std::set<std::string> outputs;
    for (const auto& output : graph->model_output_names) {
        OPENVINO_ASSERT(available.count(output), "[GGUF] unknown output '", output, "'");
        OPENVINO_ASSERT(outputs.insert(output).second, "[GGUF] duplicate output '", output, "'");
    }
    m_impl->finished = true;
    return graph;
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
