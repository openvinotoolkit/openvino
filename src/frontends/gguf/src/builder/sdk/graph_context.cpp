// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Implementation of the extension-facing graph-building vocabulary.
//
// This is a FACADE over GraphEmitter, not a second builder: it adds the two things a ported
// llama.cpp model file needs and the internal string-based API deliberately does not have --
// value handles that carry shape and type, and per-op output-shape inference -- and then emits
// exactly the same GGML-vocabulary nodes the in-tree builder emits. The in-tree blocks/ keep their
// own string-based API; nothing here is on their path.
//
// Ground truth for each block's expansion is llama.cpp's llm_graph_context (build_norm, build_ffn,
// build_attn) and, for the KV-cache store, the SET_ROWS/VIEW pattern in blocks/attention.cpp,
// which this must match node-for-node so a ported architecture converts to the same graph as the
// built-in path.

#include "openvino/frontend/gguf/builder/graph_context.hpp"

#include <cmath>

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

// Pad a shape to rank 4 with leading 1s. Shapes are carried in [ne3, ne2, ne1, ne0] order, so a
// rank-2 weight [rows, cols] becomes [1, 1, rows, cols] -- which is what ggml means by it too,
// every tensor being nominally 4D with trailing 1s in ne[] order. Shape inference below indexes
// positionally, so it needs every operand at the same rank.
ov::PartialShape to4d(const ov::PartialShape& s) {
    if (s.rank().is_dynamic()) {
        return s;
    }
    const size_t r = s.size();
    OPENVINO_ASSERT(r <= 4, "[GGUF] builder SDK: shape of rank ", r, " exceeds ggml's 4 dimensions");
    std::vector<ov::Dimension> dims(4 - r, ov::Dimension(1));
    dims.insert(dims.end(), s.begin(), s.end());
    return ov::PartialShape(dims);
}

}  // namespace

GgufGraphContext::GgufGraphContext(const BuildContext& ctx) : m_impl(std::make_unique<Impl>(ctx)) {
    OPENVINO_ASSERT(ctx.weights, "[GGUF] builder SDK: BuildContext has no weight table");
}

GgufGraphContext::~GgufGraphContext() = default;

const GgufMetadata& GgufGraphContext::metadata() const {
    return m_impl->build_ctx.metadata;
}

const GgufHparams& GgufGraphContext::hparams() const {
    return m_impl->hparams;
}

const std::string& GgufGraphContext::arch() const {
    return m_impl->build_ctx.arch;
}

GgufTensors GgufGraphContext::tensors() {
    return GgufTensors(*this);
}

int64_t GgufGraphContext::n_tokens() const {
    return Impl::T;
}

// ---- model inputs ----

GgufValue GgufGraphContext::add_input(const std::string& name, ov::element::Type type, const ov::PartialShape& shape) {
    auto& e = m_impl->emitter;
    if (!e.has_model_input(name)) {
        e.add_input(name, type, shape);
    }
    // The Parameter is dynamic in the token axis, but per-node metadata is static: record the
    // representative shape, with any dynamic dimension pinned to T.
    ov::PartialShape meta = shape;
    for (auto& d : meta) {
        if (d.is_dynamic()) {
            d = ov::Dimension(Impl::T);
        }
    }
    e.set_tensor_meta(name, meta, type);
    return GgufValue(name, meta, type);
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
    if (swa) {
        add_input("self_kq_mask_swa", f32, ov::PartialShape({1, 1, D, D}));
    }
    add_input("inp_kv_idx", i32, ov::PartialShape({1, 1, 1, D}));
}

// ---- ggml op vocabulary ----

namespace {

// ggml's elementwise ops give the result the FIRST operand's shape; the second is broadcast into
// it. Taking a per-axis maximum instead would be wrong for the commonest case in a transformer: a
// norm scale or bias is a ggml 1-D vector, which the parser stores as an [n, 1] tensor, so a
// per-axis max against a [1, 1, T, n] activation would invent an [1, 1, n, n] result.
ov::PartialShape binary_shape(const GgufValue& a, const GgufValue& b) {
    (void)b;
    return to4d(a.shape());
}

}  // namespace

GgufValue GgufGraphContext::add(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_ADD", {a, b}, binary_shape(a, b), a.type());
}

GgufValue GgufGraphContext::sub(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_SUB", {a, b}, binary_shape(a, b), a.type());
}

GgufValue GgufGraphContext::mul(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_MUL", {a, b}, binary_shape(a, b), a.type());
}

GgufValue GgufGraphContext::div(const GgufValue& a, const GgufValue& b) {
    return m_impl->emit("GGML_OP_DIV", {a, b}, binary_shape(a, b), a.type());
}

GgufValue GgufGraphContext::scale(const GgufValue& x, float factor) {
    return m_impl->emit("GGML_OP_SCALE", {x}, x.shape(), x.type(), 0, {{"scale", factor}, {"bias", 0.0f}});
}

GgufValue GgufGraphContext::mul_mat(const GgufValue& a, const GgufValue& b) {
    // ggml_mul_mat(a, b) -> ne = {a->ne[1], b->ne[1], b->ne[2], b->ne[3]}.
    // In [ne3, ne2, ne1, ne0] order that is [b0, b1, b2, a2] of the 4D-padded operand shapes.
    const auto sa = to4d(a.shape());
    const auto sb = to4d(b.shape());
    ov::PartialShape out({sb[0], sb[1], sb[2], sa[2]});
    return m_impl->emit("GGML_OP_MUL_MAT", {a, b}, out, f32);
}

GgufValue GgufGraphContext::get_rows(const GgufValue& a, const GgufValue& b) {
    // ggml_get_rows(a, b) -> ne = {a->ne[0], b->ne[0], b->ne[1], b->ne[2]}.
    const auto sa = to4d(a.shape());
    const auto sb = to4d(b.shape());
    ov::PartialShape out({sb[1], sb[2], sb[3], sa[3]});
    return m_impl->emit("GGML_OP_GET_ROWS", {a, b}, out, f32);
}

GgufValue GgufGraphContext::rms_norm(const GgufValue& x, float eps) {
    return m_impl->emit("GGML_OP_RMS_NORM", {x}, x.shape(), x.type(), 0, {{"eps", eps}});
}

GgufValue GgufGraphContext::norm(const GgufValue& x, float eps) {
    return m_impl->emit("GGML_OP_NORM", {x}, x.shape(), x.type(), 0, {{"eps", eps}});
}

GgufValue GgufGraphContext::soft_max(const GgufValue& x) {
    return m_impl->emit("GGML_OP_SOFT_MAX", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::silu(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_SILU", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::gelu(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_GELU", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::gelu_quick(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_GELU_QUICK", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::sigmoid(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_SIGMOID", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::tanh(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_TANH", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::relu(const GgufValue& x) {
    return m_impl->emit("GGML_UNARY_OP_RELU", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::sqr(const GgufValue& x) {
    return m_impl->emit("GGML_OP_SQR", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::sqrt(const GgufValue& x) {
    return m_impl->emit("GGML_OP_SQRT", {x}, x.shape(), x.type());
}

GgufValue GgufGraphContext::reshape(const GgufValue& x, const std::vector<int64_t>& ne) {
    OPENVINO_ASSERT(!ne.empty() && ne.size() <= 4, "[GGUF] reshape: expected 1..4 ggml dimensions");
    // `ne` arrives in ggml order (fastest-varying first); shapes are stored reversed.
    std::vector<int64_t> dims(ne.rbegin(), ne.rend());
    const ov::PartialShape out = to4d(ov::PartialShape(dims));

    // The RESHAPE translator has one case per layout change rather than a general reshape, so pick
    // the one this target expresses. The two that matter in a transformer are the head split and
    // its inverse, which are also the only two that must keep the token axis dynamic:
    //   3 ggml dims {head_size, n_head, n_tokens} -- split a projection into heads   (case 1)
    //   2 ggml dims {n_head*head_size, n_tokens}  -- merge the heads back            (case 2)
    // Anything else is a fully static reshape (case 7).
    int op_case = 7;
    if (ne.size() == 3) {
        op_case = 1;
    } else if (ne.size() == 2) {
        op_case = 2;
    }
    return m_impl->emit("GGML_OP_RESHAPE", {x}, out, x.type(), op_case);
}

GgufValue GgufGraphContext::cont(const GgufValue& x) {
    // ggml_cont makes a tensor memory-contiguous. A permute here already emitted a real Transpose,
    // so the result is contiguous by construction and case 1 is a pass-through -- which is exactly
    // what a ported `ggml_cont(ctx, ggml_permute(...))` means.
    return m_impl->emit("GGML_OP_CONT", {x}, x.shape(), x.type(), 1);
}

GgufValue GgufGraphContext::permute(const GgufValue& x, const std::vector<int64_t>& perm) {
    OPENVINO_ASSERT(perm.size() == 4, "[GGUF] permute: expected 4 axes");
    const auto s = to4d(x.shape());
    std::vector<ov::Dimension> dims;
    dims.reserve(4);
    for (auto axis : perm) {
        OPENVINO_ASSERT(axis >= 0 && axis < 4, "[GGUF] permute: axis out of range");
        dims.push_back(s[static_cast<size_t>(axis)]);
    }
    return m_impl->emit("GGML_OP_PERMUTE", {x}, ov::PartialShape(dims), x.type(), 1, {{"perm", perm}});
}

GgufValue GgufGraphContext::transpose(const GgufValue& x) {
    auto s = to4d(x.shape());
    ov::PartialShape out({s[0], s[1], s[3], s[2]});
    return m_impl->emit("GGML_OP_TRANSPOSE", {x}, out, x.type());
}

GgufValue GgufGraphContext::concat(const GgufValue& a, const GgufValue& b, int ggml_dim) {
    OPENVINO_ASSERT(ggml_dim >= 0 && ggml_dim < 4, "[GGUF] concat: ggml dimension must be 0..3");
    auto sa = to4d(a.shape());
    const auto sb = to4d(b.shape());
    const size_t axis = 3 - static_cast<size_t>(ggml_dim);
    std::vector<ov::Dimension> dims(sa.begin(), sa.end());
    if (sa[axis].is_static() && sb[axis].is_static()) {
        dims[axis] = ov::Dimension(sa[axis].get_length() + sb[axis].get_length());
    } else {
        dims[axis] = ov::Dimension::dynamic();
    }
    return m_impl->emit("GGML_OP_CONCAT", {a, b}, ov::PartialShape(dims), a.type(), ggml_dim);
}

GgufValue GgufGraphContext::rope_ext(const GgufValue& x,
                                     const GgufValue& positions,
                                     const GgufValue& freq_factors,
                                     const RopeConfig& cfg,
                                     int rope_op_case) {
    std::vector<GgufValue> inputs{x, positions};
    if (freq_factors) {
        inputs.push_back(freq_factors);
    }
    return m_impl->emit("GGML_OP_ROPE", inputs, x.shape(), x.type(), rope_op_case, {{"rope_config", cfg}});
}

GgufValue GgufGraphContext::raw_op(const std::string& op_type,
                                   const std::vector<GgufValue>& inputs,
                                   const ov::PartialShape& out_shape,
                                   ov::element::Type out_type,
                                   int op_case,
                                   const std::map<std::string, ov::Any>& attrs) {
    return m_impl->emit(op_type, inputs, out_shape, out_type, op_case, attrs);
}

// ---- llm_graph_context-style blocks ----

GgufValue GgufGraphContext::build_norm(const GgufValue& cur, const GgufValue& w, float eps) {
    auto out = rms_norm(cur, eps);
    // A NULL weight in llama.cpp means a plain normalization with no multiplicative term.
    if (w) {
        out = mul(out, w);
    }
    return out;
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

GgufValue GgufGraphContext::build_ffn(const GgufValue& cur,
                                      const GgufValue& up,
                                      const GgufValue& up_b,
                                      const GgufValue& gate,
                                      const GgufValue& gate_b,
                                      const GgufValue& down,
                                      const GgufValue& down_b,
                                      FfnOp op) {
    OPENVINO_ASSERT(up, "[GGUF] build_ffn: the up projection is required");
    OPENVINO_ASSERT(down, "[GGUF] build_ffn: the down projection is required");

    auto tmp = mul_mat(up, cur);
    if (up_b) {
        tmp = add(tmp, up_b);
    }

    GgufValue activated;
    if (gate) {
        // Gated (parallel) form: activation on the gate branch, multiplied into the up branch.
        auto g = mul_mat(gate, cur);
        if (gate_b) {
            g = add(g, gate_b);
        }
        switch (op) {
        case FfnOp::Silu:
            g = silu(g);
            break;
        case FfnOp::Gelu:
            g = gelu(g);
            break;
        case FfnOp::Relu:
            g = relu(g);
            break;
        }
        activated = mul(g, tmp);
    } else {
        switch (op) {
        case FfnOp::Silu:
            activated = silu(tmp);
            break;
        case FfnOp::Gelu:
            activated = gelu(tmp);
            break;
        case FfnOp::Relu:
            activated = relu(tmp);
            break;
        }
    }

    auto out = mul_mat(down, activated);
    if (down_b) {
        out = add(out, down_b);
    }
    return out;
}

GgufValue GgufGraphContext::build_attn(int il,
                                       const GgufValue& q,
                                       const GgufValue& k,
                                       const GgufValue& v,
                                       const GgufValue& wo,
                                       const GgufValue& wo_b,
                                       float kq_scale,
                                       const AttnOptions& opts) {
    auto& e = m_impl->emitter;
    auto& graph = *e.graph();
    const int64_t T = Impl::T;

    // Q/K/V arrive ggml-natural: [1, n_tokens, n_head(_kv), head_size].
    const auto kq = to4d(k.shape());
    const auto qs = to4d(q.shape());
    OPENVINO_ASSERT(kq[2].is_static() && kq[3].is_static() && qs[2].is_static(),
                    "[GGUF] build_attn: Q/K must have a static head count and head size; "
                    "reshape them to [n_tokens, n_head, head_size] first");
    const int64_t n_head_kv = kq[2].get_length();
    const int64_t head_size = kq[3].get_length();
    const int64_t n_head = qs[2].get_length();

    // ---- KV cache store ----
    // Per-layer f16 cache Parameters, written through by SET_ROWS. The frontend lowers SET_ROWS to
    // a stateless ScatterUpdate; a caller that registers the MakeStateful transformation extension
    // gets a real OpenVINO state instead. Both need the updated cache to be a model output.
    const std::string kc = "cache_k_l" + std::to_string(il);
    const std::string vc = "cache_v_l" + std::to_string(il);
    const ov::PartialShape cache_shape({1, D, n_head_kv, head_size});
    const ov::PartialShape cache_meta({1, T, n_head_kv, head_size});
    if (!e.has_model_input(kc)) {
        e.add_input(kc, f16, cache_shape);
        e.add_input(vc, f16, cache_shape);
    }
    e.set_tensor_meta(kc, cache_meta, f16);
    e.set_tensor_meta(vc, cache_meta, f16);

    OPENVINO_ASSERT(e.has_model_input("inp_kv_idx"),
                    "[GGUF] build_attn: the attention inputs are not declared; "
                    "call build_attn_inp_kv() before the layer loop");

    e.add_op("GGML_OP_SET_ROWS", kc, {k.name(), "inp_kv_idx", kc}, cache_meta, f16);
    e.add_op("GGML_OP_SET_ROWS", vc, {v.name(), "inp_kv_idx", vc}, cache_meta, f16);
    graph.model_output_names.push_back(kc);
    graph.model_output_names.push_back(vc);

    const GgufValue k_cache(kc, cache_meta, f16);
    const GgufValue v_cache(vc, cache_meta, f16);

    // ---- attention ----
    // Q/K/V stay ggml-natural here: the permute to [1, n_head, n_tokens, head_size] happens inside
    // the FLASH_ATTN translator, AFTER the GQA broadcast of K/V. That order (concat -> tile ->
    // single transpose -> SDPA) is what lets the CPU plugin fuse the whole thing into
    // ScaledDotProductAttentionWithKVCache, so op_case 100 (the builder layout) matters.
    std::vector<GgufValue> attn_in{q, k_cache, v_cache};
    const std::string mask = opts.mask;
    OPENVINO_ASSERT(e.has_model_input(mask),
                    "[GGUF] build_attn: mask input '",
                    mask,
                    "' is not declared; pass swa=true to build_attn_inp_kv() for a sliding-window mask");
    attn_in.emplace_back(mask, e.shape_of_tensor(mask), e.type_of_tensor(mask));
    if (opts.sinks) {
        attn_in.push_back(opts.sinks);
    }

    std::map<std::string, ov::Any> attrs{{"scale", kq_scale}};
    if (opts.kq_soft_cap != 0.0f) {
        attrs["kq_soft_cap"] = opts.kq_soft_cap;
    }
    auto attn = m_impl->emit("GGML_OP_FLASH_ATTN_EXT",
                             attn_in,
                             ov::PartialShape({1, T, n_head, head_size}),
                             f32,
                             100,
                             std::move(attrs));

    // Merge the heads back: [1, 1, n_tokens, n_head*head_size].
    auto merged = m_impl->emit("GGML_OP_RESHAPE", {attn}, ov::PartialShape({1, 1, T, n_head * head_size}), f32, 2);

    if (!wo) {
        return merged;
    }
    auto out = mul_mat(wo, merged);
    if (wo_b) {
        out = add(out, wo_b);
    }
    return out;
}

GgufValue GgufGraphContext::build_lora_mm(const GgufValue& w, const GgufValue& cur) {
    return mul_mat(w, cur);
}

void GgufGraphContext::cb(const GgufValue&, const std::string&, int) {
    // Names are generated when a node is emitted, and the graph is consumed by op type and
    // topology rather than by name, so this is a no-op. It exists so the cb() calls a ported
    // model file is littered with compile untouched.
}

void GgufGraphContext::set_output(const GgufValue& logits) {
    OPENVINO_ASSERT(logits, "[GGUF] set_output: the output value is empty");
    m_impl->emitter.graph()->model_output_names.push_back(logits.name());
}

std::shared_ptr<GgufGraph> GgufGraphContext::finish() {
    return m_impl->emitter.graph();
}

}  // namespace gguf
}  // namespace frontend
}  // namespace ov
