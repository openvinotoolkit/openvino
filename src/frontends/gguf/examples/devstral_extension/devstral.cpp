// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "devstral.hpp"

#include <cmath>
#include <limits>

#include "openvino/core/except.hpp"
#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace example {
namespace {
using namespace ov::frontend::gguf;

GgufValue reshape(GgufGraphContext& graph, const GgufValue& x, const std::vector<int64_t>& shape) {
    return graph.node("GGML_OP_RESHAPE", {x}, 6, {{"reshape_target", shape}, {"special_zero", true}});
}

// This example owns the Devstral text graph and reads raw GGUF keys, bypassing DecoderConfig.
class DevstralBuilder : public ModelBuilder {
public:
    explicit DevstralBuilder(const BuildContext& context) : m_context(context) {}

    std::shared_ptr<GgufGraph> build() override {
        GgufGraphContext graph(m_context);
        const auto& metadata = graph.metadata();
        const auto prefix = graph.arch() + ".";
        const auto integer = [&](const std::string& key, int64_t fallback = 0) {
            const auto value = metadata.get_int(prefix + key).value_or(fallback);
            OPENVINO_ASSERT(value > 0 && value <= std::numeric_limits<int>::max(),
                            "[Devstral] expected a positive integer: ",
                            prefix,
                            key);
            return static_cast<int>(value);
        };
        const auto number = [&](const std::string& key, float fallback) {
            const auto value = metadata.get_float(prefix + key).value_or(fallback);
            OPENVINO_ASSERT(std::isfinite(value), "[Devstral] non-finite metadata: ", prefix, key);
            return static_cast<float>(value);
        };
        const auto layers = integer("block_count");
        const auto heads = integer("attention.head_count");
        const auto width = integer("embedding_length");
        const auto head_size = integer("attention.key_length", width / heads);
        const auto kv_heads = integer("attention.head_count_kv", heads);
        OPENVINO_ASSERT(heads % kv_heads == 0, "[Devstral] KV heads must divide query heads");
        OPENVINO_ASSERT(integer("attention.value_length", head_size) == head_size,
                        "[Devstral] requires equal key and value head sizes");
        const auto epsilon = number("attention.layer_norm_rms_epsilon", 1e-5f);
        const auto temperature = number("attention.temperature_scale", 0.0f);
        OPENVINO_ASSERT(epsilon > 0, "[Devstral] RMS epsilon must be positive");
        OPENVINO_ASSERT(metadata.get_int(prefix + "attention.sliding_window").value_or(0) == 0,
                        "[Devstral] example supports full attention only");

        RopeConfig rope;
        rope.n_dims = integer("rope.dimension_count", head_size);
        rope.n_ctx_orig = integer("rope.scaling.original_context_length", integer("context_length"));
        rope.freq_base = number("rope.freq_base", 10000.0f);
        const auto factor = number("rope.scaling.factor", 1.0f);
        OPENVINO_ASSERT(factor > 0 && rope.freq_base > 0, "[Devstral] RoPE scale and base must be positive");
        rope.freq_scale = 1.0f / factor;
        const auto scaling = metadata.get_str(prefix + "rope.scaling.type").value_or("none");
        OPENVINO_ASSERT(scaling == "none" || scaling == "linear" || scaling == "yarn",
                        "[Devstral] unsupported RoPE scaling: ",
                        scaling);
        rope.ext_factor = scaling == "yarn" ? 1.0f : 0.0f;
        rope.beta_fast = number("rope.scaling.yarn_beta_fast", 32.0f);
        rope.beta_slow = number("rope.scaling.yarn_beta_slow", 1.0f);
        rope.attn_factor = 1.0f;
        const auto log_multiplier = number("rope.scaling.yarn_log_multiplier", 0.0f);
        if (rope.ext_factor != 0.0f && factor > 1.0f) {
            // The ROPE converter already applies the default YaRN magnitude factor.
            rope.attn_factor /= 1.0f + 0.1f * log_multiplier * std::log(factor);
        }
        graph.configure_rope(rope);
        const auto positions = graph.build_inp_pos();
        graph.build_attn_inp_kv();
        const auto indices = graph.add_input("inp_kv_idx", ov::element::i32, {1, 1, 1, -1});
        const auto mask = graph.add_input("self_kq_mask", ov::element::f32, {1, 1, -1, -1});
        const auto out_ids = graph.build_inp_out_ids();
        auto tensors = graph.tensors();
        auto cur = graph.build_inp_embd(tensors.require("token_embd.weight"));
        const auto frequencies = tensors("rope_freqs.weight");
        GgufValue query_scale;
        if (temperature != 0.0f) {
            query_scale = graph.node("GGML_OP_CPY", {positions}, 0, {{"dst_type", ov::element::f32}});
            query_scale = graph.node("GGML_OP_SCALE", {query_scale}, 0, {{"scale", 1.0f / rope.n_ctx_orig}});
            // Token positions are nonnegative, so integer conversion computes floor.
            query_scale = graph.node("GGML_OP_CPY", {query_scale}, 0, {{"dst_type", ov::element::i32}});
            query_scale = graph.node("GGML_OP_CPY", {query_scale}, 0, {{"dst_type", ov::element::f32}});
            query_scale = graph.node("GGML_OP_SCALE", {query_scale}, 0, {{"bias", 1.0f}});
            query_scale = graph.node("GGML_OP_LOG", {query_scale});
            query_scale = graph.node("GGML_OP_SCALE", {query_scale}, 0, {{"scale", temperature}, {"bias", 1.0f}});
            query_scale = reshape(graph, query_scale, {0, -1, 1, 1});
        }
        for (int layer = 0; layer < layers; ++layer) {
            const auto name = "blk." + std::to_string(layer) + ".";
            const auto project = [&](const std::string& suffix, const GgufValue& input) {
                auto result = graph.node("GGML_OP_MUL_MAT", {tensors.require(name + suffix + ".weight"), input});
                if (auto bias = tensors(name + suffix + ".bias"))
                    result = graph.node("GGML_OP_ADD", {result, bias});
                return result;
            };
            auto norm = graph.build_norm(cur, tensors.require(name + "attn_norm.weight"), epsilon);
            auto q = reshape(graph, project("attn_q", norm), {0, -1, heads, head_size});
            auto k = reshape(graph, project("attn_k", norm), {0, -1, kv_heads, head_size});
            auto v = reshape(graph, project("attn_v", norm), {0, -1, kv_heads, head_size});
            const auto rotate = [&](const GgufValue& x) {
                std::vector<GgufValue> inputs{x, positions};
                if (frequencies)
                    inputs.push_back(frequencies);
                return graph.node("GGML_OP_ROPE", inputs, 0, {{"rope_config", rope}});
            };
            q = rotate(q);
            k = rotate(k);
            if (query_scale)
                q = graph.node("GGML_OP_MUL", {q, query_scale});
            const auto cache = [&](const std::string& kind, const GgufValue& update) {
                const auto input = graph.add_input("cache_" + kind + "_l" + std::to_string(layer),
                                                   ov::element::f16,
                                                   {1, -1, kv_heads, head_size});
                const auto output = graph.node("GGML_OP_SET_ROWS", {update, indices, input});
                graph.set_output(output);
                return output;
            };
            k = cache("k", k);
            v = cache("v", v);
            auto attention = graph.node("GGML_OP_FLASH_ATTN_EXT",
                                        {q, k, v, mask},
                                        100,
                                        {{"scale", 1.0f / std::sqrt(static_cast<float>(head_size))}});
            attention = project("attn_output", reshape(graph, attention, {0, 1, -1, int64_t(heads) * head_size}));
            if (layer == layers - 1) {
                attention = graph.node("GGML_OP_GET_ROWS", {attention, out_ids});
                cur = graph.node("GGML_OP_GET_ROWS", {cur, out_ids});
            }
            cur = graph.node("GGML_OP_ADD", {cur, attention});
            norm = graph.build_norm(cur, tensors.require(name + "ffn_norm.weight"), epsilon);
            auto gate = graph.node("GGML_UNARY_OP_SILU", {project("ffn_gate", norm)});
            auto ffn = graph.node("GGML_OP_MUL", {gate, project("ffn_up", norm)});
            cur = graph.node("GGML_OP_ADD", {cur, project("ffn_down", ffn)});
        }
        cur = graph.build_norm(cur, tensors.require("output_norm.weight"), epsilon);
        auto output_weight = tensors("output.weight");
        if (!output_weight)
            output_weight = tensors.require("token_embd.weight");
        graph.set_output(graph.node("GGML_OP_MUL_MAT", {output_weight, cur}));
        return graph.finish();
    }

private:
    BuildContext m_context;
};
}  // namespace

ov::frontend::gguf::ArchitectureDefinition devstral_decoder(const std::string& architecture) {
    OPENVINO_ASSERT(architecture == "llama" || architecture == "mistral3", "[Devstral] unsupported family");
    return {architecture, architecture, [](const ov::frontend::gguf::BuildContext& context) {
                return std::make_shared<DevstralBuilder>(context);
            }};
}
}  // namespace example
