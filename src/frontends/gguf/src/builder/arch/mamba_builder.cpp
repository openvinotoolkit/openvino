// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mamba_builder.hpp"

#include <cmath>
#include <limits>

#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {

// Mamba2 has a norm/mixer/residual stack, without attention or a feed-forward sublayer.
// Dense Nemotron-H interleaves those mixers with attention and ReLU-squared FFN layers.
// Reference: llama.cpp models/mamba.cpp and mamba-base.cpp::build_mamba2_layer.
class Mamba2Builder : public ModelBuilder {
public:
    explicit Mamba2Builder(const BuildContext& context) : g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        const auto positive = [&](const std::string& suffix) {
            const auto key = g.arch() + "." + suffix;
            const auto value = g.metadata().get_int(key);
            OPENVINO_ASSERT(value && *value > 0 && *value <= std::numeric_limits<int>::max(),
                            "[GGUF] Mamba2 requires positive metadata '",
                            key,
                            "'");
            return *value;
        };
        const auto layers = positive("block_count");
        const auto inner = positive("ssm.inner_size");
        const auto heads = positive("ssm.time_step_rank");
        const auto groups = positive("ssm.group_count");
        const auto state_size = positive("ssm.state_size");
        const auto kernel = positive("ssm.conv_kernel");
        OPENVINO_ASSERT(inner % heads == 0 && heads % groups == 0 && kernel > 1,
                        "[GGUF] Mamba2 requires inner_size divisible by heads, heads divisible by groups, "
                        "and conv_kernel > 1");
        const auto head_dim = inner / heads;
        const auto conv_dim = inner + 2 * groups * state_size;
        const auto eps_value = g.metadata().get_float(g.arch() + ".attention.layer_norm_rms_epsilon");
        OPENVINO_ASSERT(eps_value && std::isfinite(*eps_value) && *eps_value > 0,
                        "[GGUF] Mamba2 requires a positive RMS normalization epsilon");
        const auto eps = static_cast<float>(*eps_value);
        auto weights = g.tensors();
        const bool hybrid = g.arch() == "nemotron_h";
        if (hybrid) {
            OPENVINO_ASSERT(g.metadata().get_int(g.arch() + ".expert_count").value_or(0) == 0,
                            "[GGUF] Nemotron-H MoE is not supported by the dense builder");
            DecoderOptions options;
            options.rope_skip_period = 1;  // Nemotron-H attention has no positional embedding.
            g.configure_decoder(RopeMode::Normal, options);
            g.build_attn_inp_kv();
        }
        auto cur = g.build_inp_embd(weights.require("token_embd.weight"));
        const auto out_ids = g.build_inp_out_ids();
        ov::Tensor ids_tensor(ov::element::i32, {1, 1, 1, 1});
        ids_tensor.data<int32_t>()[0] = 0;
        const auto ids = g.add_constant("ssm_sequence_ids", ids_tensor);

        for (int64_t layer = 0; layer < layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto w = [&](const std::string& name) {
                return weights.require(prefix + name);
            };
            auto residual = cur;
            cur = g.build_norm(cur, w("attn_norm.weight"), eps);
            if (hybrid && !weights.has(prefix + "ssm_in.weight")) {
                if (weights.has(prefix + "attn_q.weight") || weights.has(prefix + "attn_qkv.weight")) {
                    cur = g.decoder_attention(static_cast<int>(layer), cur);
                } else {
                    cur = g.node("GGML_OP_MUL_MAT", {w("ffn_up.weight"), cur});
                    if (weights.has(prefix + "ffn_up.bias"))
                        cur = g.node("GGML_OP_ADD", {cur, w("ffn_up.bias")});
                    cur = g.node("GGML_OP_SQR", {g.node("GGML_UNARY_OP_RELU", {cur})});
                    cur = g.node("GGML_OP_MUL_MAT", {w("ffn_down.weight"), cur});
                    if (weights.has(prefix + "ffn_down.bias"))
                        cur = g.node("GGML_OP_ADD", {cur, w("ffn_down.bias")});
                }
            } else {
                auto projected = g.node("GGML_OP_MUL_MAT", {w("ssm_in.weight"), cur});
                const auto z = reshape(slice(projected, 0, inner), {0, -1, heads, head_dim});
                auto xbc = slice(projected, inner, conv_dim);
                const auto xbc_layout = xbc;
                auto dt = slice(projected, inner + conv_dim, heads);
                dt = g.node("GGML_OP_ADD", {dt, w("ssm_dt.bias")});

                const auto conv_state =
                    g.add_input("conv_state_l" + std::to_string(layer), ov::element::f32, {1, 1, conv_dim, kernel - 1});
                const auto window =
                    g.node("GGML_OP_CONCAT", {conv_state, g.node("GGML_OP_TRANSPOSE", {xbc})}, 0, {{"concat_axis", 0}});
                g.add_recurrent_state(conv_state, slice(window, -(kernel - 1), kernel - 1));
                xbc = g.node("GGML_OP_SSM_CONV", {window, w("ssm_conv1d.weight")});
                if (weights.has(prefix + "ssm_conv1d.bias"))
                    xbc = g.node("GGML_OP_ADD", {xbc, w("ssm_conv1d.bias")});
                xbc = g.node("GGML_UNARY_OP_SILU", {xbc});
                xbc = g.node("GGML_OP_VIEW", {xbc, xbc_layout}, 3);
                const auto x = reshape(slice(xbc, 0, inner), {0, -1, heads, head_dim});
                const auto b = reshape(slice(xbc, inner, groups * state_size), {0, -1, groups, state_size});
                const auto c =
                    reshape(slice(xbc, inner + groups * state_size, groups * state_size), {0, -1, groups, state_size});
                const auto state = g.add_input("ssm_state_l" + std::to_string(layer),
                                               ov::element::f32,
                                               {1, heads, head_dim, state_size});
                const auto scan = g.node("GGML_OP_SSM_SCAN", {state, x, dt, w("ssm_a"), b, c, ids});
                const auto state_elements = inner * state_size;
                g.add_recurrent_state(
                    state,
                    reshape(slice(scan, -state_elements, state_elements), {1, heads, head_dim, state_size}));
                auto y =
                    g.node("GGML_OP_VIEW", {scan, x}, 3, {{"view_slice", std::vector<int64_t>{3, 0, -state_elements}}});
                y = g.node("GGML_OP_ADD", {y, g.node("GGML_OP_MUL", {x, w("ssm_d")})});
                y = g.node("GGML_OP_MUL", {y, g.node("GGML_UNARY_OP_SILU", {z})});
                y = reshape(y, {0, -1, groups, inner / groups});
                y = g.build_norm(y, w("ssm_norm.weight"), eps);
                y = reshape(y, {0, 1, -1, inner});
                cur = g.node("GGML_OP_MUL_MAT", {w("ssm_out.weight"), y});
            }
            if (layer == layers - 1) {
                cur = g.node("GGML_OP_GET_ROWS", {cur, out_ids});
                residual = g.node("GGML_OP_GET_ROWS", {residual, out_ids});
            }
            cur = g.node("GGML_OP_ADD", {cur, residual});
        }
        cur = g.build_norm(cur, weights.require("output_norm.weight"), eps);
        const auto output =
            weights.has("output.weight") ? weights.require("output.weight") : weights.require("token_embd.weight");
        g.set_primary_output(g.node("GGML_OP_MUL_MAT", {output, cur}));
        return g.finish();
    }

private:
    GgufValue reshape(const GgufValue& value, const std::vector<int64_t>& shape) {
        return g.node("GGML_OP_RESHAPE", {value}, 6, {{"reshape_target", shape}, {"special_zero", true}});
    }

    GgufValue slice(const GgufValue& value, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {value}, 3, {{"view_slice", std::vector<int64_t>{3, start, length}}});
    }

    GgufGraphContext g;
};

}  // namespace

ArchitectureDefinition mamba2_architecture(const std::string& architecture) {
    return {architecture,
            architecture,
            [](const BuildContext& context) {
                return std::make_shared<Mamba2Builder>(context);
            },
            {},
            Maturity::Verified};
}

}  // namespace ov::frontend::gguf
