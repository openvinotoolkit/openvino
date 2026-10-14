// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lfm2_builder.hpp"

#include <limits>

#include "openvino/frontend/gguf/builder/graph_context.hpp"

namespace ov::frontend::gguf {
namespace {
class Lfm2Builder : public ModelBuilder {
public:
    explicit Lfm2Builder(const BuildContext& context) : g(context) {}

    std::shared_ptr<GgufGraph> build() override {
        const auto kernel = g.metadata().get_int(g.arch() + ".shortconv.l_cache");
        OPENVINO_ASSERT(kernel && *kernel > 1 && *kernel <= std::numeric_limits<int>::max(),
                        "[GGUF] LFM2 requires shortconv.l_cache > 1");
        OPENVINO_ASSERT(g.metadata().get_bool(g.arch() + ".attention.causal").value_or(true),
                        "[GGUF] noncausal LFM2 embedding models require a centered convolution builder");
        DecoderOptions options;
        options.normalize_expert_weights = true;
        const auto dims = g.configure_decoder(RopeMode::Neox, options);
        OPENVINO_ASSERT(dims.layers > 0 && dims.embedding > 0,
                        "[GGUF] LFM2 requires positive layer and embedding counts");
        g.build_inp_pos();
        g.build_attn_inp_kv();
        auto weights = g.tensors();
        auto cur = g.build_inp_embd(weights.require("token_embd.weight"));
        const auto out_ids = g.build_inp_out_ids();
        for (int layer = 0; layer < dims.layers; ++layer) {
            const auto prefix = "blk." + std::to_string(layer) + ".";
            const auto w = [&](const std::string& name) {
                return weights.require(prefix + name);
            };
            auto residual = cur;
            cur = g.build_norm(cur, w("attn_norm.weight"), dims.norm_epsilon);
            if (weights.has(prefix + "shortconv.in_proj.weight")) {
                const auto projected = g.node("GGML_OP_MUL_MAT", {w("shortconv.in_proj.weight"), cur});
                const auto b = slice(projected, 0, dims.embedding);
                const auto c = slice(projected, dims.embedding, dims.embedding);
                const auto x = slice(projected, 2 * int64_t{dims.embedding}, dims.embedding);
                const auto bx = g.node("GGML_OP_TRANSPOSE", {g.node("GGML_OP_MUL", {b, x})});
                const auto state = g.add_input("conv_state_l" + std::to_string(layer),
                                               ov::element::f32,
                                               {1, 1, dims.embedding, *kernel - 1});
                const auto window = g.node("GGML_OP_CONCAT", {state, bx}, 0, {{"concat_axis", 0}});
                g.add_recurrent_state(state, slice(window, -(*kernel - 1), *kernel - 1));
                const auto convolution =
                    g.node("GGML_OP_SSM_CONV", {window, w("shortconv.conv.weight")}, 0, {{"batch_major", true}});
                cur = g.node("GGML_OP_MUL_MAT",
                             {w("shortconv.out_proj.weight"), g.node("GGML_OP_MUL", {c, convolution})});
            } else {
                cur = g.decoder_attention(layer, cur);
            }
            if (layer == dims.layers - 1) {
                cur = g.node("GGML_OP_GET_ROWS", {cur, out_ids});
                residual = g.node("GGML_OP_GET_ROWS", {residual, out_ids});
            }
            cur = g.node("GGML_OP_ADD", {cur, residual});
            residual = cur;
            cur = g.build_norm(cur, w("ffn_norm.weight"), dims.norm_epsilon);
            cur = g.node("GGML_OP_ADD", {residual, g.decoder_ffn(layer, cur)});
        }
        cur = g.build_norm(cur, weights.require("token_embd_norm.weight"), dims.norm_epsilon);
        const auto output =
            weights.has("output.weight") ? weights.require("output.weight") : weights.require("token_embd.weight");
        g.set_primary_output(g.node("GGML_OP_MUL_MAT", {output, cur}));
        return g.finish();
    }

private:
    GgufValue slice(const GgufValue& value, int64_t start, int64_t length) {
        return g.node("GGML_OP_VIEW", {value}, 3, {{"view_slice", std::vector<int64_t>{3, start, length}}});
    }

    GgufGraphContext g;
};
}  // namespace

std::shared_ptr<ModelBuilder> make_lfm2_builder(const BuildContext& context) {
    return std::make_shared<Lfm2Builder>(context);
}
}  // namespace ov::frontend::gguf
